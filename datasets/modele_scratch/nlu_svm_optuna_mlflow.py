# nlu_svm_optuna_mlflow.py
"""
Modele NLU avec SVM + Embeddings + Optimisation Optuna + MLflow Tracking
Version corrigee avec validation croisee adaptative
"""

import numpy as np
import pickle
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow non installe. Installez: pip install mlflow")

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna non installe. Installez: pip install optuna")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers non installe. Installez: pip install sentence-transformers")


BEST_PARAMS_SVM = {
    'kernel': 'linear',
    'C': 0.5,
    'gamma': 'scale',
    'probability': True,
    'class_weight': 'balanced',
    'random_state': 42,
    'max_iter': 1000,
    'tol': 1e-3
}


class NLUSVMOptunaMLflow:
    
    def __init__(self, use_optuna: bool = True, use_mlflow: bool = True, 
                 experiment_name: str = "svm_telecom"):
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers requis")
        
        print("Chargement du modele d'embeddings...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Modele d'embeddings charge")
        
        self.model = None
        self.label_encoder = None
        self.classes_ = None
        self.entraine = False
        self.best_params = None
        self.use_optuna = use_optuna
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        
        if self.use_mlflow:
            os.makedirs("mlflow_logs", exist_ok=True)
            mlflow.set_tracking_uri("file:./mlflow_logs")
            try:
                mlflow.create_experiment(experiment_name)
            except:
                pass
            mlflow.set_experiment(experiment_name)
            print(f"MLflow active - Experience: {experiment_name}")
        
        self.stats = {
            "total_predictions": 0,
            "avg_confidence": 0,
            "train_accuracy": 0,
            "val_accuracy": 0,
            "test_accuracy": 0,
            "test_f1": 0,
            "cv_mean": 0,
            "cv_std": 0,
            "best_score": 0,
            "n_trials": 0,
            "optuna_study_time": 0,
            "pruned_trials": 0
        }
        
        self.BEST_PARAMS = BEST_PARAMS_SVM.copy()
    
    def _generer_embeddings(self, textes: List[str], verbose: bool = True) -> np.ndarray:
        if verbose:
            print(f"Generation embeddings pour {len(textes)} textes...")
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(textes), batch_size):
            batch = textes[i:i+batch_size]
            try:
                batch_emb = self.encoder.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                embeddings.extend(batch_emb)
            except Exception as e:
                print(f"Erreur sur batch {i}: {e}")
                embeddings.extend([np.zeros(384)] * len(batch))
            
            if verbose and i % 100 == 0 and i > 0:
                print(f"   {i}/{len(textes)} textes traites")
        
        return np.array(embeddings)
    
    def _objective(self, trial: optuna.Trial, X_emb: np.ndarray, y_enc: np.ndarray,
                   X_val_emb: np.ndarray = None, y_val_enc: np.ndarray = None) -> float:
        
        kernel = trial.suggest_categorical('kernel', ['linear', 'rbf'])
        C = trial.suggest_float('C', 0.01, 5.0, log=True)
        
        params = {
            'kernel': kernel,
            'C': C,
            'probability': True,
            'class_weight': 'balanced',
            'random_state': 42,
            'cache_size': 500,
            'max_iter': 1000,
            'tol': 1e-3
        }
        
        if kernel == 'rbf':
            params['gamma'] = trial.suggest_categorical('gamma', ['scale', 'auto', 0.01, 0.1])
        
        if X_val_emb is not None and y_val_enc is not None:
            model = SVC(**params)
            model.fit(X_emb, y_enc)
            y_pred = model.predict(X_val_emb)
            score = accuracy_score(y_val_enc, y_pred)
            trial.report(score, step=int(C * 10))
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score
        else:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_emb, y_enc)):
                X_tr, X_va = X_emb[train_idx], X_emb[val_idx]
                y_tr, y_va = y_enc[train_idx], y_enc[val_idx]
                model = SVC(**params)
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_va)
                scores.append(accuracy_score(y_va, y_pred))
                current_score = np.mean(scores)
                trial.report(current_score, step=fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return np.mean(scores)
    
    def optimiser_hyperparametres(self, X_train: List[str], y_train: List[str],
                                   X_val: List[str] = None, y_val: List[str] = None,
                                   n_trials: int = 30, timeout: int = 3600) -> Dict:
        
        print("\n" + "="*70)
        print("OPTUNA - OPTIMISATION DES HYPERPARAMETRES SVM")
        print("="*70)
        
        start_time = time.time()
        
        if self.use_mlflow:
            mlflow.start_run(run_name="optuna_optimization", nested=True)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("timeout", timeout)
        
        print("\nGeneration des embeddings...")
        X_emb = self._generer_embeddings(X_train, verbose=True)
        
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_
        
        print(f"Dimensions: {X_emb.shape}")
        print(f"Classes: {len(self.classes_)}")
        
        X_val_emb = None
        y_val_enc = None
        if X_val and y_val:
            X_val_emb = self._generer_embeddings(X_val, verbose=False)
            y_val_enc = self.label_encoder.transform(y_val)
            print(f"Validation: {len(X_val)} exemples")
        
        print(f"\nOptimisation avec {n_trials} essais...")
        
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
        
        study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner)
        study.optimize(
            lambda trial: self._objective(trial, X_emb, y_enc, X_val_emb, y_val_enc),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )
        
        self.stats['optuna_study_time'] = time.time() - start_time
        self.stats['n_trials'] = len(study.trials)
        self.stats['pruned_trials'] = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        self.best_params = study.best_params
        self.stats['best_score'] = study.best_value
        
        if self.use_mlflow:
            mlflow.log_metrics({
                "best_validation_score": self.stats['best_score'],
                "optuna_time": self.stats['optuna_study_time'],
                "n_trials": self.stats['n_trials'],
                "pruned_trials": self.stats['pruned_trials']
            })
            for param, value in self.best_params.items():
                mlflow.log_param(f"best_{param}", value)
            mlflow.end_run()
        
        print("\n" + "="*70)
        print("MEILLEURS HYPERPARAMETRES SVM TROUVES")
        print("="*70)
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        print(f"\nMeilleur score validation: {self.stats['best_score']*100:.2f}%")
        print(f"Temps optimisation: {self.stats['optuna_study_time']:.2f}s")
        
        return self.best_params
    
    def entrainer(self, X_train: List[str], y_train: List[str],
                  X_val: List[str] = None, y_val: List[str] = None,
                  optimiser: bool = False, n_trials: int = 30,
                  use_best_params: bool = True,
                  use_cross_validation: bool = False) -> Dict:
        
        print("\n" + "="*60)
        print("ENTRAINEMENT SVM + OPTUNA + MLflow")
        print("="*60)
        
        start_time = time.time()
        
        if self.use_mlflow:
            mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_param("model_type", "SVM_Optuna")
            mlflow.log_param("embeddings_model", "paraphrase-multilingual-MiniLM-L12-v2")
            mlflow.log_param("use_optuna", optimiser)
            mlflow.log_param("n_trials", n_trials)
        
        X_emb = self._generer_embeddings(X_train, verbose=True)
        
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_
        
        print(f"Dimensions: {X_emb.shape}")
        print(f"Classes: {len(self.classes_)}")
        
        if optimiser and self.use_optuna:
            params = self.optimiser_hyperparametres(X_train, y_train, X_val, y_val, n_trials)
        elif use_best_params:
            params = self.BEST_PARAMS.copy()
            print("\nUtilisation des meilleurs parametres pre-optimises:")
            for param, value in params.items():
                print(f"   {param}: {value}")
            self.best_params = params
        else:
            params = {
                'kernel': 'linear',
                'C': 0.5,
                'probability': True,
                'class_weight': 'balanced',
                'random_state': 42,
                'cache_size': 500,
                'max_iter': 1000,
                'tol': 1e-3
            }
            self.best_params = params
        
        if self.use_mlflow:
            for param, value in self.best_params.items():
                mlflow.log_param(param, value)
        
        print("\nEntrainement du modele final...")
        
        self.model = SVC(**self.best_params)
        self.model.fit(X_emb, y_enc)
        
        self.entraine = True
        self.train_time = time.time() - start_time
        
        y_pred_train = self.model.predict(X_emb)
        self.stats['train_accuracy'] = accuracy_score(y_enc, y_pred_train)
        
        # Validation croisee adaptative
        if use_cross_validation:
            print("\nValidation croisee...")
            class_counts = Counter(y_enc)
            min_samples = min(class_counts.values())
            
            n_splits = min(3, min_samples)
            if n_splits < 2:
                print(f"   Pas assez de donnees pour validation croisee (min_samples={min_samples})")
                self.stats['cv_mean'] = self.stats['train_accuracy']
                self.stats['cv_std'] = 0
            else:
                print(f"   Utilisation de {n_splits}-fold cross-validation")
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                cv_scores = []
                
                for fold, (train_idx, val_idx) in enumerate(skf.split(X_emb, y_enc)):
                    X_tr, X_va = X_emb[train_idx], X_emb[val_idx]
                    y_tr, y_va = y_enc[train_idx], y_enc[val_idx]
                    
                    model_cv = SVC(**self.best_params)
                    model_cv.fit(X_tr, y_tr)
                    y_pred_cv = model_cv.predict(X_va)
                    score = accuracy_score(y_va, y_pred_cv)
                    cv_scores.append(score)
                    print(f"   Fold {fold+1}: {score*100:.2f}%")
                
                self.stats['cv_mean'] = np.mean(cv_scores)
                self.stats['cv_std'] = np.std(cv_scores)
                print(f"   Mean CV accuracy: {self.stats['cv_mean']*100:.2f}% (+/- {self.stats['cv_std']*100:.2f}%)")
                
                if self.stats['train_accuracy'] - self.stats['cv_mean'] > 0.1:
                    print("\n   ATTENTION: Overfitting detecte!")
                    print(f"   Train accuracy: {self.stats['train_accuracy']*100:.2f}%")
                    print(f"   CV mean accuracy: {self.stats['cv_mean']*100:.2f}%")
                    print(f"   Ecart: {(self.stats['train_accuracy'] - self.stats['cv_mean'])*100:.2f}%")
                    print("   Suggestion: Reduire C ou utiliser kernel lineaire")
            
            if self.use_mlflow:
                mlflow.log_metrics({
                    "cv_mean": self.stats['cv_mean'],
                    "cv_std": self.stats['cv_std']
                })
        
        if X_val and y_val:
            X_val_emb = self._generer_embeddings(X_val, verbose=False)
            y_val_enc = self.label_encoder.transform(y_val)
            y_pred_val = self.model.predict(X_val_emb)
            self.stats['val_accuracy'] = accuracy_score(y_val_enc, y_pred_val)
            print(f"\nEntrainement termine")
            print(f"   Train accuracy: {self.stats['train_accuracy']*100:.2f}%")
            print(f"   Val accuracy: {self.stats['val_accuracy']*100:.2f}%")
            
            if self.use_mlflow:
                mlflow.log_metrics({
                    "train_accuracy": self.stats['train_accuracy'],
                    "val_accuracy": self.stats['val_accuracy']
                })
        else:
            print(f"\nEntrainement termine")
            print(f"   Train accuracy: {self.stats['train_accuracy']*100:.2f}%")
            if self.use_mlflow:
                mlflow.log_metric("train_accuracy", self.stats['train_accuracy'])
        
        print(f"   Temps total: {self.train_time:.2f}s")
        
        if self.use_mlflow:
            mlflow.log_metric("train_time", self.train_time)
            mlflow.end_run()
        
        return {
            'train_accuracy': self.stats['train_accuracy'],
            'val_accuracy': self.stats.get('val_accuracy', 0),
            'cv_mean': self.stats.get('cv_mean', 0),
            'cv_std': self.stats.get('cv_std', 0),
            'train_time': self.train_time,
            'best_params': self.best_params,
            'n_trials': self.stats['n_trials']
        }
    
    def evaluer_test(self, X_test: List[str], y_test: List[str]) -> Dict:
        
        if not self.entraine or self.model is None:
            raise Exception("Modele non entraine")
        
        print("\n" + "="*60)
        print("EVALUATION SUR LE JEU DE TEST")
        print("="*60)
        
        if self.use_mlflow:
            mlflow.start_run(run_name="test_evaluation", nested=True)
        
        y_pred = []
        confidences = []
        
        for texte in X_test:
            pred = self.predire(texte)
            y_pred.append(pred['intention'])
            confidences.append(pred['confiance'])
        
        self.stats['test_accuracy'] = accuracy_score(y_test, y_pred)
        self.stats['test_f1'] = f1_score(y_test, y_pred, average='weighted')
        self.stats['avg_confidence'] = sum(confidences) / len(confidences)
        
        print(f"Accuracy: {self.stats['test_accuracy']*100:.2f}%")
        print(f"F1-score: {self.stats['test_f1']*100:.2f}%")
        print(f"Confiance moyenne: {self.stats['avg_confidence']*100:.2f}%")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        if self.use_mlflow:
            mlflow.log_metrics({
                "test_accuracy": self.stats['test_accuracy'],
                "test_f1_score": self.stats['test_f1'],
                "avg_confidence": self.stats['avg_confidence']
            })
            mlflow.sklearn.log_model(
                self.model,
                "svm_model",
                registered_model_name="SVM_Telecom"
            )
            mlflow.end_run()
        
        return {
            'accuracy': self.stats['test_accuracy'],
            'f1_score': self.stats['test_f1'],
            'avg_confidence': self.stats['avg_confidence'],
            'y_pred': y_pred,
            'confidences': confidences
        }
    
    def predire(self, texte: str) -> Dict:
        """Predire l'intention d'un texte"""
        
        if not self.entraine or self.model is None:
            return {"intention": "inconnu", "confiance": 0.0, "source": "non_entraine"}
        
        # Generer l'embedding
        embedding = self.encoder.encode([texte], convert_to_numpy=True, normalize_embeddings=True)
        
        try:
            # Convertir en DataFrame pour eviter les warnings
            import pandas as pd
            X_pred = pd.DataFrame(embedding)
            
            probas = self.model.predict_proba(X_pred)[0]
            y_pred = np.argmax(probas)
            confiance = float(max(probas))
        except Exception as e:
            print(f"Erreur prediction: {e}")
            return {"intention": "inconnu", "confiance": 0.0, "source": "erreur"}
        
        intention = self.label_encoder.inverse_transform([y_pred])[0]
        
        self.stats["total_predictions"] += 1
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * (self.stats["total_predictions"] - 1) + confiance
        ) / self.stats["total_predictions"]
        
        return {
            "intention": intention,
            "confiance": confiance,
            "source": "svm_optuna_mlflow",
            "probas": {intent: float(p) for intent, p in zip(self.classes_, probas)}
        }
    
    def sauvegarder(self, chemin: str = "modele_scratch/modele_svm_optuna.pkl"):
        
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        
        params_path = chemin.replace('.pkl', '_best_params.json')
        with open(params_path, 'w', encoding='utf-8') as f:
            json.dump(self.best_params, f, indent=2, ensure_ascii=False)
        
        with open(chemin, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'best_params': self.best_params,
                'label_encoder': self.label_encoder,
                'classes_': self.classes_,
                'entraine': self.entraine,
                'stats': self.stats,
                'version': 'svm_optuna_mlflow_v2'
            }, f)
        
        print(f"\nModele SVM sauvegarde: {chemin}")
        print(f"   Intentions: {len(self.classes_)}")
        print(f"   Train accuracy: {self.stats['train_accuracy']*100:.1f}%")
    
    def charger(self, chemin: str = "modele_scratch/modele_svm_optuna.pkl"):
        
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"Fichier modele introuvable: {chemin}")
        
        with open(chemin, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.best_params = data.get('best_params', {})
        self.label_encoder = data['label_encoder']
        self.classes_ = data['classes_']
        self.entraine = data['entraine']
        self.stats = data.get('stats', self.stats)
        
        print(f"\nModele SVM charge: {chemin}")
        print(f"   Intentions: {len(self.classes_)}")
        if self.best_params:
            print(f"   Meilleurs parametres: {list(self.best_params.keys())[:5]}...")
    
    def afficher_stats(self):
        
        print("\n" + "="*60)
        print("STATISTIQUES SVM + OPTUNA + MLflow")
        print("="*60)
        print(f"Total predictions: {self.stats['total_predictions']}")
        print(f"Confiance moyenne: {self.stats['avg_confidence']*100:.1f}%")
        print(f"Train accuracy: {self.stats['train_accuracy']*100:.1f}%")
        if self.stats.get('val_accuracy', 0):
            print(f"Val accuracy: {self.stats['val_accuracy']*100:.1f}%")
        if self.stats.get('test_accuracy', 0):
            print(f"Test accuracy: {self.stats['test_accuracy']*100:.1f}%")
        if self.stats.get('cv_mean', 0):
            print(f"CV mean accuracy: {self.stats['cv_mean']*100:.1f}% (+/- {self.stats['cv_std']*100:.1f}%)")
        print(f"Meilleur score Optuna: {self.stats['best_score']*100:.1f}%")
        print(f"Nombre d essais: {self.stats['n_trials']}")
        print(f"Temps optimisation: {self.stats['optuna_study_time']:.2f}s")
        
        if self.best_params:
            print("\nMeilleurs hyperparametres:")
            for param, value in self.best_params.items():
                print(f"   {param}: {value}")


def mode_svm_optuna_mlflow():
    print("\n" + "="*60)
    print("MODE SVM + OPTUNA + MLflow")
    print("="*60)
    
    from data import DONNEES, diviser_donnees
    
    train, val, test = diviser_donnees(DONNEES)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    print(f"\nDonnees:")
    print(f"   Train: {len(X_train)} exemples")
    print(f"   Validation: {len(X_val)} exemples")
    print(f"   Test: {len(X_test)} exemples")
    
    model = NLUSVMOptunaMLflow(use_optuna=False, use_mlflow=True)
    
    print("\nUtilisation des meilleurs parametres pre-optimises")
    model.entrainer(
        X_train, y_train,
        X_val, y_val,
        optimiser=False,
        use_best_params=True,
        use_cross_validation=True
    )
    
    results = model.evaluer_test(X_test, y_test)
    model.sauvegarder("modele_scratch/modele_svm_optuna_mlflow.pkl")
    
    return model


if __name__ == "__main__":
    print("Test du modele SVM avec Optuna et MLflow")
    
    X_train = [
        "bonjour", "salut", "hello",
        "merci", "merci beaucoup", "je vous remercie",
        "au revoir", "bye", "a bientot",
        "mon offre", "quel est mon forfait", "mon abonnement",
        "mes recharges", "combien de recharges", "nombre de recharges"
    ]
    y_train = [
        "saluer", "saluer", "saluer",
        "remercier", "remercier", "remercier",
        "au_revoir", "au_revoir", "au_revoir",
        "voir_offre_commerciale", "voir_offre_commerciale", "voir_offre_commerciale",
        "voir_nbr_recharge", "voir_nbr_recharge", "voir_nbr_recharge"
    ]
    
    X_val = ["bonsoir", "merci infiniment", "a plus", "mon pack", "mes rechargements"]
    y_val = ["saluer", "remercier", "au_revoir", "voir_offre_commerciale", "voir_nbr_recharge"]
    
    X_test = ["coucou", "thanks", "ciao", "mon pack", "combien de recharges"]
    y_test = ["saluer", "remercier", "au_revoir", "voir_offre_commerciale", "voir_nbr_recharge"]
    
    model = NLUSVMOptunaMLflow(use_optuna=True, use_mlflow=True)
    model.entrainer(X_train, y_train, X_val, y_val, optimiser=False, use_best_params=True, use_cross_validation=True)
    model.evaluer_test(X_test, y_test)
    model.afficher_stats()
    
    print("\nPour voir les resultats MLflow:")
    print("   mlflow ui --backend-store-uri file:./mlflow_logs")