"""
Modèle NLU avec XGBoost + Embeddings + Optimisation Optuna + MLflow Tracking
Version complète et corrigée
"""

import numpy as np
import pickle
import os
import time
import json
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Imports conditionnels
try:
    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow non installé. Installez: pip install mlflow")

try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna non installé. Installez: pip install optuna")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    XGB_VERSION = xgb.__version__
    print(f"XGBoost version {XGB_VERSION} disponible")
except ImportError:
    XGB_AVAILABLE = False
    XGB_VERSION = None
    print("XGBoost non installé. Installez: pip install xgboost")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("sentence-transformers non installé. Installez: pip install sentence-transformers")


# Meilleurs paramètres pré-optimisés
BEST_PARAMS_XGBOOST = {
    'n_estimators': 350,
    'max_depth': 4,
    'learning_rate': 0.11063299396152436,
    'reg_alpha': 0.19681710008345749,
    'reg_lambda': 0.8525934539003075,
    'gamma': 0.13360050908311025,
    'min_child_weight': 2,
    'subsample': 0.7337947656998619,
    'colsample_bytree': 0.7702459242704259,
    'colsample_bylevel': 0.8861211984850834,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss',
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}


class NLUXGBoostOptunaMLflow:
    """
    Modèle XGBoost pour NLU avec embeddings, Optuna et MLflow
    """
    
    def __init__(self, use_optuna: bool = True, use_mlflow: bool = True, 
                 experiment_name: str = "xgboost_telecom"):
        """
        Initialise le modèle XGBoost
        
        Args:
            use_optuna: Activer l'optimisation Optuna
            use_mlflow: Activer le tracking MLflow
            experiment_name: Nom de l'expérience MLflow
        """
        if not XGB_AVAILABLE:
            raise ImportError("XGBoost requis. Installez: pip install xgboost")
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers requis. Installez: pip install sentence-transformers")
        
        print("Chargement du modèle d'embeddings...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Modèle d'embeddings chargé")
        
        self.model = None
        self.label_encoder = None
        self.classes_ = None
        self.entraine = False
        self.best_params = None
        self.use_optuna = use_optuna and OPTUNA_AVAILABLE
        self.use_mlflow = use_mlflow and MLFLOW_AVAILABLE
        self.experiment_name = experiment_name
        
        # Configuration MLflow
        if self.use_mlflow:
            os.makedirs("mlflow_logs", exist_ok=True)
            mlflow.set_tracking_uri("file:./mlflow_logs")
            try:
                mlflow.create_experiment(experiment_name)
            except:
                pass
            mlflow.set_experiment(experiment_name)
            print(f"MLflow actif - Expérience: {experiment_name}")
        
        # Statistiques
        self.stats = {
            "total_predictions": 0,
            "avg_confidence": 0,
            "train_accuracy": 0,
            "val_accuracy": 0,
            "test_accuracy": 0,
            "test_f1": 0,
            "best_score": 0,
            "n_trials": 0,
            "optuna_study_time": 0,
            "pruned_trials": 0,
            "training_time": 0
        }
        
        self.BEST_PARAMS = BEST_PARAMS_XGBOOST.copy()
    
    def _generer_embeddings(self, textes: List[str], verbose: bool = True) -> np.ndarray:
        """
        Génère les embeddings pour une liste de textes
        
        Args:
            textes: Liste des textes
            verbose: Afficher la progression
        
        Returns:
            Tableau numpy des embeddings
        """
        if verbose:
            print(f"Génération embeddings pour {len(textes)} textes...")
        
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
                print(f"   {i}/{len(textes)} textes traités")
        
        return np.array(embeddings)
    
    def _objective(self, trial: optuna.Trial, X_emb: np.ndarray, y_enc: np.ndarray,
                   X_val_emb: np.ndarray = None, y_val_enc: np.ndarray = None) -> float:
        """
        Fonction objectif pour Optuna
        
        Args:
            trial: Essai Optuna
            X_emb: Embeddings d'entraînement
            y_enc: Labels encodés
            X_val_emb: Embeddings de validation
            y_val_enc: Labels de validation
        
        Returns:
            Score de validation
        """
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.03, 0.3, log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.5),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.5),
            'gamma': trial.suggest_float('gamma', 0.0, 0.4),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 8),
            'subsample': trial.suggest_float('subsample', 0.6, 0.95),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 0.95),
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
        if XGB_VERSION >= '1.5.0':
            params['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.6, 0.95)
        
        if X_val_emb is not None and y_val_enc is not None:
            model = xgb.XGBClassifier(**params)
            try:
                model.fit(X_emb, y_enc, eval_set=[(X_val_emb, y_val_enc)],
                         early_stopping_rounds=15, verbose=False)
            except TypeError:
                model.fit(X_emb, y_enc, verbose=False)
            
            y_pred = model.predict(X_val_emb)
            score = accuracy_score(y_val_enc, y_pred)
            
            trial.report(score, step=params['n_estimators'])
            if trial.should_prune():
                raise optuna.TrialPruned()
            return score
        else:
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_emb, y_enc)):
                X_tr, X_va = X_emb[train_idx], X_emb[val_idx]
                y_tr, y_va = y_enc[train_idx], y_enc[val_idx]
                model = xgb.XGBClassifier(**params)
                try:
                    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                             early_stopping_rounds=15, verbose=False)
                except TypeError:
                    model.fit(X_tr, y_tr, verbose=False)
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
        """
        Optimise les hyperparamètres avec Optuna
        
        Args:
            X_train: Textes d'entraînement
            y_train: Labels
            X_val: Textes de validation
            y_val: Labels de validation
            n_trials: Nombre d'essais
            timeout: Timeout en secondes
        
        Returns:
            Meilleurs paramètres
        """
        if not self.use_optuna:
            print("Optuna désactivé")
            return self.BEST_PARAMS
        
        print("\n" + "="*70)
        print("OPTUNA - OPTIMISATION DES HYPERPARAMETRES XGBOOST")
        print("="*70)
        
        start_time = time.time()
        
        if self.use_mlflow:
            mlflow.start_run(run_name="optuna_optimization", nested=True)
            mlflow.log_param("n_trials", n_trials)
            mlflow.log_param("timeout", timeout)
        
        print("\nGénération des embeddings...")
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
        
        # Ajouter les paramètres fixes
        fixed_params = {
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        self.best_params.update(fixed_params)
        
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
        print("MEILLEURS HYPERPARAMETRES TROUVES")
        print("="*70)
        for param, value in self.best_params.items():
            print(f"   {param}: {value}")
        
        print(f"\nMeilleur score validation: {self.stats['best_score']*100:.2f}%")
        print(f"Temps optimisation: {self.stats['optuna_study_time']:.2f}s")
        
        return self.best_params
    
    def entrainer(self, X_train: List[str], y_train: List[str],
                  X_val: List[str] = None, y_val: List[str] = None,
                  optimiser: bool = False, n_trials: int = 30,
                  use_best_params: bool = True) -> Dict:
        """
        Entraîne le modèle XGBoost
        
        Args:
            X_train: Textes d'entraînement
            y_train: Labels
            X_val: Textes de validation
            y_val: Labels de validation
            optimiser: Activer l'optimisation Optuna
            n_trials: Nombre d'essais Optuna
            use_best_params: Utiliser les meilleurs paramètres pré-optimisés
        
        Returns:
            Dictionnaire des métriques
        """
        print("\n" + "="*60)
        print("ENTRAINEMENT XGBOOST + OPTUNA + MLflow")
        print("="*60)
        
        start_time = time.time()
        
        if self.use_mlflow:
            mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            mlflow.log_param("model_type", "XGBoost_Optuna")
            mlflow.log_param("embeddings_model", "paraphrase-multilingual-MiniLM-L12-v2")
            mlflow.log_param("use_optuna", optimiser)
            mlflow.log_param("n_trials", n_trials)
        
        # Génération des embeddings
        X_emb = self._generer_embeddings(X_train, verbose=True)
        
        # Encodage des labels
        self.label_encoder = LabelEncoder()
        y_enc = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_
        
        print(f"Dimensions: {X_emb.shape}")
        print(f"Classes: {len(self.classes_)}")
        
        # Optimisation ou chargement des paramètres
        if optimiser and self.use_optuna:
            params = self.optimiser_hyperparametres(X_train, y_train, X_val, y_val, n_trials)
        elif use_best_params:
            params = self.BEST_PARAMS.copy()
            print("\nUtilisation des meilleurs paramètres pré-optimisés:")
            for param, value in list(params.items())[:10]:
                print(f"   {param}: {value}")
            self.best_params = params
        else:
            params = {
                'n_estimators': 200, 'max_depth': 5, 'learning_rate': 0.1,
                'reg_alpha': 0.5, 'reg_lambda': 1.5, 'gamma': 0.1,
                'min_child_weight': 3, 'subsample': 0.8, 'colsample_bytree': 0.8,
                'objective': 'multi:softprob', 'eval_metric': 'mlogloss',
                'random_state': 42, 'n_jobs': -1, 'verbosity': 0
            }
            if XGB_VERSION >= '1.5.0':
                params['colsample_bylevel'] = 0.8
            self.best_params = params
        
        if self.use_mlflow:
            for param, value in self.best_params.items():
                mlflow.log_param(param, value)
        
        print("\nEntraînement du modèle final...")
        
        self.model = xgb.XGBClassifier(**self.best_params)
        
        eval_set = None
        if X_val and y_val:
            X_val_emb = self._generer_embeddings(X_val, verbose=False)
            y_val_enc = self.label_encoder.transform(y_val)
            eval_set = [(X_val_emb, y_val_enc)]
        
        try:
            self.model.fit(X_emb, y_enc, eval_set=eval_set,
                          early_stopping_rounds=20 if eval_set else None,
                          verbose=True)
        except TypeError:
            self.model.fit(X_emb, y_enc, verbose=True)
        
        self.entraine = True
        self.stats['training_time'] = time.time() - start_time
        
        # Évaluation sur train
        y_pred_train = self.model.predict(X_emb)
        self.stats['train_accuracy'] = accuracy_score(y_enc, y_pred_train)
        
        # Évaluation sur validation
        if X_val and y_val:
            y_pred_val = self.model.predict(X_val_emb)
            self.stats['val_accuracy'] = accuracy_score(y_val_enc, y_pred_val)
            print(f"\nEntraînement terminé")
            print(f"   Train accuracy: {self.stats['train_accuracy']*100:.2f}%")
            print(f"   Val accuracy: {self.stats['val_accuracy']*100:.2f}%")
            
            if self.use_mlflow:
                mlflow.log_metrics({
                    "train_accuracy": self.stats['train_accuracy'],
                    "val_accuracy": self.stats['val_accuracy']
                })
        else:
            print(f"\nEntraînement terminé")
            print(f"   Train accuracy: {self.stats['train_accuracy']*100:.2f}%")
            if self.use_mlflow:
                mlflow.log_metric("train_accuracy", self.stats['train_accuracy'])
        
        print(f"   Temps total: {self.stats['training_time']:.2f}s")
        
        if self.use_mlflow:
            mlflow.log_metric("training_time", self.stats['training_time'])
            mlflow.end_run()
        
        return {
            'train_accuracy': self.stats['train_accuracy'],
            'val_accuracy': self.stats.get('val_accuracy', 0),
            'training_time': self.stats['training_time'],
            'best_params': self.best_params,
            'n_trials': self.stats['n_trials']
        }
    
    def evaluer_test(self, X_test: List[str], y_test: List[str]) -> Dict:
        """
        Évalue le modèle sur un jeu de test
        
        Args:
            X_test: Textes de test
            y_test: Labels
        
        Returns:
            Dictionnaire des métriques
        """
        if not self.entraine or self.model is None:
            raise Exception("Modèle non entraîné")
        
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
                "xgboost_model",
                registered_model_name="XGBoost_Telecom"
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
        """
        Prédit l'intention d'un texte
        
        Args:
            texte: Texte à analyser
        
        Returns:
            Dictionnaire avec l'intention et la confiance
        """
        if not self.entraine or self.model is None:
            return {"intention": "inconnu", "confiance": 0.0, "source": "non_entraine"}
        
        # Générer l'embedding
        embedding = self.encoder.encode([texte], convert_to_numpy=True, normalize_embeddings=True)
        
        try:
            probas = self.model.predict_proba(embedding)[0]
            y_pred = np.argmax(probas)
            confiance = float(max(probas))
        except Exception as e:
            print(f"Erreur prédiction: {e}")
            return {"intention": "inconnu", "confiance": 0.0, "source": "erreur"}
        
        intention = self.label_encoder.inverse_transform([y_pred])[0]
        
        self.stats["total_predictions"] += 1
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * (self.stats["total_predictions"] - 1) + confiance
        ) / self.stats["total_predictions"]
        
        return {
            "intention": intention,
            "confiance": confiance,
            "source": "xgboost_optuna_mlflow",
            "probas": {intent: float(p) for intent, p in zip(self.classes_, probas)}
        }
    
    def sauvegarder(self, chemin: str = "modele_scratch/modele_xgboost_optuna.pkl"):
        """
        Sauvegarde le modèle
        
        Args:
            chemin: Chemin de sauvegarde
        """
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
                'xgboost_version': XGB_VERSION,
                'version': 'xgboost_optuna_mlflow_v1'
            }, f)
        
        print(f"\nModèle sauvegardé: {chemin}")
        print(f"   Intentions: {len(self.classes_)}")
        print(f"   Train accuracy: {self.stats['train_accuracy']*100:.1f}%")
    
    def charger(self, chemin: str = "modele_scratch/modele_xgboost_optuna.pkl"):
        """
        Charge un modèle sauvegardé
        
        Args:
            chemin: Chemin du modèle
        """
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"Fichier modèle introuvable: {chemin}")
        
        with open(chemin, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.best_params = data.get('best_params', {})
        self.label_encoder = data['label_encoder']
        self.classes_ = data['classes_']
        self.entraine = data['entraine']
        self.stats = data.get('stats', self.stats)
        
        print(f"\nModèle chargé: {chemin}")
        print(f"   Intentions: {len(self.classes_)}")
        if self.best_params:
            print(f"   Meilleurs paramètres: {list(self.best_params.keys())[:5]}...")
    
    def afficher_stats(self):
        """Affiche les statistiques du modèle"""
        print("\n" + "="*60)
        print("STATISTIQUES XGBOOST + OPTUNA + MLflow")
        print("="*60)
        print(f"Total predictions: {self.stats['total_predictions']}")
        print(f"Confiance moyenne: {self.stats['avg_confidence']*100:.1f}%")
        print(f"Train accuracy: {self.stats['train_accuracy']*100:.1f}%")
        if self.stats.get('val_accuracy', 0):
            print(f"Val accuracy: {self.stats['val_accuracy']*100:.1f}%")
        if self.stats.get('test_accuracy', 0):
            print(f"Test accuracy: {self.stats['test_accuracy']*100:.1f}%")
        print(f"Meilleur score Optuna: {self.stats['best_score']*100:.1f}%")
        print(f"Nombre d essais: {self.stats['n_trials']}")
        print(f"Temps optimisation: {self.stats['optuna_study_time']:.2f}s")
        
        if self.best_params:
            print("\nMeilleurs hyperparamètres:")
            for param, value in list(self.best_params.items())[:12]:
                print(f"   {param}: {value}")


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================

def entrainer_modele_xgboost():
    """
    Fonction principale pour entraîner le modèle XGBoost
    """
    print("\n" + "="*60)
    print("ENTRAINEMENT XGBOOST AVEC OPTUNA ET MLFLOW")
    print("="*60)
    
    try:
        from data import DONNEES, diviser_donnees
    except ImportError:
        print("Erreur: Impossible d'importer les données")
        print("Assurez-vous que le fichier data.py est présent")
        return None
    
    train, val, test = diviser_donnees(DONNEES)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    print(f"\nDonnées:")
    print(f"   Train: {len(X_train)} exemples")
    print(f"   Validation: {len(X_val)} exemples")
    print(f"   Test: {len(X_test)} exemples")
    print(f"   Intentions: {len(set(y_train))} différentes")
    
    # Création du modèle
    model = NLUXGBoostOptunaMLflow(
        use_optuna=False,  # Mettre True pour l'optimisation
        use_mlflow=True,
        experiment_name="xgboost_full"
    )
    
    # Entraînement
    print("\n" + "="*60)
    print("DÉBUT DE L'ENTRAÎNEMENT...")
    print("="*60)
    
    results = model.entrainer(
        X_train, y_train,
        X_val, y_val,
        optimiser=False,
        use_best_params=True
    )
    
    # Évaluation
    test_results = model.evaluer_test(X_test, y_test)
    
    # Sauvegarde
    model.sauvegarder("modele_scratch/modele_xgboost_optuna_mlflow.pkl")
    
    # Affichage des stats
    model.afficher_stats()
    
    print("\n" + "="*60)
    print("ENTRAÎNEMENT TERMINÉ AVEC SUCCÈS")
    print("="*60)
    print(f"Résultats finaux:")
    print(f"  Train accuracy: {results['train_accuracy']*100:.2f}%")
    print(f"  Val accuracy: {results['val_accuracy']*100:.2f}%")
    print(f"  Test accuracy: {test_results['accuracy']*100:.2f}%")
    
    return model


# ============================================================
# TEST RAPIDE
# ============================================================

if __name__ == "__main__":
    print("Test du modèle XGBoost avec Optuna et MLflow")
    
    # Données de test
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
    
    # Test du modèle
    model = NLUXGBoostOptunaMLflow(use_optuna=False, use_mlflow=True)
    model.entrainer(X_train, y_train, X_val, y_val, optimiser=False, use_best_params=True)
    model.evaluer_test(X_test, y_test)
    model.afficher_stats()
    
    print("\nPour voir les résultats MLflow:")
    print("   mlflow ui --backend-store-uri file:./mlflow_logs")