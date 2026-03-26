# ============================================================
# main.py - Chatbot Tunisie Telecom avec SVM + XGBoost + Random Forest + Optuna
# Version : 9.0 - Support multi-modèles complet
# ============================================================

import argparse
import sys
import os
import json
from datetime import datetime

# Ajouter le répertoire courant au path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from ner import NER
from dialogue import GestionnaireDialogue
from data import DONNEES, diviser_donnees, afficher_statistiques

# Vérifier disponibilité des packages optionnels
SENTENCE_TRANSFORMERS_AVAILABLE = False
OPTUNA_AVAILABLE = False
XGB_AVAILABLE = False
RANDOMFOREST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers non installé")
    print("   Installez: pip install sentence-transformers")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ optuna non installé")
    print("   Installez: pip install optuna")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("⚠️ xgboost non installé")
    print("   Installez: pip install xgboost")

try:
    from sklearn.ensemble import RandomForestClassifier
    RANDOMFOREST_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn non installé")
    print("   Installez: pip install scikit-learn")


# ============================================================
# MODÈLE SVM + EMBEDDINGS (original)
# ============================================================

class NLUEmbeddings:
    """
    Modèle NLU basé sur Embeddings sémantiques + SVM
    """
    
    def __init__(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers requis")
        
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        self.SVC = SVC
        self.LabelEncoder = LabelEncoder
        self.np = np
        
        self.svm = SVC(
            kernel='linear',
            probability=True,
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        
        self.label_encoder = LabelEncoder()
        self.entraine = False
        self.embeddings_dim = 384
        
        self.stats = {
            "total_predictions": 0,
            "avg_confidence": 0,
            "intentions_counts": {}
        }
    
    def _generer_embeddings(self, textes, verbose=True):
        import numpy as np
        if verbose:
            print(f"Génération embeddings pour {len(textes)} textes...")
        
        embeddings = []
        batch_size = 32
        
        for i in range(0, len(textes), batch_size):
            batch = textes[i:i+batch_size]
            batch_emb = self.encoder.encode(
                batch,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            embeddings.extend(batch_emb)
            
            if verbose and i % 100 == 0 and i > 0:
                print(f"   {i}/{len(textes)} textes traités")
        
        return np.array(embeddings)
    
    def entrainer(self, X_train, y_train):
        print("\n" + "="*60)
        print("ENTRAINEMENT SVM SUR EMBEDDINGS")
        print("="*60)
        
        X_emb = self._generer_embeddings(X_train, verbose=True)
        y_enc = self.label_encoder.fit_transform(y_train)
        
        print(f"Dimensions: {X_emb.shape}")
        print(f"Intentions: {len(self.label_encoder.classes_)}")
        print("Entraînement SVM...")
        
        self.svm.fit(X_emb, y_enc)
        self.entraine = True
        
        score = self.svm.score(X_emb, y_enc)
        print(f"Entraînement terminé")
        print(f"   Accuracy train: {score*100:.2f}%")
        
        return score
    
    def valider(self, X_val, y_val):
        if not self.entraine:
            raise Exception("Modèle non entraîné")
        
        X_emb = self._generer_embeddings(X_val, verbose=False)
        y_enc = self.label_encoder.transform(y_val)
        y_pred = self.svm.predict(X_emb)
        
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_enc, y_pred)
        f1 = f1_score(y_enc, y_pred, average='weighted')
        
        print(f"Validation:")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   F1-Score: {f1*100:.2f}%")
        
        return accuracy, f1
    
    def predire(self, texte):
        if not self.entraine:
            return {"intention": "inconnu", "confiance": 0.0, "source": "non_entraine"}
        
        embedding = self.encoder.encode([texte], convert_to_numpy=True, normalize_embeddings=True)
        
        y_pred = self.svm.predict(embedding)[0]
        probas = self.svm.predict_proba(embedding)[0]
        confiance = float(max(probas))
        
        intention = self.label_encoder.inverse_transform([y_pred])[0]
        
        self.stats["total_predictions"] += 1
        self.stats["avg_confidence"] = (
            self.stats["avg_confidence"] * (self.stats["total_predictions"] - 1) + confiance
        ) / self.stats["total_predictions"]
        self.stats["intentions_counts"][intention] = self.stats["intentions_counts"].get(intention, 0) + 1
        
        return {
            "intention": intention,
            "confiance": confiance,
            "source": "svm_embeddings",
            "probas": {intent: float(p) for intent, p in zip(self.label_encoder.classes_, probas)}
        }
    
    def sauvegarder(self, chemin=None):
        import pickle
        import os
        chemin = chemin or config.CHEMIN_MODELE_NLU
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        
        with open(chemin, 'wb') as f:
            pickle.dump({
                'svm': self.svm,
                'label_encoder': self.label_encoder,
                'entraine': self.entraine,
                'stats': self.stats,
                'version': '9.0'
            }, f)
        
        print(f"Modèle SVM sauvegardé: {chemin}")
    
    def charger(self, chemin=None):
        import pickle
        chemin = chemin or config.CHEMIN_MODELE_NLU
        
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"Fichier modèle introuvable: {chemin}")
        
        with open(chemin, 'rb') as f:
            data = pickle.load(f)
        
        self.svm = data['svm']
        self.label_encoder = data['label_encoder']
        self.entraine = data['entraine']
        self.stats = data.get('stats', self.stats)
        
        print(f"Modèle SVM chargé: {chemin}")
    
    def afficher_stats(self):
        print("\n" + "="*60)
        print("STATISTIQUES SVM + EMBEDDINGS")
        print("="*60)
        print(f"Total prédictions: {self.stats['total_predictions']}")
        print(f"Confiance moyenne: {self.stats['avg_confidence']*100:.1f}%")
        print(f"Intentions disponibles: {len(self.label_encoder.classes_)}")


# ============================================================
# CHATBOT PRINCIPAL
# ============================================================

class ChatbotTunisieTelecom:
    """Chatbot principal avec SVM + Embeddings"""
    
    def __init__(self):
        self.nlu = None
        self.ner = NER()
        self.dialogue = None
        self.modele_charge = False
        self.initialise = False
    
    def initialiser(self):
        if self.initialise:
            return
        
        try:
            self.nlu = NLUEmbeddings()
        except ImportError as e:
            print(f"Erreur: {e}")
            sys.exit(1)
        
        self.dialogue = GestionnaireDialogue(modele_nlu=self.nlu)
        self.initialise = True
    
    def charger_modeles(self):
        self.initialiser()
        try:
            self.nlu.charger()
            self.modele_charge = True
            return True
        except FileNotFoundError:
            print("Aucun modèle trouvé. Lancez d'abord l'entraînement:")
            print("   python main.py --mode train_svm")
            return False
        except Exception as e:
            print(f"Erreur chargement: {e}")
            return False
    
    def repondre(self, message):
        if not self.modele_charge:
            return {
                "reponse": "Modèle non chargé. Veuillez d'abord entraîner le modèle.",
                "erreur": True,
                "intention": None,
                "confiance": 0
            }
        
        entites = self.ner.extraire(message)
        resultat_nlu = self.nlu.predire(message)
        
        return self.dialogue.traiter(
            message,
            resultat_nlu["intention"],
            resultat_nlu["confiance"],
            entites
        )


# ============================================================
# FONCTIONS D'ENTRAÎNEMENT POUR CHAQUE MODÈLE
# ============================================================

def entrainer_modele_svm():
    """Entraîne le modèle SVM"""
    print("\n" + "="*60)
    print("ENTRAINEMENT SVM SUR LES 5 FICHIERS CSV")
    print("="*60)
    
    train, val, test = diviser_donnees(DONNEES)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    print(f"\nDonnées:")
    print(f"  Train: {len(X_train)} exemples")
    print(f"  Validation: {len(X_val)} exemples")
    print(f"  Test: {len(X_test)} exemples")
    
    from collections import Counter
    print(f"  Intentions: {len(Counter(y_train))} différentes")
    
    from nlu_svm_optuna_mlflow import NLUSVMOptunaMLflow, BEST_PARAMS_SVM
    
    model = NLUSVMOptunaMLflow(
        use_optuna=False,
        use_mlflow=True,
        experiment_name="svm_full"
    )
    
    print("\nParamètres SVM utilisés:")
    for k, v in BEST_PARAMS_SVM.items():
        print(f"  {k}: {v}")
    
    print("\nDébut de l'entraînement...")
    model.entrainer(
        X_train, y_train,
        X_val, y_val,
        optimiser=False,
        use_best_params=True,
        use_cross_validation=True
    )
    
    print("\nÉvaluation sur test...")
    results = model.evaluer_test(X_test, y_test)
    
    model.sauvegarder("modele_scratch/modele_svm_full.pkl")
    
    print(f"\nRésultats finaux:")
    print(f"  Train accuracy: {model.stats['train_accuracy']*100:.2f}%")
    print(f"  Val accuracy: {model.stats.get('val_accuracy', 0)*100:.2f}%")
    print(f"  Test accuracy: {results['accuracy']*100:.2f}%")
    
    return model


def entrainer_modele_xgboost():
    """Entraîne le modèle XGBoost"""
    print("\n" + "="*60)
    print("ENTRAINEMENT XGBOOST SUR LES 5 FICHIERS CSV")
    print("="*60)
    
    if not XGB_AVAILABLE:
        print("XGBoost non installé. Installez: pip install xgboost")
        return None
    
    train, val, test = diviser_donnees(DONNEES)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    print(f"\nDonnées:")
    print(f"  Train: {len(X_train)} exemples")
    print(f"  Validation: {len(X_val)} exemples")
    print(f"  Test: {len(X_test)} exemples")
    
    from collections import Counter
    print(f"  Intentions: {len(Counter(y_train))} différentes")
    
    from nlu_xgboost_optuna_mlflow import NLUXGBoostOptunaMLflow, BEST_PARAMS_XGBOOST
    
    model = NLUXGBoostOptunaMLflow(
        use_optuna=False,
        use_mlflow=True,
        experiment_name="xgboost_full"
    )
    
    print("\nParamètres XGBoost utilisés:")
    for k, v in list(BEST_PARAMS_XGBOOST.items())[:10]:
        print(f"  {k}: {v}")
    
    print("\nDébut de l'entraînement...")
    model.entrainer(
        X_train, y_train,
        X_val, y_val,
        optimiser=False,
        use_best_params=True
    )
    
    print("\nÉvaluation sur test...")
    results = model.evaluer_test(X_test, y_test)
    
    model.sauvegarder("modele_scratch/modele_xgboost_full.pkl")
    
    print(f"\nRésultats finaux:")
    print(f"  Train accuracy: {model.stats['train_accuracy']*100:.2f}%")
    print(f"  Val accuracy: {model.stats.get('val_accuracy', 0)*100:.2f}%")
    print(f"  Test accuracy: {results['accuracy']*100:.2f}%")
    
    return model
def entrainer_modele_randomforest():
    """Entraîne le modèle Random Forest (remplace LightGBM)"""
    print("\n" + "="*60)
    print("ENTRAINEMENT RANDOM FOREST SUR LES 5 FICHIERS CSV")
    print("="*60)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("sentence-transformers non installé. Installez: pip install sentence-transformers")
        return None
    
    train, val, test = diviser_donnees(DONNEES)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    print(f"\nDonnées:")
    print(f"  Train: {len(X_train)} exemples")
    print(f"  Validation: {len(X_val)} exemples")
    print(f"  Test: {len(X_test)} exemples")
    
    from collections import Counter
    class_counts = Counter(y_train)
    print(f"  Intentions: {len(class_counts)} différentes")
    print(f"  Classes avec peu d'exemples (<30): {sum(1 for c in class_counts.values() if c < 30)}")
    
    # Importer Random Forest
    try:
        from nlu_randomforest_optuna_mlflow import NLURandomForestMLflow, BEST_PARAMS_RF
        print("✅ Import du module Random Forest réussi")
    except ImportError as e:
        print(f"❌ Erreur import Random Forest: {e}")
        print("Vérifiez que le fichier nlu_randomforest_optuna_mlflow.py existe")
        return None
    
    # MODIFICATION ICI : Ajout de regularization_level='moderate'

# Dans entrainer_modele_randomforest()
    model = NLURandomForestMLflow(
    use_mlflow=True,
    experiment_name="randomforest_full",
    regularization_level='optimal'  # ← MEILLEUR RÉSULTAT (91.4% test)
)
    
    print("\nParamètres Random Forest utilisés:")
    for k, v in list(BEST_PARAMS_RF.items())[:10]:
        print(f"  {k}: {v}")
    
    print("\nDébut de l'entraînement...")
    results = model.entrainer(
        X_train, y_train,
        X_val, y_val,
        use_best_params=True,
        use_cross_validation=True
    )
    
    print("\nÉvaluation sur test...")
    test_results = model.evaluer_test(X_test, y_test)
    
    model.sauvegarder("modele_scratch/modele_randomforest_full.pkl")
    
    print(f"\n" + "="*60)
    print("RÉSULTATS FINAUX - RANDOM FOREST")
    print("="*60)
    print(f"  Train accuracy: {results['train_accuracy']*100:.2f}%")
    print(f"  Train F1: {results['train_f1']*100:.2f}%")
    print(f"  Val accuracy: {results.get('val_accuracy', 0)*100:.2f}%")
    print(f"  Val F1: {results.get('val_f1', 0)*100:.2f}%")
    print(f"  Test accuracy: {test_results['accuracy']*100:.2f}%")
    print(f"  Test F1: {test_results['f1_score']*100:.2f}%")
    print(f"  CV accuracy: {results.get('cv_mean', 0)*100:.2f}% (+/- {results.get('cv_std', 0)*100:.2f}%)")
    print(f"  Temps entraînement: {results['train_time']:.2f}s")
    
    return model

def comparer_modeles():
    """Compare les 3 modèles"""
    print("\n" + "="*70)
    print("COMPARAISON DES MODELES")
    print("="*70)
    
    train, val, test = diviser_donnees(DONNEES)
    X_train, y_train = train
    X_test, y_test = test
    
    # Sous-ensemble pour test rapide
    X_train_small = X_train[:500]
    y_train_small = y_train[:500]
    X_test_small = X_test[:100]
    y_test_small = y_test[:100]
    
    print(f"\nDonnées de test:")
    print(f"  Train: {len(X_train_small)} exemples")
    print(f"  Test: {len(X_test_small)} exemples")
    
    results = {}
    models = [
        ("SVM", "nlu_svm_optuna_mlflow", "NLUSVMOptunaMLflow", "BEST_PARAMS_SVM"),
        ("XGBoost", "nlu_xgboost_optuna_mlflow", "NLUXGBoostOptunaMLflow", "BEST_PARAMS_XGBOOST"),
        ("Random Forest", "nlu_randomforest_optuna_mlflow", "NLURandomForestMLflow", "BEST_PARAMS_RF")
    ]
    
    for name, module_name, class_name, params_name in models:
        print("\n" + "-"*50)
        print(f"{name}")
        print("-"*50)
        
        try:
            module = __import__(module_name)
            model_class = getattr(module, class_name)
            best_params = getattr(module, params_name)
            
            import time
            
            # Différencier selon le type de modèle
            if "Random Forest" in name:
                model = model_class(use_mlflow=False)
                start = time.time()
                train_results = model.entrainer(X_train_small, y_train_small, use_best_params=True)
                train_time = time.time() - start
                train_acc = train_results['train_accuracy']
            else:
                model = model_class(use_optuna=False, use_mlflow=False)
                start = time.time()
                model.entrainer(X_train_small, y_train_small, optimiser=False, use_best_params=True)
                train_time = time.time() - start
                train_acc = model.stats['train_accuracy']
            
            # Prédictions
            y_pred = []
            for texte in X_test_small:
                pred = model.predire(texte)
                y_pred.append(pred['intention'])
            
            from sklearn.metrics import accuracy_score, f1_score
            acc = accuracy_score(y_test_small, y_pred)
            f1 = f1_score(y_test_small, y_pred, average='weighted')
            
            results[name] = {
                'train_acc': train_acc,
                'test_acc': acc,
                'f1': f1,
                'time': train_time
            }
            
            print(f"  Train accuracy: {train_acc*100:.2f}%")
            print(f"  Test accuracy: {acc*100:.2f}%")
            print(f"  Test F1: {f1*100:.2f}%")
            print(f"  Temps: {train_time:.2f}s")
            
        except Exception as e:
            print(f"  Erreur: {e}")
            results[name] = {'train_acc': 0, 'test_acc': 0, 'f1': 0, 'time': 0}
    
    print("\n" + "="*70)
    print("RÉSUMÉ COMPARATIF")
    print("="*70)
    print(f"{'Modèle':<15} {'Train Acc':<12} {'Test Acc':<12} {'F1':<12} {'Temps(s)':<12}")
    print("-"*65)
    for name, metrics in results.items():
        print(f"{name:<15} {metrics['train_acc']*100:.1f}%      {metrics['test_acc']*100:.1f}%      {metrics['f1']*100:.1f}%      {metrics['time']:.2f}")
    
    # Afficher le meilleur modèle
    best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
    print(f"\n🏆 Meilleur modèle: {best_model[0]} avec {best_model[1]['test_acc']*100:.1f}% de test accuracy")
    
    return results


# ============================================================
# MODES D'EXÉCUTION
# ============================================================

def mode_entrainement_svm():
    entrainer_modele_svm()

def mode_entrainement_xgboost():
    entrainer_modele_xgboost()

def mode_entrainement_randomforest():
    entrainer_modele_randomforest()

def mode_comparaison():
    comparer_modeles()

def mode_chat_interactif():
    print("\n" + "="*60)
    print("MODE CHATBOT INTERACTIF - SVM")
    print("="*60)
    
    bot = ChatbotTunisieTelecom()
    
    if not bot.charger_modeles():
        print("\nModèle non trouvé. Lancez d'abord:")
        print("   python main.py --mode train_svm")
        return
    
    print("\nAssistant Tunisie Telecom prêt !")
    print("Commandes spéciales:")
    print("  /quit ou /exit : Quitter")
    print("  /reset : Réinitialiser la conversation")
    print("-"*60)
    
    while True:
        try:
            message = input("\nVous: ").strip()
            if not message:
                continue
            
            if message.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("Au revoir !")
                break
            
            if message.lower() == '/reset':
                bot.dialogue.reinitialiser()
                print("Conversation réinitialisée.")
                continue
            
            reponse = bot.repondre(message)
            print(f"Bot: {reponse['reponse']}")
            
        except KeyboardInterrupt:
            print("\nAu revoir !")
            break


# ============================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chatbot Tunisie Telecom - SVM + XGBoost + Random Forest + Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  train_svm     : Entraîner le modèle SVM
  train_xgb     : Entraîner le modèle XGBoost
  train_rf      : Entraîner le modèle Random Forest
  compare       : Comparer les 3 modèles
  chat          : Lancer le chatbot interactif (SVM)
  test          : Tester le modèle sur des exemples
  stats         : Afficher les statistiques du modèle
  optuna        : Optimiser XGBoost avec Optuna
        
Exemples:
  python main.py --mode train_svm
  python main.py --mode train_rf
  python main.py --mode compare
  python main.py --mode chat
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["train_svm", "train_xgb", "train_rf", "compare", "chat", "test", "stats", "optuna"],
        default="chat",
        help="Mode d'exécution"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Chatbot Tunisie Telecom v9.0 (Multi-modèles: SVM, XGBoost, Random Forest)"
    )
    
    args = parser.parse_args()
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ sentence-transformers non installé.")
        print("Installez: pip install sentence-transformers\n")
        sys.exit(1)
    
    if args.mode == "train_svm":
        mode_entrainement_svm()
    elif args.mode == "train_xgb":
        mode_entrainement_xgboost()
    elif args.mode == "train_rf":
        mode_entrainement_randomforest()
    elif args.mode == "compare":
        mode_comparaison()
    elif args.mode == "chat":
        mode_chat_interactif()
    elif args.mode == "test":
        from main_original import mode_test
        mode_test()
    elif args.mode == "stats":
        from main_original import mode_stats
        mode_stats()
    elif args.mode == "optuna":
        from main_original import mode_optuna_optimize
        mode_optuna_optimize()


if __name__ == "__main__":
    main()