# ============================================================
# main.py - Chatbot Tunisie Telecom avec tous les modèles
# Version : 10.1 - Support complet des modes
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

# Vérifier disponibilité des packages
SENTENCE_TRANSFORMERS_AVAILABLE = False
OPTUNA_AVAILABLE = False
XGB_AVAILABLE = False
RANDOMFOREST_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ sentence-transformers non installé")

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    print("⚠️ optuna non installé")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("⚠️ xgboost non installé")

try:
    from sklearn.ensemble import RandomForestClassifier
    RANDOMFOREST_AVAILABLE = True
except ImportError:
    print("⚠️ scikit-learn non installé")

# Vérifier spaCy français
try:
    import spacy
    nlp_fr = spacy.load("fr_core_news_sm")
    print("✅ Modèle français spaCy chargé: fr_core_news_sm")
except OSError:
    print("⚠️ Téléchargement du modèle français spaCy...")
    import spacy.cli
    spacy.cli.download("fr_core_news_sm")
    nlp_fr = spacy.load("fr_core_news_sm")
    print("✅ Modèle français spaCy téléchargé et chargé")


# ============================================================
# MODÈLE SVM + EMBEDDINGS (compatible avec vos fichiers existants)
# ============================================================

class NLUSVM:
    """Modèle SVM simple pour compatibilité"""
    
    def __init__(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers requis")
        
        print("Chargement du modèle d'embeddings...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Modèle d'embeddings chargé")
        
        from sklearn.svm import SVC
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        self.svm = SVC(
            kernel='linear',
            probability=True,
            C=1.0,
            class_weight='balanced',
            random_state=42
        )
        
        self.label_encoder = LabelEncoder()
        self.entraine = False
        self.stats = {
            "total_predictions": 0,
            "avg_confidence": 0,
            "train_accuracy": 0,
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
        self.stats['train_accuracy'] = score
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
    
    def evaluer_test(self, X_test, y_test):
        if not self.entraine:
            raise Exception("Modèle non entraîné")
        
        print("\n" + "="*60)
        print("ÉVALUATION SUR TEST")
        print("="*60)
        
        X_test_emb = self._generer_embeddings(X_test, verbose=False)
        y_test_enc = self.label_encoder.transform(y_test)
        y_pred = self.svm.predict(X_test_emb)
        
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        accuracy = accuracy_score(y_test_enc, y_pred)
        f1 = f1_score(y_test_enc, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"F1-Score: {f1*100:.2f}%")
        
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_labels, zero_division=0))
        
        return {'accuracy': accuracy, 'f1_score': f1, 'y_pred': y_pred_labels}
    
    def predire(self, texte):
        if not self.entraine:
            return {"intention": "inconnu", "confiance": 0.0, "source": "non_entraine"}
        
        embedding = self.encoder.encode([texte], convert_to_numpy=True, normalize_embeddings=True)
        
        probas = self.svm.predict_proba(embedding)[0]
        y_pred = int(self.svm.predict(embedding)[0])
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
    
    def sauvegarder(self, chemin):
        import pickle
        import os
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        
        with open(chemin, 'wb') as f:
            pickle.dump({
                'svm': self.svm,
                'label_encoder': self.label_encoder,
                'entraine': self.entraine,
                'stats': self.stats,
                'version': '10.1'
            }, f)
        
        print(f"Modèle SVM sauvegardé: {chemin}")
    
    def charger(self, chemin):
        import pickle
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
        print(f"Train accuracy: {self.stats['train_accuracy']*100:.1f}%")
        print(f"Intentions disponibles: {len(self.label_encoder.classes_)}")


# ============================================================
# MODÈLE RANDOM FOREST
# ============================================================

class NLURandomForest:
    """Modèle Random Forest optimisé"""
    
    def __init__(self):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers requis")
        
        print("Chargement du modèle d'embeddings...")
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        print("Modèle d'embeddings chargé")
        
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        import numpy as np
        
        self.rf = RandomForestClassifier(
            n_estimators=350,
            max_depth=14,
            min_samples_split=9,
            min_samples_leaf=4,
            max_features='sqrt',
            max_samples=0.85,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            bootstrap=True,
            oob_score=True,
            min_impurity_decrease=0.0005
        )
        
        self.label_encoder = LabelEncoder()
        self.entraine = False
        self.stats = {
            "total_predictions": 0,
            "avg_confidence": 0,
            "train_accuracy": 0,
            "val_accuracy": 0,
            "test_accuracy": 0,
            "oob_score": 0,
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
        print("ENTRAINEMENT RANDOM FOREST")
        print("="*60)
        
        X_emb = self._generer_embeddings(X_train, verbose=True)
        y_enc = self.label_encoder.fit_transform(y_train)
        
        print(f"Dimensions: {X_emb.shape}")
        print(f"Intentions: {len(self.label_encoder.classes_)}")
        print("Entraînement Random Forest...")
        
        self.rf.fit(X_emb, y_enc)
        self.entraine = True
        
        score = self.rf.score(X_emb, y_enc)
        self.stats['train_accuracy'] = score
        
        if hasattr(self.rf, 'oob_score_') and self.rf.oob_score_:
            self.stats['oob_score'] = self.rf.oob_score_
            print(f"OOB Score: {self.stats['oob_score']*100:.2f}%")
        
        print(f"Entraînement terminé")
        print(f"   Accuracy train: {score*100:.2f}%")
        
        return score
    
    def valider(self, X_val, y_val):
        if not self.entraine:
            raise Exception("Modèle non entraîné")
        
        X_emb = self._generer_embeddings(X_val, verbose=False)
        y_enc = self.label_encoder.transform(y_val)
        y_pred = self.rf.predict(X_emb)
        
        from sklearn.metrics import accuracy_score, f1_score
        accuracy = accuracy_score(y_enc, y_pred)
        f1 = f1_score(y_enc, y_pred, average='weighted')
        
        self.stats['val_accuracy'] = accuracy
        
        print(f"Validation:")
        print(f"   Accuracy: {accuracy*100:.2f}%")
        print(f"   F1-Score: {f1*100:.2f}%")
        
        return accuracy, f1
    
    def evaluer_test(self, X_test, y_test):
        if not self.entraine:
            raise Exception("Modèle non entraîné")
        
        print("\n" + "="*60)
        print("ÉVALUATION SUR TEST")
        print("="*60)
        
        X_test_emb = self._generer_embeddings(X_test, verbose=False)
        y_test_enc = self.label_encoder.transform(y_test)
        y_pred = self.rf.predict(X_test_emb)
        
        from sklearn.metrics import accuracy_score, f1_score, classification_report
        accuracy = accuracy_score(y_test_enc, y_pred)
        f1 = f1_score(y_test_enc, y_pred, average='weighted')
        
        self.stats['test_accuracy'] = accuracy
        
        print(f"Accuracy: {accuracy*100:.2f}%")
        print(f"F1-Score: {f1*100:.2f}%")
        
        y_pred_labels = self.label_encoder.inverse_transform(y_pred)
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_labels, zero_division=0))
        
        return {'accuracy': accuracy, 'f1_score': f1, 'y_pred': y_pred_labels}
    
    def predire(self, texte):
        if not self.entraine:
            return {"intention": "inconnu", "confiance": 0.0, "source": "non_entraine"}
        
        embedding = self.encoder.encode([texte], convert_to_numpy=True, normalize_embeddings=True)
        
        probas = self.rf.predict_proba(embedding)[0]
        y_pred = int(self.rf.predict(embedding)[0])
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
            "source": "randomforest_embeddings",
            "probas": {intent: float(p) for intent, p in zip(self.label_encoder.classes_, probas)}
        }
    
    def sauvegarder(self, chemin):
        import pickle
        import os
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        
        with open(chemin, 'wb') as f:
            pickle.dump({
                'rf': self.rf,
                'label_encoder': self.label_encoder,
                'entraine': self.entraine,
                'stats': self.stats,
                'version': '10.1'
            }, f)
        
        print(f"Modèle Random Forest sauvegardé: {chemin}")
    
    def charger(self, chemin):
        import pickle
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"Fichier modèle introuvable: {chemin}")
        
        with open(chemin, 'rb') as f:
            data = pickle.load(f)
        
        self.rf = data['rf']
        self.label_encoder = data['label_encoder']
        self.entraine = data['entraine']
        self.stats = data.get('stats', self.stats)
        
        print(f"Modèle Random Forest chargé: {chemin}")
    
    def afficher_stats(self):
        print("\n" + "="*60)
        print("STATISTIQUES RANDOM FOREST")
        print("="*60)
        print(f"Total prédictions: {self.stats['total_predictions']}")
        print(f"Confiance moyenne: {self.stats['avg_confidence']*100:.1f}%")
        print(f"Train accuracy: {self.stats['train_accuracy']*100:.1f}%")
        if self.stats.get('val_accuracy', 0):
            print(f"Validation accuracy: {self.stats['val_accuracy']*100:.1f}%")
        if self.stats.get('test_accuracy', 0):
            print(f"Test accuracy: {self.stats['test_accuracy']*100:.1f}%")
        print(f"OOB Score: {self.stats.get('oob_score', 0)*100:.1f}%")
        print(f"Intentions disponibles: {len(self.label_encoder.classes_)}")


# ============================================================
# CHATBOT PRINCIPAL
# ============================================================

class ChatbotTunisieTelecom:
    """Chatbot principal avec choix du modèle"""
    
    def __init__(self, model_type='rf'):
        self.model_type = model_type
        self.nlu = None
        self.ner = None
        self.dialogue = None
        self.modele_charge = False
    
    def initialiser(self):
        try:
            if self.model_type == 'svm':
                self.nlu = NLUSVM()
            else:
                self.nlu = NLURandomForest()
            
            self.ner = NER()  # Utilise spaCy français
            self.dialogue = GestionnaireDialogue(modele_nlu=self.nlu)
            
        except ImportError as e:
            print(f"Erreur: {e}")
            sys.exit(1)
    
    def charger_modeles(self):
        self.initialiser()
        try:
            if self.model_type == 'svm':
                self.nlu.charger("modele_scratch/modele_svm_full.pkl")
            else:
                self.nlu.charger("modele_scratch/modele_randomforest_full.pkl")
            
            self.modele_charge = True
            print(f"✅ Modèle {self.model_type.upper()} chargé avec succès")
            print("✅ NER avec spaCy français prêt")
            return True
            
        except FileNotFoundError:
            print(f"Aucun modèle trouvé. Lancez d'abord l'entraînement:")
            if self.model_type == 'svm':
                print("   python main.py --mode train_svm")
            else:
                print("   python main.py --mode train_rf")
            return False
        except Exception as e:
            print(f"Erreur chargement: {e}")
            return False
    
    def repondre(self, message):
        if not self.modele_charge:
            return {
                "reponse": "Modèle non chargé.",
                "erreur": True
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
# FONCTIONS D'ENTRAÎNEMENT
# ============================================================

def entrainer_modele_svm():
    """Entraîne le modèle SVM"""
    print("\n" + "="*60)
    print("ENTRAINEMENT SVM")
    print("="*60)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers non installé.")
        return None
    
    train, val, test = diviser_donnees(DONNEES)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    print(f"\n📊 Données:")
    print(f"   Train: {len(X_train)} exemples")
    print(f"   Validation: {len(X_val)} exemples")
    print(f"   Test: {len(X_test)} exemples")
    
    from collections import Counter
    print(f"   Intentions: {len(Counter(y_train))} différentes")
    
    model = NLUSVM()
    
    print("\n🔧 Paramètres SVM:")
    print("   kernel: linear")
    print("   C: 1.0")
    print("   class_weight: balanced")
    
    print("\n🏋️ Début de l'entraînement...")
    train_acc = model.entrainer(X_train, y_train)
    
    print("\n📊 Évaluation sur validation...")
    val_acc, val_f1 = model.valider(X_val, y_val)
    
    print("\n📊 Évaluation sur test...")
    test_results = model.evaluer_test(X_test, y_test)
    
    model.sauvegarder("modele_scratch/modele_svm_full.pkl")
    
    print(f"\n" + "="*60)
    print("📈 RÉSULTATS FINAUX - SVM")
    print("="*60)
    print(f"   Train accuracy: {train_acc*100:.2f}%")
    print(f"   Validation accuracy: {val_acc*100:.2f}%")
    print(f"   Test accuracy: {test_results['accuracy']*100:.2f}%")
    
    return model


def entrainer_modele_randomforest():
    """Entraîne le modèle Random Forest"""
    print("\n" + "="*60)
    print("ENTRAINEMENT RANDOM FOREST")
    print("="*60)
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("❌ sentence-transformers non installé.")
        return None
    
    train, val, test = diviser_donnees(DONNEES)
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    print(f"\n📊 Données:")
    print(f"   Train: {len(X_train)} exemples")
    print(f"   Validation: {len(X_val)} exemples")
    print(f"   Test: {len(X_test)} exemples")
    
    from collections import Counter
    class_counts = Counter(y_train)
    print(f"   Intentions: {len(class_counts)} différentes")
    print(f"   Classes avec peu d'exemples (<30): {sum(1 for c in class_counts.values() if c < 30)}")
    
    model = NLURandomForest()
    
    print("\n🔧 Paramètres Random Forest:")
    print("   n_estimators: 350")
    print("   max_depth: 14")
    print("   min_samples_split: 9")
    print("   min_samples_leaf: 4")
    print("   max_features: sqrt")
    print("   class_weight: balanced")
    
    print("\n🏋️ Début de l'entraînement...")
    train_acc = model.entrainer(X_train, y_train)
    
    print("\n📊 Évaluation sur validation...")
    val_acc, val_f1 = model.valider(X_val, y_val)
    
    print("\n📊 Évaluation sur test...")
    test_results = model.evaluer_test(X_test, y_test)
    
    model.sauvegarder("modele_scratch/modele_randomforest_full.pkl")
    
    print(f"\n" + "="*60)
    print("📈 RÉSULTATS FINAUX - RANDOM FOREST")
    print("="*60)
    print(f"   Train accuracy: {train_acc*100:.2f}%")
    print(f"   Validation accuracy: {val_acc*100:.2f}%")
    print(f"   Test accuracy: {test_results['accuracy']*100:.2f}%")
    
    return model


# ============================================================
# MODES D'EXÉCUTION
# ============================================================

def mode_chat_interactif(model_type='rf'):
    """Mode chat interactif"""
    print("\n" + "="*60)
    print(f"🤖 CHATBOT TUNISIE TELECOM - {model_type.upper()}")
    print("   NER: spaCy français (fr_core_news_sm)")
    print("="*60)
    
    bot = ChatbotTunisieTelecom(model_type=model_type)
    
    if not bot.charger_modeles():
        print("\n❌ Modèle non trouvé. Lancez d'abord:")
        if model_type == 'svm':
            print("   python main.py --mode train_svm")
        else:
            print("   python main.py --mode train_rf")
        return
    
    print("\n✅ Assistant Tunisie Telecom prêt !")
    print("\nCommandes spéciales:")
    print("   /quit ou /exit : Quitter")
    print("   /reset : Réinitialiser la conversation")
    print("   /stats : Afficher les statistiques")
    print("-"*60)
    
    while True:
        try:
            message = input("\n👤 Vous: ").strip()
            if not message:
                continue
            
            if message.lower() in ['/quit', '/exit', 'quit', 'exit']:
                print("👋 Au revoir !")
                break
            
            if message.lower() == '/reset':
                bot.dialogue.reinitialiser()
                print("🔄 Conversation réinitialisée.")
                continue
            
            if message.lower() == '/stats':
                bot.nlu.afficher_stats()
                continue
            
            reponse = bot.repondre(message)
            print(f"🤖 Bot: {reponse['reponse']}")
            
            if reponse.get('intention') and reponse.get('confiance', 0) > 0.3:
                print(f"   [Intention: {reponse['intention']} ({reponse['confiance']*100:.0f}%)]")
            
        except KeyboardInterrupt:
            print("\n👋 Au revoir !")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")
            continue


def mode_stats(model_type='rf'):
    """Affiche les statistiques du modèle chargé"""
    print("\n" + "="*60)
    print(f"📊 STATISTIQUES - {model_type.upper()}")
    print("="*60)
    
    bot = ChatbotTunisieTelecom(model_type=model_type)
    if bot.charger_modeles():
        bot.nlu.afficher_stats()
    else:
        print("❌ Aucun modèle chargé")


# ============================================================
# POINT D'ENTRÉE PRINCIPAL
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chatbot Tunisie Telecom - SVM / Random Forest + spaCy français",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  train_svm     : Entraîner le modèle SVM
  train_rf      : Entraîner le modèle Random Forest
  chat          : Lancer le chatbot interactif (Random Forest par défaut)
  chat_svm      : Lancer le chatbot avec SVM
  stats         : Afficher les statistiques du modèle Random Forest
  stats_svm     : Afficher les statistiques du modèle SVM
        
Exemples:
  python main.py --mode train_svm
  python main.py --mode train_rf
  python main.py --mode chat
  python main.py --mode chat_svm
  python main.py --mode stats
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["train_svm", "train_rf", "chat", "chat_svm", "stats", "stats_svm"],
        default="chat",
        help="Mode d'exécution"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Chatbot Tunisie Telecom v10.1 (SVM/Random Forest + spaCy français)"
    )
    
    args = parser.parse_args()
    
    if not SENTENCE_TRANSFORMERS_AVAILABLE:
        print("\n❌ sentence-transformers non installé.")
        print("   Installez: pip install sentence-transformers\n")
        sys.exit(1)
    
    if args.mode == "train_svm":
        entrainer_modele_svm()
    elif args.mode == "train_rf":
        entrainer_modele_randomforest()
    elif args.mode == "chat":
        mode_chat_interactif('rf')
    elif args.mode == "chat_svm":
        mode_chat_interactif('svm')
    elif args.mode == "stats":
        mode_stats('rf')
    elif args.mode == "stats_svm":
        mode_stats('svm')


if __name__ == "__main__":
    main()