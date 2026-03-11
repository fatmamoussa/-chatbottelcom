# ============================================================
# nlu.py - Modèle NLU from scratch avec TF-IDF + SVM
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 5.0 - Complète et corrigée
# ============================================================

import re
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold
import spacy

from config import config
from data import DONNEES, diviser_donnees, afficher_statistiques


# ============================================================
# PRÉTRAITEMENT
# ============================================================

class Pretraitement:
    """
    Prétraitement avancé des textes avec spaCy
    - Nettoyage
    - Tokenisation
    - Lemmatisation optionnelle
    - Features supplémentaires
    """
    
    def __init__(self, utiliser_lemmatisation=None, min_token_length=None):
        self.nlp = spacy.load("fr_core_news_sm")
        self.utiliser_lemmatisation = utiliser_lemmatisation or config.NLU.UTILISER_LEMMATISATION
        self.min_token_length = min_token_length or config.NLU.MIN_TOKEN_LENGTH
    
    def nettoyer_texte(self, texte):
        """Nettoyage de base du texte"""
        if not isinstance(texte, str):
            return ""
        
        # Mise en minuscules
        texte = texte.lower().strip()
        
        # Suppression des caractères spéciaux (garder lettres, chiffres, underscores)
        texte = re.sub(r'[^\w\s_]', ' ', texte)
        
        # Suppression des espaces multiples
        texte = re.sub(r'\s+', ' ', texte)
        
        return texte.strip()
    
    def tokeniser(self, texte):
        """Tokenisation avec spaCy"""
        doc = self.nlp(texte)
        
        if self.utiliser_lemmatisation:
            # Utiliser les lemmes
            tokens = [token.lemma_ for token in doc 
                     if not token.is_stop and len(token.text) >= self.min_token_length]
        else:
            # Utiliser les tokens bruts
            tokens = [token.text for token in doc 
                     if not token.is_stop and len(token.text) >= self.min_token_length]
        
        return tokens
    
    def extraire_features_metier(self, texte):
        """
        Extrait des features métier spécifiques au domaine
        """
        texte_lower = texte.lower()
        features = []
        
        # Mots-clés par intention
        if "cc_" in texte_lower:
            features.append("a_numero_client")
        if "merci" in texte_lower:
            features.append("a_merci")
        if "non" in texte_lower or "pas" in texte_lower:
            features.append("a_negation")
        if "oui" in texte_lower or "d'accord" in texte_lower:
            features.append("a_affirmation")
        if "recharge" in texte_lower:
            features.append("a_recharge")
        if "appel" in texte_lower:
            features.append("a_appel")
        if "internet" in texte_lower or "data" in texte_lower or "conso" in texte_lower:
            features.append("a_internet")
        if "forfait" in texte_lower or "offre" in texte_lower:
            features.append("a_offre")
        if "cout" in texte_lower or "depense" in texte_lower or "facture" in texte_lower:
            features.append("a_cout")
        if "combien" in texte_lower:
            features.append("a_combien")
        if "total" in texte_lower:
            features.append("a_total")
        if "info" in texte_lower or "profil" in texte_lower:
            features.append("a_info")
        if "nombre" in texte_lower or "compteur" in texte_lower:
            features.append("a_nombre")
        if "montant" in texte_lower or "somme" in texte_lower:
            features.append("a_montant")
        if "bonus" in texte_lower:
            features.append("a_bonus")
        if "volume" in texte_lower:
            features.append("a_volume")
        if "durée" in texte_lower or "duree" in texte_lower:
            features.append("a_duree")
        if "destination" in texte_lower:
            features.append("a_destination")
        if "réseau" in texte_lower or "reseau" in texte_lower:
            features.append("a_reseau")
        if "taxation" in texte_lower:
            features.append("a_taxation")
        if "type" in texte_lower:
            features.append("a_type")
        if "service" in texte_lower:
            features.append("a_service")
        if "code" in texte_lower:
            features.append("a_code")
        if "option" in texte_lower:
            features.append("a_option")
        if "statut" in texte_lower:
            features.append("a_statut")
        if "segment" in texte_lower:
            features.append("a_segment")
        if "date" in texte_lower:
            features.append("a_date")
        if "activation" in texte_lower:
            features.append("a_activation")
        if "session" in texte_lower:
            features.append("a_session")
        
        return features
    
    def pretraiter(self, texte, ajouter_features=True):
        """
        Pipeline complet de prétraitement
        
        Args:
            texte: Texte à prétraiter
            ajouter_features: Ajouter les features métier
        
        Returns:
            Texte prétraité
        """
        # Nettoyage
        texte_propre = self.nettoyer_texte(texte)
        
        # Tokenisation
        tokens = self.tokeniser(texte_propre)
        
        # Reconstruction
        texte_final = " ".join(tokens) if tokens else texte_propre
        
        # Ajout des features métier
        if ajouter_features:
            features = self.extraire_features_metier(texte)
            if features:
                texte_final += " " + " ".join(features)
        
        return texte_final


# ============================================================
# MODÈLE NLU PRINCIPAL
# ============================================================

class ModeleNLU:
    """
    Modèle NLU basé sur TF-IDF + SVM
    - Classification d'intentions
    - Gestion des ambiguïtés
    - Validation croisée
    - Sauvegarde/chargement
    """
    
    def __init__(self):
        self.pretraitement = Pretraitement()
        
        # Pipeline TF-IDF + SVM
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(
                analyzer='char_wb',
                ngram_range=config.NLU.TFIDF_NGRAM_RANGE,
                max_features=config.NLU.TFIDF_MAX_FEATURES,
                sublinear_tf=config.NLU.TFIDF_SUBLINEAR_TF,
                min_df=config.NLU.TFIDF_MIN_DF,
                max_df=config.NLU.TFIDF_MAX_DF,
                use_idf=config.NLU.TFIDF_USE_IDF
            )),
            ('svm', SVC(
                kernel=config.NLU.SVM_KERNEL,
                probability=config.NLU.SVM_PROBABILITY,
                C=config.NLU.SVM_C,
                class_weight=config.NLU.SVM_CLASS_WEIGHT,
                gamma=config.NLU.SVM_GAMMA,
                random_state=config.NLU.RANDOM_STATE
            ))
        ])
        
        self.label_encoder = LabelEncoder()
        self.entraine = False
        self.historique = {
            'train': {},
            'validation': {},
            'test': {},
            'cross_val': []
        }
        
        print("✅ Modèle NLU initialisé")
    
    def _preparer_donnees(self, textes, entrainement=False):
        """Prépare les données pour l'entraînement ou la prédiction"""
        if isinstance(textes, str):
            textes = [textes]
        
        textes_prep = [self.pretraitement.pretraiter(t) for t in textes]
        
        if entrainement:
            tailles = [len(t.split()) for t in textes_prep]
            print(f"  📊 Prétraitement : {len(textes_prep)} exemples")
            print(f"     Longueur moyenne: {np.mean(tailles):.1f} tokens")
        
        return textes_prep
    
    def entrainer(self, X_train, y_train, verbose=True):
        """Entraîne le modèle sur les données d'entraînement"""
        print("\n" + "="*60)
        print("  PHASE 1 — ENTRAÎNEMENT")
        print("="*60)
        
        X_prep = self._preparer_donnees(X_train, entrainement=True)
        y_enc = self.label_encoder.fit_transform(y_train)
        
        self.pipeline.fit(X_prep, y_enc)
        self.entraine = True
        
        y_pred = self.pipeline.predict(X_prep)
        accuracy = accuracy_score(y_enc, y_pred)
        
        if verbose:
            print(f"\n✅ Modèle entraîné sur {len(X_train)} exemples")
            print(f"🎯 Intentions: {len(self.label_encoder.classes_)}")
            print(f"📈 Accuracy train: {accuracy*100:.2f}%")
        
        self.historique['train'] = {
            'accuracy': round(accuracy, 4),
            'n_exemples': len(X_train),
            'n_intentions': len(self.label_encoder.classes_)
        }
        
        return accuracy
    
    def valider(self, X_val, y_val, verbose=True):
        """Évalue le modèle sur l'ensemble de validation"""
        print("\n" + "="*60)
        print("  PHASE 2 — VALIDATION")
        print("="*60)
        
        if not self.entraine:
            raise Exception("❌ Modèle non entraîné !")
        
        X_prep = self._preparer_donnees(X_val)
        y_enc = self.label_encoder.transform(y_val)
        y_pred = self.pipeline.predict(X_prep)
        
        accuracy = accuracy_score(y_enc, y_pred)
        f1 = f1_score(y_enc, y_pred, average='weighted')
        
        if verbose:
            print(f"\n📈 Accuracy validation: {accuracy*100:.2f}%")
            print(f"📈 F1-Score validation: {f1*100:.2f}%")
            
            # Détection d'overfitting
            train_acc = self.historique['train']['accuracy']
            ecart = train_acc - accuracy
            
            if ecart > 0.15:
                print(f"⚠️  ALERTE OVERFITTING: Écart Train/Val = {ecart*100:.1f}% > 15%")
            elif ecart > 0.10:
                print(f"⚠️  Attention: Écart Train/Val = {ecart*100:.1f}% > 10%")
            else:
                print(f"✅ Bonne généralisation: Écart Train/Val = {ecart*100:.1f}%")
            
            print("\n📊 Résultats par intention:")
            rapport = classification_report(
                y_enc, y_pred,
                target_names=self.label_encoder.classes_,
                output_dict=True,
                zero_division=0
            )
            
            for intent in self.label_encoder.classes_:
                if intent in rapport:
                    r = rapport[intent]
                    if r['f1-score'] < 0.7:
                        print(f"   ⚠️ {intent:<25} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f}")
                    else:
                        print(f"   ✅ {intent:<25} P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1-score']:.2f}")
        
        self.historique['validation'] = {
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1, 4),
            'n_exemples': len(X_val)
        }
        
        return accuracy, f1
    
    def tester(self, X_test, y_test, verbose=True):
        """Évaluation finale sur l'ensemble de test"""
        print("\n" + "="*60)
        print("  PHASE 3 — TEST FINAL")
        print("="*60)
        
        if not self.entraine:
            raise Exception("❌ Modèle non entraîné !")
        
        X_prep = self._preparer_donnees(X_test)
        y_enc = self.label_encoder.transform(y_test)
        y_pred = self.pipeline.predict(X_prep)
        
        accuracy = accuracy_score(y_enc, y_pred)
        f1 = f1_score(y_enc, y_pred, average='weighted')
        
        if verbose:
            print(f"\n📈 Accuracy test: {accuracy*100:.2f}%")
            print(f"📈 F1-Score test: {f1*100:.2f}%")
            
            print("\n📊 Rapport de classification détaillé:")
            print(classification_report(
                y_enc, y_pred,
                target_names=self.label_encoder.classes_,
                zero_division=0
            ))
            
            print("\n📊 Matrice de confusion:")
            cm = confusion_matrix(y_enc, y_pred)
            print(cm)
        
        self.historique['test'] = {
            'accuracy': round(accuracy, 4),
            'f1_score': round(f1, 4),
            'n_exemples': len(X_test)
        }
        
        return accuracy, f1
    
    def validation_croisee(self, X, y, k=5, verbose=True):
        """Validation croisée k-fold"""
        print("\n" + "="*60)
        print(f"  VALIDATION CROISÉE {k}-FOLD")
        print("="*60)
        
        X_prep = self._preparer_donnees(X)
        y_enc = self.label_encoder.fit_transform(y)
        
        skf = StratifiedKFold(
            n_splits=k, 
            shuffle=True, 
            random_state=config.NLU.RANDOM_STATE
        )
        
        scores = []
        fold = 1
        
        for train_idx, val_idx in skf.split(X_prep, y_enc):
            X_train = [X_prep[i] for i in train_idx]
            X_val = [X_prep[i] for i in val_idx]
            y_train = y_enc[train_idx]
            y_val = y_enc[val_idx]
            
            self.pipeline.fit(X_train, y_train)
            score = self.pipeline.score(X_val, y_val)
            scores.append(score)
            
            if verbose:
                print(f"  Fold {fold}: {score*100:.2f}%")
            
            fold += 1
        
        moyenne = np.mean(scores)
        ecart_type = np.std(scores)
        
        if verbose:
            print(f"\n📊 Résultats validation croisée:")
            print(f"   Moyenne: {moyenne*100:.2f}%")
            print(f"   Écart-type: ±{ecart_type*100:.2f}%")
            if ecart_type > 0.05:
                print(f"   ⚠️  Écart-type élevé (>5%) - instabilité")
            else:
                print(f"   ✅ Modèle stable")
        
        self.historique['cross_val'] = {
            'scores': [round(s, 4) for s in scores],
            'moyenne': round(moyenne, 4),
            'ecart_type': round(ecart_type, 4)
        }
        
        return scores
    
    def predire(self, texte, avec_probas=False, analyser_ambiguite=True):
        """
        Prédit l'intention d'un texte avec gestion d'ambiguïté
        """
        if not self.entraine:
            raise Exception("❌ Modèle non entraîné !")
        
        texte_prep = self.pretraitement.pretraiter(texte)
        probas = self.pipeline.predict_proba([texte_prep])[0]
        
        # Trier les indices par probabilité décroissante
        indices_tries = np.argsort(probas)[::-1]
        top1_idx = indices_tries[0]
        top2_idx = indices_tries[1] if len(indices_tries) > 1 else None
        
        intention = self.label_encoder.inverse_transform([top1_idx])[0]
        confiance = round(float(probas[top1_idx]), 4)
        
        resultat = {
            "intention": intention,
            "confiance": confiance,
            "texte_original": texte,
            "texte_preproces": texte_prep
        }
        
        # Vérifier l'ambiguïté
        if analyser_ambiguite and top2_idx is not None:
            top2_confiance = probas[top2_idx]
            ecart = confiance - top2_confiance
            
            if ecart < config.NLU.SEUIL_AMBIGUITE:
                top2_intention = self.label_encoder.inverse_transform([top2_idx])[0]
                resultat["ambigu"] = True
                resultat["intention_alternative"] = top2_intention
                resultat["ecart"] = round(float(ecart), 4)
        
        if avec_probas:
            resultat["probas"] = {
                intent: round(float(prob), 4)
                for intent, prob in zip(self.label_encoder.classes_, probas)
            }
        
        return resultat
    
    def predire_batch(self, textes):
        """Prédit les intentions pour une liste de textes"""
        return [self.predire(t) for t in textes]
    
    def afficher_resume(self):
        """Affiche un résumé complet des performances"""
        print("\n" + "="*60)
        print("  RÉSUMÉ DES PERFORMANCES")
        print("="*60)
        
        if self.historique['train']:
            train_acc = self.historique['train']['accuracy']
            print(f"📊 TRAIN:      {train_acc*100:.2f}%")
        
        if self.historique['validation']:
            val_acc = self.historique['validation']['accuracy']
            val_f1 = self.historique['validation']['f1_score']
            print(f"📊 VALIDATION: {val_acc*100:.2f}% (F1: {val_f1*100:.2f}%)")
        
        if self.historique['test']:
            test_acc = self.historique['test']['accuracy']
            test_f1 = self.historique['test']['f1_score']
            print(f"📊 TEST:       {test_acc*100:.2f}% (F1: {test_f1*100:.2f}%)")
        
        # Analyse d'overfitting
        if self.historique['train'] and self.historique['test']:
            train_acc = self.historique['train']['accuracy']
            test_acc = self.historique['test']['accuracy']
            ecart = train_acc - test_acc
            
            print("\n🔍 ANALYSE OVERFITTING:")
            if ecart > 0.15:
                print(f"❌ OVERFITTING SÉVÈRE: Écart Train/Test = {ecart*100:.1f}%")
            elif ecart > 0.10:
                print(f"⚠️  OVERFITTING MODÉRÉ: Écart Train/Test = {ecart*100:.1f}%")
            else:
                print(f"✅ BONNE GÉNÉRALISATION: Écart = {ecart*100:.1f}%")
        
        if 'cross_val' in self.historique and self.historique['cross_val']:
            print(f"\n📊 VALIDATION CROISÉE:")
            print(f"   Moyenne: {self.historique['cross_val']['moyenne']*100:.2f}%")
            print(f"   Écart-type: ±{self.historique['cross_val']['ecart_type']*100:.2f}%")
        
        print("="*60)
    
    def obtenir_intentions(self):
        """Retourne la liste des intentions possibles"""
        return list(self.label_encoder.classes_) if self.entraine else []
    
    def sauvegarder(self, chemin=None):
        """Sauvegarde le modèle entraîné"""
        if not self.entraine:
            raise Exception("❌ Rien à sauvegarder - modèle non entraîné !")
        
        chemin = chemin or config.CHEMIN_MODELE_NLU
        os.makedirs(os.path.dirname(chemin), exist_ok=True)
        
        data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'historique': self.historique,
            'version': '5.0'
        }
        
        with open(chemin, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n✅ Modèle sauvegardé: {chemin}")
        print(f"   Intentions: {len(self.label_encoder.classes_)}")
        print(f"   Version: 5.0")
    
    def charger(self, chemin=None):
        """Charge un modèle sauvegardé"""
        chemin = chemin or config.CHEMIN_MODELE_NLU
        
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"❌ Fichier modèle introuvable: {chemin}")
        
        with open(chemin, 'rb') as f:
            data = pickle.load(f)
        
        self.pipeline = data['pipeline']
        self.label_encoder = data['label_encoder']
        self.historique = data.get('historique', {})
        self.entraine = True
        
        print(f"\n✅ Modèle chargé: {chemin}")
        print(f"   Intentions: {len(self.label_encoder.classes_)}")
        print(f"   Version: {data.get('version', '1.0')}")


# ============================================================
# FONCTION D'ENTRAÎNEMENT COMPLET
# ============================================================

def entrainer_modele_complet(avec_cross_val=True, k_folds=5):
    """
    Entraîne le modèle complet avec division train/val/test
    
    Args:
        avec_cross_val: Effectuer une validation croisée
        k_folds: Nombre de folds pour la validation croisée
    
    Returns:
        Modèle entraîné
    """
    print("\n" + "="*60)
    print("  ENTRAÎNEMENT COMPLET DU MODÈLE NLU")
    print("="*60)
    
    # 1. Diviser les données
    train, val, test = diviser_donnees(DONNEES)
    afficher_statistiques(train, val, test)
    
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    
    # 2. Créer le modèle
    modele = ModeleNLU()
    
    # 3. Entraînement
    modele.entrainer(X_train, y_train)
    
    # 4. Validation
    modele.valider(X_val, y_val)
    
    # 5. Test
    modele.tester(X_test, y_test)
    
    # 6. Validation croisée
    if avec_cross_val:
        X_all = X_train + X_val + X_test
        y_all = y_train + y_val + y_test
        modele.validation_croisee(X_all, y_all, k=k_folds)
    
    # 7. Résumé
    modele.afficher_resume()
    
    # 8. Sauvegarde
    modele.sauvegarder()
    
    return modele


if __name__ == "__main__":
    print("="*60)
    print("  TEST DU MODÈLE NLU")
    print("="*60)
    
    # Mode 1: Entraînement complet
    print("\n1️⃣  Entraînement complet du modèle...")
    modele = entrainer_modele_complet(avec_cross_val=True, k_folds=5)
    
    # Mode 2: Tests de prédiction
    print("\n" + "="*60)
    print("  TESTS DE PRÉDICTION")
    print("="*60)
    
    tests = [
        "bonjour",
        "nombre de recharges",
        "montant des recharges",
        "bonus des recharges",
        "CC_52099260",
        "volume internet",
        "nombre d'appels",
        "durée des appels",
        "mon offre",
        "mon statut",
        "merci",
        "au revoir",
    ]
    
    print("\nRésultats des prédictions:")
    print("-" * 70)
    
    for msg in tests:
        resultat = modele.predire(msg, analyser_ambiguite=True)
        confiance = resultat['confiance'] * 100
        
        if confiance >= 80:
            marker = "✅"
        elif confiance >= 50:
            marker = "⚠️"
        else:
            marker = "❓"
        
        print(f"{marker} '{msg}'")
        print(f"   → {resultat['intention']} ({confiance:.1f}%)")
        
        if resultat.get('ambigu', False):
            print(f"   ⚠️  Ambigu: aussi {resultat['intention_alternative']} "
                  f"(écart {resultat['ecart']*100:.1f}%)")
    
    print("-" * 70)
    print("\n✅ Tests terminés")