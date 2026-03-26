# dialogue.py - Version corrigée (utilise SVM existant)

import sys
from actions import ACTIONS_CSV
from config import config

class GestionnaireDialogue:
    """
    Gestionnaire de dialogue avec mémoire (slots)
    """
    
    def __init__(self, modele_nlu=None):
        # Mémoire de la conversation
        self.slots = {
            "contrat_client": None,
            "mois_annee": None,
        }
        self.derniere_intention = None
        self.compteur_tours = 0
        self.historique = []
        
        # Index pour alterner les réponses
        self._idx_reponse = 0
        self._idx_demande_cc = 0
        self._idx_fallback = 0
        
        # Modèle NLU (passé de l'extérieur)
        self.modele_nlu = modele_nlu
        
        print("✅ Gestionnaire de dialogue initialisé")
    
    def _choisir_reponse(self, liste):
        idx = self._idx_reponse % len(liste)
        self._idx_reponse += 1
        return liste[idx]
    
    def _choisir_demande_cc(self):
        idx = self._idx_demande_cc % len(config.DIALOGUE.DEMANDE_CC)
        self._idx_demande_cc += 1
        return config.DIALOGUE.DEMANDE_CC[idx]
    
    def _choisir_fallback(self):
        idx = self._idx_fallback % len(config.DIALOGUE.FALLBACK)
        self._idx_fallback += 1
        return config.DIALOGUE.FALLBACK[idx]
    
    def _mettre_a_jour_slots(self, entites):
        if "contrat_client" in entites:
            self.slots["contrat_client"] = entites["contrat_client"]
            print(f"📝 Slot mis à jour: contrat_client = {self.slots['contrat_client']}")
        if "mois_annee" in entites:
            self.slots["mois_annee"] = entites["mois_annee"]
            print(f"📝 Slot mis à jour: mois_annee = {self.slots['mois_annee']}")
    
    def _construire_contexte(self):
        contexte = []
        if self.slots["contrat_client"]:
            contexte.append(f"Client: {self.slots['contrat_client']}")
        if self.slots["mois_annee"]:
            contexte.append(f"Période: {self.slots['mois_annee']}")
        if self.derniere_intention and self.derniere_intention not in ["saluer", "au_revoir", "remercier"]:
            contexte.append(f"Dernière action: {self.derniere_intention}")
        return " | ".join(contexte) if contexte else None
    
    def _enregistrer_echange(self, message, intention, confiance, reponse):
        self.historique.append({
            "tour": self.compteur_tours,
            "message": message,
            "intention": intention,
            "confiance": confiance,
            "reponse": reponse,
            "slots": dict(self.slots)
        })
        if len(self.historique) > 10:
            self.historique = self.historique[-10:]
    
    def _appliquer_filtre_mois(self, intention, entites):
        mois = entites.get("mois_annee") or self.slots.get("mois_annee")
        if mois:
            print(f"📅 Filtre appliqué : {mois}")
            return mois
        return None
    
    def traiter(self, message, intention_old, confiance_old, entites):
        """
        Traiter un message avec le modèle NLU
        """
        self.compteur_tours += 1
        
        # Mettre à jour les slots avec les entités NER
        self._mettre_a_jour_slots(entites)
        
        # ========== UTILISER LE MODÈLE SVM SI DISPONIBLE ==========
        intention = intention_old
        confiance = confiance_old
        source = "fallback"
        
        # Si un modèle NLU est fourni, l'utiliser
        if self.modele_nlu and hasattr(self.modele_nlu, 'entraine') and self.modele_nlu.entraine:
            try:
                resultat = self.modele_nlu.predire(message)
                intention = resultat["intention"]
                confiance = resultat["confiance"]
                source = "svm"
                print(f"🎯 [{source}] Intention: {intention} ({confiance*100:.1f}%)")
            except Exception as e:
                print(f"⚠️ Erreur prédiction: {e}")
        else:
            print(f"🎯 [{source}] Utilisation valeurs passées: intention={intention}, confiance={confiance*100:.1f}%")
        
        # ========== VÉRIFICATION SPÉCIALE: Numéro client détecté par NER ==========
        # Si un numéro client a été détecté par NER mais l'intention n'est pas "donner_id_client"
        if entites.get("contrat_client") and intention != "donner_id_client":
            # C'est probablement un numéro client, on force l'intention
            if len(message.strip()) < 30 and "CC_" in message:
                intention = "donner_id_client"
                confiance = 0.95
                print(f"🎯 [NER] Force intention: donner_id_client")
        
        # ========== CAS 1: Intention inconnue ==========
        if intention == "inconnu" or confiance < 0.3:
            reponse = "Je n'ai pas bien compris. Pouvez-vous reformuler votre question ?"
            self._enregistrer_echange(message, intention, confiance, reponse)
            return self._formater_retour(reponse, intention, confiance)
        
        # ========== CAS 2: Client donne son numéro ==========
        if intention == "donner_id_client":
            if self.slots["contrat_client"]:
                reponse = f"✅ Numéro client {self.slots['contrat_client']} enregistré."
                # Si on attendait une action, la relancer
                if self.derniere_intention in ACTIONS_CSV:
                    fn = ACTIONS_CSV[self.derniere_intention]
                    mois_filtre = self._appliquer_filtre_mois(self.derniere_intention, entites)
                    if mois_filtre:
                        entites["mois_annee"] = mois_filtre
                    reponse_csv, self.slots = fn(entites, self.slots)
                    reponse = reponse_csv
                else:
                    reponse += " Que souhaitez-vous consulter ?"
            else:
                reponse = self._choisir_demande_cc()
            
            self._enregistrer_echange(message, intention, confiance, reponse)
            return self._formater_retour(reponse, intention, confiance)
        
        # ========== CAS 3: Réponse simple ==========
        if intention in config.DIALOGUE.REPONSES_SIMPLES:
            self.derniere_intention = intention
            reponse = self._choisir_reponse(config.DIALOGUE.REPONSES_SIMPLES[intention])
            self._enregistrer_echange(message, intention, confiance, reponse)
            return self._formater_retour(reponse, intention, confiance)
        
        # ========== CAS 4: Action CSV ==========
        if intention in ACTIONS_CSV:
            self.derniere_intention = intention
            fn = ACTIONS_CSV[intention]
            
            # Appliquer le filtre mois
            mois_filtre = self._appliquer_filtre_mois(intention, entites)
            if mois_filtre:
                entites["mois_annee"] = mois_filtre
            
            # Vérifier si numéro client disponible
            if not self.slots["contrat_client"] and "contrat_client" not in entites:
                reponse = self._choisir_demande_cc()
                self._enregistrer_echange(message, intention, confiance, reponse)
                return self._formater_retour(reponse, intention, confiance)
            
            # Exécuter l'action sur les données CSV
            try:
                reponse, self.slots = fn(entites, self.slots)
            except Exception as e:
                reponse = f"Erreur lors de la récupération des données: {e}"
            
            self._enregistrer_echange(message, intention, confiance, reponse)
            return self._formater_retour(reponse, intention, confiance)
        
        # ========== CAS 5: Fallback ==========
        reponse = self._choisir_fallback()
        self._enregistrer_echange(message, intention, confiance, reponse)
        return self._formater_retour(reponse, intention, confiance)
    
    def _formater_retour(self, reponse, intention, confiance):
        return {
            "reponse": reponse,
            "intention": intention,
            "confiance": confiance,
            "slots": dict(self.slots),
            "contexte": self._construire_contexte(),
            "tour": self.compteur_tours,
        }
    
    def reinitialiser(self):
        self.slots = {"contrat_client": None, "mois_annee": None}
        self.derniere_intention = None
        self.compteur_tours = 0
        self.historique = []
        print("🔄 Conversation réinitialisée")
    
    def obtenir_historique(self):
        return self.historique