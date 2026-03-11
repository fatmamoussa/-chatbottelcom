# ============================================================
# dialogue.py - Gestionnaire de dialogue from scratch
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 3.0 - Avec gestion des mois/dates
# ============================================================

from actions import ACTIONS_CSV
from config import config


class GestionnaireDialogue:
    """
    Gestionnaire de dialogue avec mémoire (slots)
    - Maintient le contexte de la conversation
    - Gère les demandes de numéro client
    - Gère les filtres de mois/dates
    - Réponses variées pour éviter la répétition
    """
    
    def __init__(self):
        # Mémoire de la conversation
        self.slots = {
            "contrat_client": None,
            "mois_annee": None,
        }
        self.derniere_intention = None
        self.compteur_tours = 0
        self.historique = []  # Historique des échanges
        
        # Index pour alterner les réponses
        self._idx_reponse = 0
        self._idx_demande_cc = 0
        self._idx_fallback = 0
        
        print("✅ Gestionnaire de dialogue initialisé")
    
    def _choisir_reponse(self, liste):
        """Alterner entre les réponses pour éviter la répétition"""
        idx = self._idx_reponse % len(liste)
        self._idx_reponse += 1
        return liste[idx]
    
    def _choisir_demande_cc(self):
        """Choisir une formulation pour demander le numéro client"""
        idx = self._idx_demande_cc % len(config.DIALOGUE.DEMANDE_CC)
        self._idx_demande_cc += 1
        return config.DIALOGUE.DEMANDE_CC[idx]
    
    def _choisir_fallback(self):
        """Choisir un message de fallback"""
        idx = self._idx_fallback % len(config.DIALOGUE.FALLBACK)
        self._idx_fallback += 1
        return config.DIALOGUE.FALLBACK[idx]
    
    def _mettre_a_jour_slots(self, entites):
        """Mettre à jour la mémoire avec les entités extraites"""
        if "contrat_client" in entites:
            self.slots["contrat_client"] = entites["contrat_client"]
        if "mois_annee" in entites:
            self.slots["mois_annee"] = entites["mois_annee"]
    
    def _construire_contexte(self):
        """Construire un résumé du contexte actuel"""
        contexte = []
        if self.slots["contrat_client"]:
            contexte.append(f"Client: {self.slots['contrat_client']}")
        if self.slots["mois_annee"]:
            contexte.append(f"Période: {self.slots['mois_annee']}")
        
        if self.derniere_intention and self.derniere_intention not in ["saluer", "au_revoir", "remercier"]:
            contexte.append(f"Dernière action: {self.derniere_intention}")
        
        return " | ".join(contexte) if contexte else None
    
    def _enregistrer_echange(self, message, intention, confiance, reponse):
        """Enregistrer un échange dans l'historique"""
        self.historique.append({
            "tour": self.compteur_tours,
            "message": message,
            "intention": intention,
            "confiance": confiance,
            "reponse": reponse,
            "slots": dict(self.slots)
        })
        
        # Garder seulement les 10 derniers échanges
        if len(self.historique) > 10:
            self.historique = self.historique[-10:]
    
    def _appliquer_filtre_mois(self, intention, entites):
        """
        Applique le filtre du mois si présent dans les entités ou les slots
        """
        mois = entites.get("mois_annee") or self.slots.get("mois_annee")
        
        if mois:
            print(f"📅 Filtre appliqué : {mois}")
            # Le mois est déjà dans les entités, il sera utilisé par les actions
            return mois
        return None
    
    def traiter(self, message, intention, confiance, entites):
        """
        Traiter un message et retourner une réponse
        
        Args:
            message: Texte de l'utilisateur
            intention: Intention prédite
            confiance: Score de confiance
            entites: Entités extraites
        
        Returns:
            Dict avec réponse et métadonnées
        """
        self.compteur_tours += 1
        
        # Mettre à jour les slots
        self._mettre_a_jour_slots(entites)
        
        # ── Cas 1: Confiance trop faible
        if confiance < config.NLU.SEUIL_CONFIANCE:
            reponse = self._choisir_fallback()
            self._enregistrer_echange(message, intention, confiance, reponse)
            return self._formater_retour(reponse, intention, confiance)
        
        # ── Cas 2: Client donne son numéro
        if intention == "donner_id_client":
            if self.slots["contrat_client"]:
                reponse = f"✅ Numéro client {self.slots['contrat_client']} enregistré."
                
                # Si on attendait une action, la relancer
                if self.derniere_intention in ACTIONS_CSV:
                    fn = ACTIONS_CSV[self.derniere_intention]
                    # 🔥 Appliquer le filtre mois
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
        
        # ── Cas 3: Réponse simple
        if intention in config.DIALOGUE.REPONSES_SIMPLES:
            self.derniere_intention = intention
            reponse = self._choisir_reponse(config.DIALOGUE.REPONSES_SIMPLES[intention])
            self._enregistrer_echange(message, intention, confiance, reponse)
            return self._formater_retour(reponse, intention, confiance)
        
        # ── Cas 4: Action CSV
        if intention in ACTIONS_CSV:
            self.derniere_intention = intention
            fn = ACTIONS_CSV[intention]
            
            # 🔥 Appliquer le filtre mois
            mois_filtre = self._appliquer_filtre_mois(intention, entites)
            if mois_filtre:
                entites["mois_annee"] = mois_filtre
            
            # Vérifier si numéro client disponible
            if not self.slots["contrat_client"] and "contrat_client" not in entites:
                reponse = self._choisir_demande_cc()
                self._enregistrer_echange(message, intention, confiance, reponse)
                return self._formater_retour(reponse, intention, confiance)
            
            reponse, self.slots = fn(entites, self.slots)
            self._enregistrer_echange(message, intention, confiance, reponse)
            return self._formater_retour(reponse, intention, confiance)
        
        # ── Cas 5: Intention inconnue
        reponse = self._choisir_fallback()
        self._enregistrer_echange(message, intention, confiance, reponse)
        return self._formater_retour(reponse, intention, confiance)
    
    def _formater_retour(self, reponse, intention, confiance):
        """Formater le retour standard"""
        return {
            "reponse": reponse,
            "intention": intention,
            "confiance": confiance,
            "slots": dict(self.slots),
            "contexte": self._construire_contexte(),
            "tour": self.compteur_tours,
        }
    
    def reinitialiser(self):
        """Réinitialiser la conversation"""
        self.slots = {
            "contrat_client": None,
            "mois_annee": None,
        }
        self.derniere_intention = None
        self.compteur_tours = 0
        self.historique = []
        print("🔄 Conversation réinitialisée")
    
    def obtenir_historique(self):
        """Retourne l'historique des échanges"""
        return self.historique


if __name__ == "__main__":
    from nlu import ModeleNLU
    from ner import ModeleNER
    from data import DONNEES, diviser_donnees
    
    print("── Test du dialogue ───────────────")
    
    # Charger les modèles
    nlu = ModeleNLU()
    try:
        nlu.charger()
    except:
        print("⚠️  Modèle non trouvé, entraînement...")
        train, val, test = diviser_donnees(DONNEES)
        nlu.entrainer(*train)
        nlu.sauvegarder()
    
    ner = ModeleNER()
    dialogue = GestionnaireDialogue()
    
    # Simulation de conversation
    conversation = [
        "bonjour",
        "je veux voir mes recharges",
        "CC_52099260",
        "merci",
        "au revoir",
    ]
    
    print("\n── Simulation conversation ────────")
    for msg in conversation:
        res_nlu = nlu.predire(msg)
        entites = ner.extraire(msg)
        res = dialogue.traiter(
            msg,
            res_nlu["intention"],
            res_nlu["confiance"],
            entites
        )
        
        print(f"\nVous: {msg}")
        print(f"Bot : {res['reponse']}")
        print(f"      [intention: {res['intention']} | "
              f"confiance: {res['confiance']*100:.1f}% | "
              f"contexte: {res['contexte']}]")