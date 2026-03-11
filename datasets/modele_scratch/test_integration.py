# ============================================================
# test_integration.py - Tests sur données CSV réelles
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 1.0
# ============================================================

import random
import json
from datetime import datetime
from actions import DF_PARC, DF_REFIL, DF_DATA, DF_TRAFIC, DF_ACTIVATION
from config import config


class TestIntegration:
    """Teste le chatbot sur les vraies données clients"""
    
    def __init__(self, chatbot):
        self.chatbot = chatbot
        self.resultats = {
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "reussites": 0,
            "echecs": 0,
            "details": [],
            "par_type": {}
        }
    
    def get_clients_avec_donnees(self):
        """Récupère les clients qui ont des données dans toutes les tables"""
        clients = set()
        
        if not DF_PARC.empty:
            clients.update(DF_PARC["CONTRAT_CLIENT"].dropna().unique())
        
        return list(clients)
    
    def tester_client(self, cc):
        """Teste toutes les requêtes pour un client"""
        print(f"\n▶️  Test client {cc}")
        
        types_requete = config.TEST.TYPES_REQUETES
        
        for type_req in types_requete:
            message = self._generer_message(type_req, cc)
            reponse = self.chatbot.repondre(message)
            
            success = "Aucun" not in reponse["reponse"] and "non trouvé" not in reponse["reponse"].lower()
            
            if success:
                self.resultats["reussites"] += 1
                marker = "✅"
            else:
                self.resultats["echecs"] += 1
                marker = "❌"
            
            self.resultats["total_tests"] += 1
            
            # Stats par type
            if type_req not in self.resultats["par_type"]:
                self.resultats["par_type"][type_req] = {"success": 0, "total": 0}
            self.resultats["par_type"][type_req]["total"] += 1
            if success:
                self.resultats["par_type"][type_req]["success"] += 1
            
            # Détail
            self.resultats["details"].append({
                "client": cc,
                "type": type_req,
                "success": success,
                "message": message,
                "reponse": reponse["reponse"][:100]
            })
            
            print(f"  {marker} {type_req}: {reponse['reponse'][:50]}...")
    
    def _generer_message(self, type_req, cc):
        """Génère un message de test"""
        messages = {
            "offre": f"quelle est mon offre {cc}",
            "recharges": f"mes recharges {cc}",
            "appels": f"mes appels {cc}",
            "internet": f"ma consommation internet {cc}",
            "cout": f"mon cout total {cc}"
        }
        return messages.get(type_req, f"info {cc}")
    
    def run(self, nb_clients=None):
        """Lance les tests"""
        nb_clients = nb_clients or config.TEST.NB_CLIENTS_TEST
        
        print("\n" + "="*70)
        print("  TEST INTÉGRATION - Données CSV réelles")
        print("="*70)
        
        clients = self.get_clients_avec_donnees()
        if not clients:
            print("❌ Aucun client trouvé dans les données")
            return
        
        print(f"📊 {len(clients)} clients disponibles")
        clients_test = random.sample(clients, min(nb_clients, len(clients)))
        
        print(f"🧪 Test sur {len(clients_test)} clients")
        print("-"*70)
        
        for cc in clients_test:
            self.tester_client(cc)
        
        self.afficher_rapport()
        self.sauvegarder_rapport()
    
    def afficher_rapport(self):
        """Affiche le rapport détaillé"""
        total = self.resultats["total_tests"]
        if total == 0:
            print("\n❌ Aucun test effectué")
            return
        
        taux = (self.resultats["reussites"] / total) * 100
        
        print("\n" + "="*70)
        print("  RAPPORT DE TEST INTÉGRATION")
        print("="*70)
        print(f"✅ Réussites: {self.resultats['reussites']}")
        print(f"❌ Échecs: {self.resultats['echecs']}")
        print(f"📊 Taux de succès global: {taux:.1f}%")
        print("-"*70)
        
        print("\n📊 Par type de requête:")
        for type_req, stats in self.resultats["par_type"].items():
            taux_type = (stats["success"] / stats["total"]) * 100
            print(f"  {type_req:<10}: {stats['success']:3d}/{stats['total']:3d} ({taux_type:5.1f}%)")
        
        print("="*70)
        
        # Alerte si en dessous du seuil
        if taux < config.TEST.SEUIL_REUSSITE * 100:
            print(f"\n⚠️  Taux de réussite ({taux:.1f}%) inférieur au seuil ({config.TEST.SEUIL_REUSSITE*100:.0f}%)")
    
    def sauvegarder_rapport(self):
        """Sauvegarde le rapport dans un fichier"""
        chemin = config.CHEMIN_RAPPORT_TEST
        with open(chemin, 'w', encoding='utf-8') as f:
            json.dump(self.resultats, f, ensure_ascii=False, indent=2)
        print(f"\n📁 Rapport sauvegardé: {chemin}")