# ============================================================
# main.py - Point d'entrée principal
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 3.0 - Avec test intégration
# ============================================================

import argparse
import sys

from nlu import ModeleNLU, entrainer_modele_complet
from ner import ModeleNER
from dialogue import GestionnaireDialogue
from test_integration import TestIntegration
from config import config


class ChatbotTunisieTelecom:
    """Chatbot principal"""
    
    def __init__(self):
        self.nlu = ModeleNLU()
        self.ner = ModeleNER()
        self.dialogue = GestionnaireDialogue()
        self.modele_charge = False
    
    def charger_modeles(self):
        """Charge les modèles"""
        try:
            self.nlu.charger()
            self.modele_charge = True
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def repondre(self, message):
        """Génère une réponse"""
        if not self.modele_charge:
            return {"reponse": "⚠️ Modèle non chargé", "erreur": True}
        
        resultat_nlu = self.nlu.predire(message)
        entites = self.ner.extraire(message)
        
        return self.dialogue.traiter(
            message,
            resultat_nlu["intention"],
            resultat_nlu["confiance"],
            entites
        )


# ========================================================
# MODES D'EXÉCUTION
# ========================================================

def mode_entrainement():
    """Mode entraînement"""
    print("\n" + "="*60)
    print("  MODE ENTRAÎNEMENT")
    print("="*60)
    entrainer_modele_complet()


def mode_chat():
    """Mode chatbot interactif"""
    print("\n" + "="*60)
    print("  MODE CHATBOT INTERACTIF")
    print("="*60)
    
    bot = ChatbotTunisieTelecom()
    
    if not bot.charger_modeles():
        print("❌ Modèle non trouvé. Lancez d'abord l'entraînement.")
        return
    
    print("\n" + "-"*60)
    print("Bot: Bonjour ! Assistant Tunisie Telecom")
    print("Commandes: /quit, /reset, /stats")
    print("-"*60)
    
    while True:
        try:
            message = input("\nVous: ").strip()
            
            if message.lower() in ['/quit', 'quit', 'exit']:
                print("Bot: Au revoir !")
                break
            elif message.lower() == '/reset':
                bot.dialogue.reinitialiser()
                continue
            elif not message:
                continue
            
            reponse = bot.repondre(message)
            
            print(f"Bot: {reponse['reponse']}")
            if reponse.get('contexte'):
                print(f"     [{reponse['contexte']}]")
            print(f"     [{reponse['intention']} | {reponse['confiance']*100:.1f}%]")
            
        except KeyboardInterrupt:
            print("\nBot: Au revoir !")
            break

def mode_test():
    """Mode test unitaire"""
    print("\n" + "="*60)
    print("  MODE TEST UNITAIRE")
    print("="*60)
    
    bot = ChatbotTunisieTelecom()
    
    if not bot.charger_modeles():
        print("❌ Modèle non trouvé. Lancez d'abord l'entraînement.")
        return
    
    tests = [
        "bonjour",
        "je veux mon offre",
        "CC_52099260",
        "mes recharges",
        "ma consommation internet",
        "merci",
        "au revoir",
    ]
    
    print("\n📋 Tests automatiques:")
    print("-"*60)
    
    for test in tests:
        reponse = bot.repondre(test)
        print(f"\nVous: {test}")
        print(f"Bot : {reponse['reponse']}")
        print(f"     [{reponse['intention']} | {reponse['confiance']*100:.1f}%]")
    
    print("\n✅ Tests terminés")


def mode_integration():
    """Mode test sur données réelles"""
    print("\n" + "="*60)
    print("  MODE TEST INTÉGRATION - Données CSV réelles")
    print("="*60)
    
    bot = ChatbotTunisieTelecom()
    
    if not bot.charger_modeles():
        print("❌ Modèle non trouvé. Lancez d'abord l'entraînement.")
        return
    
    testeur = TestIntegration(bot)
    testeur.run(nb_clients=50)  # Test sur 50 clients aléatoires


def mode_stats():
    """Affiche les statistiques du modèle"""
    print("\n" + "="*60)
    print("  STATISTIQUES DU MODÈLE")
    print("="*60)
    
    try:
        with open(config.CHEMIN_RAPPORT_TEST, 'r', encoding='utf-8') as f:
            rapport = json.load(f)
        
        print(f"\n📊 Dernier test: {rapport['timestamp']}")
        print(f"✅ Réussites: {rapport['reussites']}/{rapport['total_tests']} ({rapport['reussites']/rapport['total_tests']*100:.1f}%)")
        print("\n📊 Par type de requête:")
        for type_req, stats in rapport['par_type'].items():
            taux = stats['success']/stats['total']*100
            print(f"  {type_req:<10}: {stats['success']:3d}/{stats['total']:3d} ({taux:5.1f}%)")
            
    except FileNotFoundError:
        print("❌ Aucun rapport de test trouvé. Lancez d'abord le mode intégration.")


# ========================================================
# POINT D'ENTRÉE
# ========================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chatbot Tunisie Telecom - From Scratch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes disponibles:
  train       : Entraîner le modèle NLU
  chat        : Lancer le chatbot interactif
  test        : Tester le modèle sur des exemples
  integration : Tester sur données CSV réelles
  stats       : Afficher les statistiques des tests
        
Exemples:
  python main.py --mode train
  python main.py --mode chat
  python main.py --mode integration
        """
    )
    
    parser.add_argument(
        "--mode",
        choices=["train", "chat", "test", "integration", "stats"],
        default="chat",
        help="Mode d'exécution (défaut: chat)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="Chatbot Tunisie Telecom v3.0"
    )
    
    args = parser.parse_args()
    
    # Exécution selon le mode
    if args.mode == "train":
        mode_entrainement()
    elif args.mode == "test":
        mode_test()
    elif args.mode == "integration":
        mode_integration()
    elif args.mode == "stats":
        mode_stats()
    else:
        mode_chat()