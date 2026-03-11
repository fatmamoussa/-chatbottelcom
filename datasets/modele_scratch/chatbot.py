# ============================================================
# chatbot.py - Point d'entrée alternatif (mode terminal simple)
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 2.0 - Corrigé (imports alignés avec actions.py v5)
#
# NOTE : Pour la production, utilisez main.py --mode chat
#        Ce fichier est conservé pour compatibilité et tests rapides.
# ============================================================

from nlu      import ModeleNLU
from ner      import ModeleNER
from dialogue import GestionnaireDialogue
from config   import config


class Chatbot:
    """Chatbot Tunisie Telecom — wrapper simplifié"""

    def __init__(self, charger_modele=True):
        print("=" * 50)
        print("  Chatbot Tunisie Telecom")
        print("=" * 50)

        self.nlu      = ModeleNLU()
        self.ner      = ModeleNER()
        self.dialogue = GestionnaireDialogue()

        if charger_modele:
            try:
                self.nlu.charger()
                print("✅ Modèle NLU chargé")
            except FileNotFoundError:
                print("⚠️  Modèle non trouvé. Lancement de l'entraînement...")
                from nlu import entrainer_modele_complet
                entrainer_modele_complet()
                self.nlu.charger()

        print("✅ Chatbot prêt !\n")

    def repondre(self, message):
        """Génère une réponse pour un message donné"""
        resultat_nlu = self.nlu.predire(message)
        entites      = self.ner.extraire(message)

        return self.dialogue.traiter(
            message,
            resultat_nlu["intention"],
            resultat_nlu["confiance"],
            entites
        )


# ── Mode terminal ────────────────────────────────────────────
if __name__ == "__main__":
    bot = Chatbot()

    print("Tapez 'quit' pour quitter | '/reset' pour réinitialiser\n")
    print("-" * 50)

    while True:
        try:
            message = input("Vous : ").strip()
            if not message:
                continue
            if message.lower() in ["quit", "exit", "quitter"]:
                print("Bot : Au revoir !")
                break
            if message.lower() == "/reset":
                bot.dialogue.reinitialiser()
                print("Bot : Conversation réinitialisée.")
                continue

            res = bot.repondre(message)

            print(f"Bot  : {res['reponse']}")
            source = res.get('source', 'svm')
            print(
                f"      [intention: {res['intention']} "
                f"| confiance: {res['confiance']*100:.1f}%"
                f" | source: {source.upper()}"
                f" | entités: {res.get('slots', {})}]"
            )
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nBot : Au revoir !")
            break