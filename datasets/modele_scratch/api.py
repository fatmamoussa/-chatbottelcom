# api.py - Version corrigée
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import pickle
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ner import NER
from config import config

# Importer la bonne classe (celle utilisée par main.py)
from main import NLUSVM  # Changé ici

app = Flask(__name__, template_folder='templates')
CORS(app)

class ChatbotTunisieTelecom:
    """Chatbot principal avec SVM"""
    
    def __init__(self):
        self.nlu = None
        self.ner = None
        self.dialogue = None
        self.modele_charge = False
        self.initialise = False
    
    def initialiser(self):
        if self.initialise:
            return
        
        try:
            self.nlu = NLUSVM()  # Utilise la classe de main.py
            self.ner = NER()
            self.initialise = True
        except Exception as e:
            print(f"❌ Erreur initialisation: {e}")
    
    def charger_modeles(self, chemin_modele="modele_scratch/modele_svm_full.pkl"):
        """Charge le modèle SVM"""
        self.initialiser()
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(chemin_modele):
                print(f"❌ Fichier modèle introuvable: {chemin_modele}")
                return False
            
            # Charger avec la méthode de NLUSVM
            self.nlu.charger(chemin_modele)
            self.modele_charge = True
            print(f"✅ Modèle chargé: {chemin_modele}")
            
            # Initialiser le gestionnaire de dialogue
            try:
                from dialogue import GestionnaireDialogue
                self.dialogue = GestionnaireDialogue(modele_nlu=self.nlu)
                print("✅ Gestionnaire de dialogue initialisé")
            except Exception as e:
                print(f"⚠️ Dialogue non disponible: {e}")
                self.dialogue = None
            
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def repondre(self, message):
        if not self.modele_charge or self.nlu is None:
            return {
                "reponse": "Modèle non chargé. Veuillez d'abord entraîner le modèle avec: python main.py --mode train_svm",
                "erreur": True,
                "intention": None,
                "confiance": 0
            }
        
        # Extraire les entités
        if self.ner:
            entites = self.ner.extraire(message)
        else:
            entites = {}
        
        # Prédire l'intention
        resultat_nlu = self.nlu.predire(message)
        
        # Utiliser le gestionnaire de dialogue si disponible
        if self.dialogue:
            try:
                reponse = self.dialogue.traiter(
                    message,
                    resultat_nlu["intention"],
                    resultat_nlu["confiance"],
                    entites
                )
                return reponse
            except Exception as e:
                print(f"Erreur dialogue: {e}")
        
        # Fallback simple
        return {
            "reponse": f"Intention détectée: {resultat_nlu['intention']} (confiance: {resultat_nlu['confiance']*100:.1f}%)",
            "intention": resultat_nlu["intention"],
            "confiance": resultat_nlu["confiance"],
            "entites_detectees": entites,
            "erreur": False
        }

# Initialiser le chatbot
print("\n" + "="*60)
print("  DÉMARRAGE DU SERVEUR API")
print("="*60)

chatbot = ChatbotTunisieTelecom()
ner = NER()

# Charger le modèle
if not chatbot.charger_modeles():
    print("❌ Modèle non trouvé. Lancez d'abord: python main.py --mode train_svm")
    # Continuer quand même pour les endpoints de test
else:
    print("✅ Chatbot chargé avec succès")

print("✅ NER chargé avec succès")

# Stockage des sessions
sessions = {}


# ============================================================
# ENDPOINTS
# ============================================================

@app.route('/', methods=['GET'])
def index():
    """Page d'accueil - interface web"""
    try:
        return send_from_directory('templates', 'index.html')
    except Exception as e:
        return jsonify({
            "message": "Bienvenue sur l'API du chatbot Tunisie Telecom",
            "endpoints": {
                "POST /webhook": "Envoyer un message (frontend)",
                "POST /api/chat": "Envoyer un message (JSON)",
                "GET /api/health": "Vérifier l'état",
                "GET /api/stats": "Statistiques"
            }
        })


@app.route('/webhook', methods=['POST', 'OPTIONS'])
def webhook():
    """Endpoint compatible avec le frontend"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        message = data.get('message', '').strip()
        sender = data.get('sender', 'default')
        
        print(f"\n📨 [WEBHOOK] {sender}: {message}")
        
        if not message:
            return jsonify([{"text": "Message vide"}]), 200
        
        reponse = chatbot.repondre(message)
        
        return jsonify([{"text": reponse["reponse"]}])
        
    except Exception as e:
        print(f"❌ Erreur webhook: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([{"text": f"Désolé, une erreur est survenue."}]), 200


@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Endpoint principal du chatbot (format JSON)"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Requête invalide"}), 400
        
        message = data.get('message', '').strip()
        session_id = data.get('session_id', 'default')
        
        print(f"\n📨 [API] {session_id}: {message}")
        
        if not message:
            return jsonify({"error": "Message vide"}), 400
        
        if session_id not in sessions:
            sessions[session_id] = {
                "historique": [],
                "dernier_message": None,
                "contexte": {}
            }
        
        entites = ner.extraire(message) if ner else {}
        reponse = chatbot.repondre(message)
        
        sessions[session_id]["historique"].append({
            "message": message,
            "reponse": reponse["reponse"],
            "intention": reponse.get("intention"),
            "confiance": reponse.get("confiance"),
            "entites": entites,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"📤 Réponse: {reponse['reponse'][:100]}...")
        
        return jsonify({
            "success": True,
            "reponse": reponse["reponse"],
            "intention": reponse.get("intention"),
            "confiance": reponse.get("confiance"),
            "entites": entites,
            "session_id": session_id
        })
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/api/session/<session_id>', methods=['GET'])
def get_session(session_id):
    """Récupère l'historique d'une session"""
    if session_id in sessions:
        return jsonify({
            "success": True,
            "historique": sessions[session_id]["historique"][-10:]
        })
    return jsonify({"success": False, "error": "Session non trouvée"}), 404


@app.route('/api/session/<session_id>', methods=['DELETE'])
def reset_session(session_id):
    """Réinitialise une session"""
    if session_id in sessions:
        del sessions[session_id]
    return jsonify({"success": True})


@app.route('/api/health', methods=['GET'])
def health():
    """Vérification de santé"""
    return jsonify({
        "status": "ok",
        "model_loaded": chatbot.modele_charge,
        "model_type": "SVM + Embeddings",
        "ner_loaded": ner is not None,
        "sessions_active": len(sessions),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Statistiques du chatbot"""
    try:
        if chatbot.modele_charge and chatbot.nlu:
            stats = chatbot.nlu.stats
            return jsonify({
                "success": True,
                "total_predictions": stats.get('total_predictions', 0),
                "avg_confidence": stats.get('avg_confidence', 0),
                "train_accuracy": stats.get('train_accuracy', 0),
                "intentions": len(chatbot.nlu.label_encoder.classes_) if hasattr(chatbot.nlu, 'label_encoder') else 0,
                "sessions_actives": len(sessions)
            })
        else:
            return jsonify({
                "success": True,
                "model_loaded": False,
                "sessions_actives": len(sessions)
            })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  API CHATBOT TUNISIE TELECOM")
    print("="*60)
    print("🚀 Serveur démarré sur http://localhost:5005")
    print("📝 Endpoints:")
    print("   POST /webhook    - Envoyer un message (frontend)")
    print("   POST /api/chat   - Envoyer un message (JSON)")
    print("   GET  /api/health - Vérification santé")
    print("   GET  /api/stats  - Statistiques")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5005, debug=True)