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

# Importer le bon modèle
from nlu_svm_optuna_mlflow import NLUSVMOptunaMLflow

app = Flask(__name__, template_folder='templates')
CORS(app)

class ChatbotTunisieTelecom:
    """Chatbot principal avec SVM + Optuna"""
    
    def __init__(self):
        self.nlu = None
        self.ner = NER()
        self.modele_charge = False
        self.initialise = False
    
    def initialiser(self):
        if self.initialise:
            return
        
        self.nlu = NLUSVMOptunaMLflow(use_optuna=False, use_mlflow=False)
        self.initialise = True
    
    def charger_modeles(self, chemin_modele="modele_scratch/modele_svm_optuna_mlflow.pkl"):
        """Charge le modèle SVM avec la bonne structure"""
        self.initialiser()
        try:
            # Vérifier si le fichier existe
            if not os.path.exists(chemin_modele):
                # Essayer l'autre nom de fichier
                chemin_modele = "modele_scratch/modele_svm_full.pkl"
                if not os.path.exists(chemin_modele):
                    print(f"❌ Fichier modèle introuvable")
                    return False
            
            # Charger avec la méthode de la classe NLUSVMOptunaMLflow
            self.nlu.charger(chemin_modele)
            self.modele_charge = True
            print(f"✅ Modèle chargé: {chemin_modele}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def repondre(self, message):
        if not self.modele_charge:
            return {
                "reponse": "Modèle non chargé. Veuillez d'abord entraîner le modèle avec: python main.py --mode train_svm",
                "erreur": True,
                "intention": None,
                "confiance": 0
            }
        
        entites = self.ner.extraire(message)
        resultat_nlu = self.nlu.predire(message)
        
        # Utiliser le gestionnaire de dialogue si disponible
        try:
            from dialogue import GestionnaireDialogue
            if not hasattr(self, 'dialogue'):
                self.dialogue = GestionnaireDialogue(modele_nlu=self.nlu)
            
            reponse = self.dialogue.traiter(
                message,
                resultat_nlu["intention"],
                resultat_nlu["confiance"],
                entites
            )
            return reponse
        except:
            # Fallback simple
            return {
                "reponse": f"Intention détectée: {resultat_nlu['intention']} (confiance: {resultat_nlu['confiance']*100:.1f}%)",
                "intention": resultat_nlu["intention"],
                "confiance": resultat_nlu["confiance"],
                "entites_detectees": entites
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

# ... (reste du code api.py identique)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Endpoint compatible avec le frontend"""
    try:
        data = request.json
        message = data.get('message', '').strip()
        sender = data.get('sender', 'default')
        
        print(f"\n📨 [WEBHOOK] {sender}: {message}")
        
        if not message:
            return jsonify([{"text": "Message vide"}]), 200
        
        entites = ner.extraire(message)
        reponse = chatbot.repondre(message)
        
        return jsonify([{"text": reponse["reponse"]}])
        
    except Exception as e:
        print(f"❌ Erreur webhook: {e}")
        import traceback
        traceback.print_exc()
        return jsonify([{"text": f"Désolé, une erreur est survenue."}]), 200

# ... (reste du code api.py)
# ============================================================
# ENDPOINTS EXISTANTS
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
            },
            "error": str(e)
        })


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
        
        entites = ner.extraire(message)
        reponse = chatbot.repondre(message)
        
        if "entites_detectees" not in reponse:
            reponse["entites_detectees"] = entites
        
        sessions[session_id]["historique"].append({
            "message": message,
            "reponse": reponse["reponse"],
            "intention": reponse.get("intention"),
            "confiance": reponse.get("confiance"),
            "entites": entites,
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"📤 Réponse: {reponse['reponse'][:100]}...")
        print(f"   Entités: {entites}")
        
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
        "ner_loaded": True,
        "sessions_active": len(sessions),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/stats', methods=['GET'])
def stats():
    """Statistiques du chatbot"""
    try:
        from data import compter_intentions
        compteurs = compter_intentions()
        
        return jsonify({
            "success": True,
            "intentions": len(compteurs),
            "exemples_total": sum(compteurs.values()),
            "repartition": compteurs,
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