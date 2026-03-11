# ============================================================
# api.py - Serveur Flask simplifié (sans flask-cors)
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 1.0
# ============================================================

from flask import Flask, request, jsonify, render_template, send_from_directory
import sys
import os
import json
from datetime import datetime

# Ajouter le chemin pour les imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import ChatbotTunisieTelecom
from config import config

app = Flask(__name__, 
           template_folder='templates',
           static_folder='static')

# Configuration CORS manuelle
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

# Initialiser le chatbot
print("\n" + "="*60)
print("  DÉMARRAGE DU SERVEUR API CHATBOT")
print("="*60)

try:
    chatbot = ChatbotTunisieTelecom()
    if not chatbot.charger_modeles():
        print("❌ Erreur: Modèle non trouvé. Lancez d'abord: python main.py --mode train")
        sys.exit(1)
    print("✅ Chatbot chargé avec succès")
except Exception as e:
    print(f"❌ Erreur lors du chargement: {e}")
    sys.exit(1)

# Dictionnaire pour stocker les sessions utilisateur
sessions = {}

@app.route('/')
def index():
    """Page d'accueil"""
    return render_template('index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    """Sert les fichiers statiques"""
    return send_from_directory('static', path)

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Sert les images"""
    return send_from_directory('static/images', filename)

@app.route('/webhook', methods=['POST', 'OPTIONS'])
def webhook():
    """Endpoint compatible avec Rasa"""
    # Gérer les requêtes OPTIONS pour CORS
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        if not data:
            return jsonify([{"text": "Erreur: requête invalide"}]), 400
        
        user_id = data.get('sender', 'default_user')
        message = data.get('message', '').strip()
        
        print(f"\n📨 [{user_id}] Message: {message}")
        
        if not message:
            return jsonify([{"text": "Veuillez entrer un message."}])
        
        # Gérer la session utilisateur
        if user_id not in sessions:
            sessions[user_id] = {
                "historique": [],
                "dernier_message": None
            }
        
        # Obtenir la réponse du chatbot
        reponse = chatbot.repondre(message)
        
        # Sauvegarder dans l'historique
        sessions[user_id]["historique"].append({
            "message": message,
            "reponse": reponse["reponse"],
            "intention": reponse["intention"],
            "confiance": reponse["confiance"],
            "timestamp": datetime.now().isoformat()
        })
        
        print(f"📤 Réponse: {reponse['reponse'][:100]}...")
        print(f"   Intention: {reponse['intention']} ({reponse['confiance']*100:.1f}%)")
        
        return jsonify([{"text": reponse["reponse"]}])
    
    except Exception as e:
        print(f"❌ Erreur: {e}")
        return jsonify([{"text": f"Désolé, une erreur s'est produite: {str(e)}"}]), 500

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat_api():
    """API alternative plus détaillée"""
    if request.method == 'OPTIONS':
        return '', 200
    
    try:
        data = request.json
        message = data.get('message', '').strip()
        user_id = data.get('user_id', 'default')
        
        if not message:
            return jsonify({"error": "Message vide"}), 400
        
        reponse = chatbot.repondre(message)
        
        return jsonify({
            "success": True,
            "reponse": reponse["reponse"],
            "intention": reponse["intention"],
            "confiance": reponse["confiance"],
            "contexte": reponse.get("contexte"),
            "slots": reponse.get("slots", {})
        })
    
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/reset', methods=['POST', 'OPTIONS'])
def reset_session():
    """Réinitialise une session utilisateur"""
    if request.method == 'OPTIONS':
        return '', 200
    
    data = request.json
    user_id = data.get('user_id', 'default')
    
    if user_id in sessions:
        del sessions[user_id]
    
    chatbot.dialogue.reinitialiser()
    
    return jsonify({"success": True, "message": "Session réinitialisée"})

@app.route('/api/health', methods=['GET'])
def health():
    """Vérifie que le serveur fonctionne"""
    return jsonify({
        "status": "ok",
        "modele_charge": chatbot.modele_charge,
        "timestamp": datetime.now().isoformat(),
        "sessions_actives": len(sessions)
    })

@app.route('/api/client/<cc>', methods=['GET'])
def get_client_info(cc):
    """Retourne les informations d'un client"""
    try:
        from actions import DF_PARC, client_existe
        
        if not client_existe(cc):
            return jsonify({"error": "Client non trouvé"}), 404
        
        client = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc].iloc[0].to_dict()
        
        # Convertir les types non sérialisables
        for k, v in client.items():
            if hasattr(v, 'isoformat'):  # Pour les dates
                client[k] = v.isoformat() if v else None
            elif pd.isna(v):  # Pour les NaN
                client[k] = None
        
        return jsonify({"success": True, "data": client})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n🚀 Serveur démarré sur http://localhost:5005")
    print("📝 Endpoints disponibles:")
    print("   • GET  /              - Interface web")
    print("   • POST /webhook        - Compatible Rasa")
    print("   • POST /api/chat       - API détaillée")
    print("   • POST /api/reset      - Réinitialiser session")
    print("   • GET  /api/health     - Vérification santé")
    print("   • GET  /api/client/<cc> - Infos client")
    print("\n" + "="*60)
    print("⚠️  Assurez-vous d'avoir un dossier 'templates' avec index.html")
    print("   et un dossier 'static/images' avec vos images")
    print("="*60)
    
    app.run(host='0.0.0.0', port=5005, debug=True, threaded=True)