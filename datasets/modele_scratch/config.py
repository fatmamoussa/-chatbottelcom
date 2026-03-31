# ========================================================
# config.py - Configuration complète
# Projet : Chatbot Tunisie Telecom
# Version : 6.3 - Compatible Windows local + Docker
# ========================================================

import os
from pathlib import Path

# Répertoire racine du projet (là où se trouve ce fichier)
BASE_DIR = Path(__file__).resolve().parent

# ========================================================
# CHEMIN DATASET :
#   - Docker   : variable d'env DATASET_PATH=/app/datasets
#   - Windows  : chemin fixe D:\chatbot_telecom\datasets
# ========================================================
_dataset_env = os.environ.get("DATASET_PATH", "")
if _dataset_env:
    # On est dans Docker
    CHEMIN_DATASET_FINAL = _dataset_env
else:
    # On est en local Windows — chemin fixe
    CHEMIN_DATASET_FINAL = r"D:\chatbot_telecom\datasets"


class Config:
    """Configuration globale du projet"""

    # ========================================================
    # CHEMINS
    # ========================================================
    CHEMIN_DATASET      = CHEMIN_DATASET_FINAL
    CHEMIN_MODELE_NLU   = str(BASE_DIR / "modele_scratch" / "modele_svm_full.pkl")
    CHEMIN_RAPPORT_TEST = str(BASE_DIR / "modele_scratch" / "modele_svm_full_best_params.json")

    # ========================================================
    # PARAMÈTRES NLU
    # ========================================================
    class NLU:
        # TF-IDF
        TFIDF_MAX_FEATURES  = 5000
        TFIDF_NGRAM_RANGE   = (1, 4)
        TFIDF_SUBLINEAR_TF  = True
        TFIDF_MIN_DF        = 1
        TFIDF_MAX_DF        = 0.95
        TFIDF_USE_IDF       = True

        # SVM
        SVM_KERNEL          = 'linear'
        SVM_C               = 1.0
        SVM_PROBABILITY     = True
        SVM_CLASS_WEIGHT    = 'balanced'
        SVM_GAMMA           = 'scale'

        # Seuils
        SEUIL_CONFIANCE     = 0.20
        SEUIL_AMBIGUITE     = 0.20

        # Entraînement
        RANDOM_STATE        = 42
        TEST_SIZE           = 0.15
        VAL_SIZE            = 0.15

        # Prétraitement
        UTILISER_LEMMATISATION  = True
        MIN_TOKEN_LENGTH        = 2

    # ========================================================
    # PARAMÈTRES NER
    # ========================================================
    class NER:
        PATTERN_CONTRAT     = r'CC_[0-9]{8}'
        PATTERN_MOIS_ANNEE  = r'(0[1-9]|1[0-2])/(20[2-9][0-9]|202[3-9])'
        PATTERN_MONTANT     = r'(\d+[.,]?\d*)\s*(DT|dinars|tnd)'

        MOIS_MAP = {
            "janvier": "01", "février": "02", "mars": "03",
            "avril": "04", "mai": "05", "juin": "06",
            "juillet": "07", "août": "08", "septembre": "09",
            "octobre": "10", "novembre": "11", "décembre": "12",
        }

    # ========================================================
    # PARAMÈTRES DIALOGUE
    # ========================================================
    class DIALOGUE:
        DEMANDE_CC = [
            "Quel est votre numéro client ? (format CC_XXXXXXXX)",
            "Pour accéder à vos informations, veuillez fournir votre numéro client.",
            "Pouvez-vous me communiquer votre numéro de contrat ? (ex: CC_52099260)"
        ]

        FALLBACK = [
            "Je n'ai pas bien compris votre demande. Pouvez-vous reformuler ?",
            "Désolé, je n'ai pas saisi votre demande.",
            "Je ne suis pas sûr de comprendre. Pouvez-vous préciser ?"
        ]

        REPONSES_SIMPLES = {
            "saluer":    [
                "Bonjour ! Comment puis-je vous aider ?",
                "Bonjour ! Je suis l'assistant Tunisie Telecom, que puis-je faire pour vous ?"
            ],
            "au_revoir": [
                "Au revoir ! Merci d'avoir contacté Tunisie Telecom.",
                "Bonne journée ! N'hésitez pas à revenir."
            ],
            "remercier": ["Avec plaisir !", "Je suis là pour vous aider !"],
            "affirmer":  ["Parfait, je continue."],
            "nier":      ["D'accord, que souhaitez-vous faire ?"]
        }

        # Liste complète des actions CSV (55 intentions)
        ACTIONS_CSV = [
            # data_activation
            "voir_nbr_activation", "voir_cout_activation", "voir_services_actives",
            "voir_code_ussd", "voir_option_ussd", "voir_mois_activation", "voir_offre_activation",
            # data_data
            "voir_volume_internet", "voir_nbr_sessions", "voir_cout_internet",
            "voir_type_trafic", "voir_taxation_internet", "voir_reseau_internet",
            "voir_heure_connexion", "voir_duree_session", "voir_conso_quotidienne",
            # data_parc
            "voir_offre_commerciale", "voir_offre_commerciale_detail",
            "voir_segment_client", "voir_date_activation", "voir_date_resiliation",
            "voir_statut_client", "consulter_offre", "info_client",
            # data_refil
            "voir_nbr_recharge", "voir_montant_recharge", "voir_bonus_recharge",
            "voir_type_recharge", "voir_recharge_moyenne", "voir_plus_grosse_recharge",
            "voir_derniere_recharge", "voir_recharges",
            # data_trafic
            "voir_nbr_appel", "voir_duree_appel", "voir_cout_appel",
            "voir_type_trafic_appel", "voir_taxation_appel", "voir_reseau_appel",
            "voir_destination_appel", "voir_appels_sortants", "voir_appels_entrants",
            "voir_sms", "voir_appels_longue_duree", "voir_appels_courte_duree",
            "historique_appels",
            # générales
            "forfaits_actives", "cout_total", "cout_total_mois", "comparaison_cout",
        ]

    # ========================================================
    # PARAMÈTRES DATASET
    # ========================================================
    class DATASET:
        FICHIERS = {
            "parc":       "data_parc.csv",
            "activation": "data_activation.csv",
            "data":       "data_data.csv",
            "refil":      "data_refil.csv",
            "trafic":     "data_trafic.csv",
        }

    # ========================================================
    # PARAMÈTRES TEST INTÉGRATION
    # ========================================================
    class TEST:
        NB_CLIENTS_TEST = 50
        SEUIL_REUSSITE  = 0.80

        TYPES_REQUETES = [
            "offre", "recharges", "appels",
            "internet", "cout", "statut",
        ]

        MESSAGES_TEST = {
            "offre":     "quelle est mon offre {cc}",
            "recharges": "mes recharges {cc}",
            "appels":    "mes appels {cc}",
            "internet":  "ma consommation internet {cc}",
            "cout":      "mon cout total {cc}",
            "statut":    "mon statut {cc}",
        }

    # ========================================================
    # PARAMÈTRES SERVEUR API
    # ========================================================
    class API:
        HOST  = os.environ.get("API_HOST", "0.0.0.0")
        PORT  = int(os.environ.get("API_PORT", 5005))
        DEBUG = os.environ.get("API_DEBUG", "false").lower() == "true"

    # ========================================================
    # PARAMÈTRES MLFLOW
    # ========================================================
    class MLFLOW:
        TRACKING_URI   = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlflow_logs")
        EXPERIMENT_SVM = "svm_telecom"
        EXPERIMENT_XGB = "xgboost_telecom"
        EXPERIMENT_RF  = "randomforest_telecom"


# Instance globale
config = Config()


# ========================================================
# VÉRIFICATION AU DÉMARRAGE (optionnelle)
# ========================================================

def verifier_configuration():
    """Vérifie que tous les fichiers nécessaires sont présents."""
    print("\n" + "=" * 60)
    print("  VÉRIFICATION DE LA CONFIGURATION")
    print("=" * 60)

    print(f"📁 Datasets : {config.CHEMIN_DATASET}")
    dataset_path = Path(config.CHEMIN_DATASET)

    if dataset_path.exists():
        for nom, fichier in config.DATASET.FICHIERS.items():
            chemin = dataset_path / fichier
            if chemin.exists():
                taille = chemin.stat().st_size / 1024
                print(f"   ✅ {fichier:<30} ({taille:.0f} Ko)")
            else:
                print(f"   ❌ {fichier:<30} MANQUANT")
    else:
        print(f"   ⚠️  Dossier introuvable : {config.CHEMIN_DATASET}")

    print(f"\n🤖 Modèle : {config.CHEMIN_MODELE_NLU}")
    if Path(config.CHEMIN_MODELE_NLU).exists():
        taille_mb = Path(config.CHEMIN_MODELE_NLU).stat().st_size / (1024 * 1024)
        print(f"   ✅ Modèle trouvé ({taille_mb:.1f} Mo)")
    else:
        print(f"   ⚠️  Modèle introuvable → python main.py --mode train_svm")

    print(f"\n🌐 API : http://{config.API.HOST}:{config.API.PORT}")
    print("=" * 60)


if __name__ == "__main__":
    verifier_configuration()