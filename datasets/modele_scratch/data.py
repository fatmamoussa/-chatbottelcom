# ============================================================
# data.py - Données NLU
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 7.0 - 3 intentions faibles corrigées
#
# CORRECTIONS v7.0 :
# 1. voir_services_actives (F1:0.42) → exemples clarifiés
#    DÉFINITION : ce que tu as ACTIVÉ (actions passées via USSD)
# 2. forfaits_actives (F1:0.55) → exemples clarifiés
#    DÉFINITION : tes FORFAITS/ABONNEMENTS en cours (état actuel)
# 3. voir_appels_sortants (F1:0.67) → doublé de 10 → 20 exemples
#    + suppression du doublon "appels effectués" avec historique_appels
# ============================================================

import random
from sklearn.model_selection import train_test_split

DONNEES = [

    # ========================================================
    # INTENTIONS GÉNÉRALES
    # ========================================================

    # saluer
    ("bonjour", "saluer"),
    ("salut", "saluer"),
    ("bonsoir", "saluer"),
    ("hello", "saluer"),
    ("bonjour à tous", "saluer"),
    ("salut l'assistant", "saluer"),
    ("bonsoir besoin d'aide", "saluer"),
    ("hey", "saluer"),
    ("coucou", "saluer"),
    ("bonjour monsieur", "saluer"),
    ("bonjour madame", "saluer"),
    ("salut comment tu vas", "saluer"),
    ("bonsoir je cherche de l'aide", "saluer"),
    ("hello je suis là", "saluer"),
    ("bonjour j'ai besoin de vous", "saluer"),

    # au_revoir
    ("au revoir", "au_revoir"),
    ("bye", "au_revoir"),
    ("à bientôt", "au_revoir"),
    ("merci au revoir", "au_revoir"),
    ("bonne journée", "au_revoir"),
    ("à plus tard", "au_revoir"),
    ("je vous quitte", "au_revoir"),
    ("au revoir et merci", "au_revoir"),
    ("bonne soirée", "au_revoir"),
    ("à très bientôt", "au_revoir"),
    ("ciao", "au_revoir"),
    ("bonne continuation", "au_revoir"),
    ("à la prochaine", "au_revoir"),
    ("on se revoit bientôt", "au_revoir"),
    ("je vous dis au revoir", "au_revoir"),

    # remercier
    ("merci", "remercier"),
    ("merci beaucoup", "remercier"),
    ("je vous remercie", "remercier"),
    ("c'est gentil merci", "remercier"),
    ("parfait merci", "remercier"),
    ("merci pour votre aide", "remercier"),
    ("super merci", "remercier"),
    ("merci c'est bon", "remercier"),
    ("merci bien", "remercier"),
    ("je te remercie", "remercier"),
    ("c'est parfait merci beaucoup", "remercier"),
    ("merci pour l'information", "remercier"),
    ("grand merci", "remercier"),
    ("merci mille fois", "remercier"),
    ("merci infiniment", "remercier"),

    # affirmer
    ("oui", "affirmer"),
    ("oui c'est ça", "affirmer"),
    ("exactement", "affirmer"),
    ("tout à fait", "affirmer"),
    ("oui bien sûr", "affirmer"),
    ("c'est correct", "affirmer"),
    ("oui je confirme", "affirmer"),
    ("oui s'il vous plaît", "affirmer"),
    ("oui c'est exact", "affirmer"),
    ("absolument", "affirmer"),
    ("parfaitement", "affirmer"),
    ("oui d'accord", "affirmer"),

    # nier
    ("non", "nier"),
    ("non merci", "nier"),
    ("pas du tout", "nier"),
    ("non ce n'est pas ça", "nier"),
    ("non je ne veux pas", "nier"),
    ("incorrect", "nier"),
    ("non c'est faux", "nier"),
    ("je ne veux pas", "nier"),
    ("non pas ça", "nier"),
    ("ce n'est pas correct", "nier"),
    ("non je refuse", "nier"),
    ("jamais", "nier"),

    # ========================================================
    # NUMÉRO CLIENT
    # ========================================================

    ("CC_52099260", "donner_id_client"),
    ("CC_57059881", "donner_id_client"),
    ("CC_12345678", "donner_id_client"),
    ("CC_87654321", "donner_id_client"),
    ("CC_11223344", "donner_id_client"),
    ("CC_55667788", "donner_id_client"),
    ("CC_99887766", "donner_id_client"),
    ("CC_33445566", "donner_id_client"),
    ("CC_77889900", "donner_id_client"),
    ("mon numéro client est CC_52099260", "donner_id_client"),
    ("voici mon contrat CC_57059881", "donner_id_client"),
    ("mon identifiant CC_12345678", "donner_id_client"),
    ("code client CC_87654321", "donner_id_client"),
    ("contrat numéro CC_11223344", "donner_id_client"),
    ("CC_52099260 c'est mon numéro", "donner_id_client"),
    ("voilà mon code CC_55667788", "donner_id_client"),
    ("mon id client CC_99887766", "donner_id_client"),
    ("numéro de contrat CC_33445566", "donner_id_client"),
    ("je suis le client CC_77889900", "donner_id_client"),
    ("CC_54729922", "donner_id_client"),

    # ========================================================
    # ACTIVATIONS
    # ========================================================

    # voir_nbr_activation
    ("nombre d'activations", "voir_nbr_activation"),
    ("combien d'activations j'ai fait", "voir_nbr_activation"),
    ("total des activations", "voir_nbr_activation"),
    ("compteur d'activations", "voir_nbr_activation"),
    ("combien de fois j'ai activé", "voir_nbr_activation"),
    ("nombre d'activations de services", "voir_nbr_activation"),
    ("total de mes activations", "voir_nbr_activation"),
    ("mes activations compteur", "voir_nbr_activation"),
    ("combien d'activations ce mois", "voir_nbr_activation"),
    ("nombre d'activations effectuées", "voir_nbr_activation"),
    ("total des activations de forfaits", "voir_nbr_activation"),
    ("combien d'options j'ai activées", "voir_nbr_activation"),
    ("fréquence de mes activations", "voir_nbr_activation"),
    ("nombre d'activations par mois", "voir_nbr_activation"),
    ("statistiques d'activations", "voir_nbr_activation"),

    # voir_cout_activation
    ("coût des activations", "voir_cout_activation"),
    ("prix des activations", "voir_cout_activation"),
    ("combien m'ont coûté les activations", "voir_cout_activation"),
    ("montant des activations", "voir_cout_activation"),
    ("facture des activations", "voir_cout_activation"),
    ("total dépensé en activations", "voir_cout_activation"),
    ("ce que j'ai payé pour activer", "voir_cout_activation"),
    ("prix de mes activations", "voir_cout_activation"),
    ("coût des options activées", "voir_cout_activation"),
    ("montant total des activations", "voir_cout_activation"),
    ("frais d'activation", "voir_cout_activation"),
    ("combien coûtent mes activations", "voir_cout_activation"),
    ("tarif des activations", "voir_cout_activation"),
    ("dépenses en activations", "voir_cout_activation"),
    ("coût de mes souscriptions", "voir_cout_activation"),

    # ============================================================
    # voir_services_actives — CORRIGÉ v7.0 (était F1:0.42)
    # DÉFINITION : ce que tu as ACTIVÉ via USSD (actions passées)
    # RÈGLE : NE PAS utiliser "forfait", "abonnement", "pack" ici
    #         → ces mots vont dans forfaits_actives
    # ============================================================
    ("quels services j'ai activés", "voir_services_actives"),
    ("services activés sur ma ligne", "voir_services_actives"),
    ("qu'est-ce que j'ai activé", "voir_services_actives"),
    ("liste des services que j'ai activés", "voir_services_actives"),
    ("services activés récemment", "voir_services_actives"),
    ("quelles options j'ai activées", "voir_services_actives"),
    ("services que j'ai activés ce mois", "voir_services_actives"),
    ("détail des services activés", "voir_services_actives"),
    ("voir les services activés", "voir_services_actives"),
    ("services activés via USSD", "voir_services_actives"),
    ("afficher mes services activés", "voir_services_actives"),
    ("quels services sont actifs sur mon compte", "voir_services_actives"),
    ("mes services téléphoniques activés", "voir_services_actives"),
    ("quelles options sont activées sur ma ligne", "voir_services_actives"),
    ("options que j'ai activées", "voir_services_actives"),
    ("services ajoutés à mon compte", "voir_services_actives"),
    ("voir ce que j'ai activé comme service", "voir_services_actives"),
    ("liste des activations de services", "voir_services_actives"),
    ("mes souscriptions de services", "voir_services_actives"),
    ("services en cours d'utilisation", "voir_services_actives"),
    ("quels services sont en cours", "voir_services_actives"),
    ("services actifs en ce moment", "voir_services_actives"),
    ("options actives sur ma ligne", "voir_services_actives"),
    ("mes services actifs actuellement", "voir_services_actives"),
    ("détail de mes souscriptions", "voir_services_actives"),

    # ============================================================
    # forfaits_actives — CORRIGÉ v7.0 (était F1:0.55)
    # DÉFINITION : tes FORFAITS/ABONNEMENTS en cours (état abonnement)
    # RÈGLE : utiliser "forfait", "abonnement", "pack" + contexte d'état
    #         → NE PAS utiliser "activé" seul sans "forfait"
    # ============================================================
    ("mes forfaits actifs", "forfaits_actives"),
    ("quels forfaits j'ai en ce moment", "forfaits_actives"),
    ("liste de mes forfaits actuels", "forfaits_actives"),
    ("forfaits en cours", "forfaits_actives"),
    ("quels sont mes forfaits en cours", "forfaits_actives"),
    ("mes abonnements actifs", "forfaits_actives"),
    ("mes abonnements en cours", "forfaits_actives"),
    ("packs souscrits actifs", "forfaits_actives"),
    ("mes forfaits internet et appels", "forfaits_actives"),
    ("détail de mes forfaits actuels", "forfaits_actives"),
    ("forfaits souscrits actifs", "forfaits_actives"),
    ("qu'est-ce que j'ai comme forfait actuellement", "forfaits_actives"),
    ("mes packs téléphoniques", "forfaits_actives"),
    ("forfaits disponibles sur mon compte", "forfaits_actives"),
    ("mes forfaits data et voix", "forfaits_actives"),
    ("abonnements en vigueur", "forfaits_actives"),
    ("mes forfaits télécom actifs", "forfaits_actives"),
    ("quels forfaits sont actifs", "forfaits_actives"),
    ("forfaits mensuels actifs", "forfaits_actives"),
    ("mon abonnement actuel", "forfaits_actives"),
    ("mes offres d'abonnement", "forfaits_actives"),
    ("quels packages j'ai", "forfaits_actives"),
    ("packages actifs sur mon compte", "forfaits_actives"),
    ("mes forfaits illimités", "forfaits_actives"),
    ("forfaits actifs sur ma ligne", "forfaits_actives"),

    # voir_code_ussd
    ("code USSD", "voir_code_ussd"),
    ("quels codes j'ai utilisés", "voir_code_ussd"),
    ("mes codes USSD", "voir_code_ussd"),
    ("codes que j'ai composés", "voir_code_ussd"),
    ("numéros USSD", "voir_code_ussd"),
    ("liste des codes USSD", "voir_code_ussd"),
    ("codes d'activation", "voir_code_ussd"),
    ("USSD utilisés", "voir_code_ussd"),
    ("mes codes", "voir_code_ussd"),
    ("quels codes USSD j'ai", "voir_code_ussd"),
    ("codes pour activer", "voir_code_ussd"),
    ("les codes que j'utilise", "voir_code_ussd"),

    # voir_option_ussd
    ("option USSD", "voir_option_ussd"),
    ("détail des options USSD", "voir_option_ussd"),
    ("mes options USSD", "voir_option_ussd"),
    ("quelles options USSD j'ai", "voir_option_ussd"),
    ("liste des options USSD", "voir_option_ussd"),
    ("options USSD souscrites", "voir_option_ussd"),
    ("détail de mes options USSD", "voir_option_ussd"),
    ("options USSD disponibles", "voir_option_ussd"),
    ("mes choix USSD", "voir_option_ussd"),
    ("options USSD que j'ai", "voir_option_ussd"),
    ("quelles sont mes options USSD", "voir_option_ussd"),
    ("options activées via USSD", "voir_option_ussd"),

    # voir_mois_activation
    ("mois des activations", "voir_mois_activation"),
    ("quand j'ai activé", "voir_mois_activation"),
    ("date des activations", "voir_mois_activation"),
    ("période d'activation", "voir_mois_activation"),
    ("en quel mois j'ai activé", "voir_mois_activation"),
    ("calendrier des activations", "voir_mois_activation"),
    ("mes activations par mois", "voir_mois_activation"),
    ("répartition des activations", "voir_mois_activation"),
    ("quand j'ai souscrit", "voir_mois_activation"),
    ("mois de souscription", "voir_mois_activation"),

    # voir_offre_activation
    ("offre des activations", "voir_offre_activation"),
    ("pour quelle offre j'ai activé", "voir_offre_activation"),
    ("activations par offre", "voir_offre_activation"),
    ("offres activées", "voir_offre_activation"),
    ("quelles offres j'ai activées", "voir_offre_activation"),
    ("forfaits activés par offre", "voir_offre_activation"),
    ("liste des offres activées", "voir_offre_activation"),
    ("détail des offres activées", "voir_offre_activation"),
    ("mes offres souscrites", "voir_offre_activation"),
    ("quelles offres j'ai souscrit", "voir_offre_activation"),

    # ========================================================
    # INTERNET
    # ========================================================

    # voir_volume_internet
    ("volume internet", "voir_volume_internet"),
    ("quantité de data", "voir_volume_internet"),
    ("combien de Mo j'ai utilisé", "voir_volume_internet"),
    ("ma data consommée", "voir_volume_internet"),
    ("volume de données", "voir_volume_internet"),
    ("ma consommation internet", "voir_volume_internet"),
    ("combien de Go j'ai consommé", "voir_volume_internet"),
    ("data utilisée", "voir_volume_internet"),
    ("internet consommé", "voir_volume_internet"),
    ("combien de data", "voir_volume_internet"),
    ("volume data", "voir_volume_internet"),
    ("ma conso data", "voir_volume_internet"),
    ("quantité de Go", "voir_volume_internet"),
    ("méga consommés", "voir_volume_internet"),
    ("giga utilisés", "voir_volume_internet"),

    # voir_nbr_sessions
    ("nombre de sessions internet", "voir_nbr_sessions"),
    ("combien de fois je me suis connecté", "voir_nbr_sessions"),
    ("sessions data", "voir_nbr_sessions"),
    ("connexions internet", "voir_nbr_sessions"),
    ("nombre de connexions", "voir_nbr_sessions"),
    ("combien de sessions", "voir_nbr_sessions"),
    ("fréquence de connexion", "voir_nbr_sessions"),
    ("fois où j'ai surfé", "voir_nbr_sessions"),
    ("sessions de navigation", "voir_nbr_sessions"),
    ("nombre de fois connecté", "voir_nbr_sessions"),
    ("total sessions", "voir_nbr_sessions"),
    ("compteur de connexions", "voir_nbr_sessions"),
    ("à quelle fréquence je me connecte", "voir_nbr_sessions"),
    ("mes sessions internet", "voir_nbr_sessions"),
    ("combien de connexions internet", "voir_nbr_sessions"),

    # voir_cout_internet
    ("coût internet", "voir_cout_internet"),
    ("prix de ma conso internet", "voir_cout_internet"),
    ("facture internet", "voir_cout_internet"),
    ("montant internet", "voir_cout_internet"),
    ("combien m'a coûté internet", "voir_cout_internet"),
    ("dépenses internet", "voir_cout_internet"),
    ("ce que j'ai payé pour internet", "voir_cout_internet"),
    ("tarif internet", "voir_cout_internet"),
    ("coût de ma data", "voir_cout_internet"),
    ("prix de ma data", "voir_cout_internet"),
    ("facture data", "voir_cout_internet"),
    ("montant data", "voir_cout_internet"),
    ("combien pour internet", "voir_cout_internet"),
    ("coût de ma connexion", "voir_cout_internet"),
    ("dépenses data", "voir_cout_internet"),

    # voir_type_trafic
    ("type de trafic internet", "voir_type_trafic"),
    ("nature de mon trafic", "voir_type_trafic"),
    ("type de données", "voir_type_trafic"),
    ("quel type de trafic internet", "voir_type_trafic"),
    ("trafic data", "voir_type_trafic"),
    ("nature de ma conso internet", "voir_type_trafic"),
    ("streaming ou navigation", "voir_type_trafic"),
    ("type de connexion internet", "voir_type_trafic"),
    ("données consommées type", "voir_type_trafic"),
    ("catégorie de trafic internet", "voir_type_trafic"),
    ("trafic 4G ou 3G", "voir_type_trafic"),
    ("nature des données internet", "voir_type_trafic"),

    # voir_taxation_internet
    ("taxation internet", "voir_taxation_internet"),
    ("comment est taxé mon internet", "voir_taxation_internet"),
    ("frais internet", "voir_taxation_internet"),
    ("taxe sur internet", "voir_taxation_internet"),
    ("tarification internet", "voir_taxation_internet"),
    ("frais de data", "voir_taxation_internet"),
    ("taxe sur data", "voir_taxation_internet"),
    ("comment est facturé internet", "voir_taxation_internet"),
    ("règles de taxation internet", "voir_taxation_internet"),
    ("taxation de ma data", "voir_taxation_internet"),
    ("frais supplémentaires internet", "voir_taxation_internet"),
    ("combien de taxes internet", "voir_taxation_internet"),

    # voir_reseau_internet
    ("réseau internet", "voir_reseau_internet"),
    ("sur quel réseau j'ai surfé", "voir_reseau_internet"),
    ("opérateur réseau internet", "voir_reseau_internet"),
    ("4G ou 3G", "voir_reseau_internet"),
    ("type de réseau internet", "voir_reseau_internet"),
    ("quel réseau internet", "voir_reseau_internet"),
    ("réseau utilisé pour internet", "voir_reseau_internet"),
    ("connexion via quel réseau", "voir_reseau_internet"),
    ("technologie réseau", "voir_reseau_internet"),
    ("bande utilisée", "voir_reseau_internet"),
    ("réseau mobile internet", "voir_reseau_internet"),
    ("couverture réseau", "voir_reseau_internet"),

    # voir_heure_connexion
    ("heure de connexion", "voir_heure_connexion"),
    ("à quelle heure je surfe", "voir_heure_connexion"),
    ("moment de connexion", "voir_heure_connexion"),
    ("période de connexion", "voir_heure_connexion"),
    ("connexions de nuit", "voir_heure_connexion"),
    ("heures de pointe", "voir_heure_connexion"),
    ("quand je me connecte", "voir_heure_connexion"),
    ("horaires de connexion", "voir_heure_connexion"),
    ("connexions jour/nuit", "voir_heure_connexion"),
    ("répartition horaire", "voir_heure_connexion"),

    # voir_duree_session
    ("durée des sessions", "voir_duree_session"),
    ("temps de connexion", "voir_duree_session"),
    ("combien de temps je surfe", "voir_duree_session"),
    ("durée moyenne session", "voir_duree_session"),
    ("temps passé en ligne", "voir_duree_session"),
    ("durée de navigation", "voir_duree_session"),
    ("combien de temps connecté", "voir_duree_session"),
    ("temps total en ligne", "voir_duree_session"),
    ("durée des connexions", "voir_duree_session"),
    ("temps internet", "voir_duree_session"),

    # voir_conso_quotidienne
    ("consommation quotidienne", "voir_conso_quotidienne"),
    ("data par jour", "voir_conso_quotidienne"),
    ("combien par jour", "voir_conso_quotidienne"),
    ("moyenne journalière", "voir_conso_quotidienne"),
    ("conso jour par jour", "voir_conso_quotidienne"),
    ("data utilisée chaque jour", "voir_conso_quotidienne"),
    ("répartition quotidienne", "voir_conso_quotidienne"),
    ("combien de Go par jour", "voir_conso_quotidienne"),
    ("consommation par jour", "voir_conso_quotidienne"),
    ("daily data usage", "voir_conso_quotidienne"),

    # ========================================================
    # PROFIL CLIENT
    # ========================================================

    # voir_offre_commerciale
    ("mon offre", "voir_offre_commerciale"),
    ("quel est mon forfait", "voir_offre_commerciale"),
    ("mon abonnement", "voir_offre_commerciale"),
    ("mon pack", "voir_offre_commerciale"),
    ("quel forfait j'ai", "voir_offre_commerciale"),
    ("décris mon offre", "voir_offre_commerciale"),
    ("caractéristiques de mon offre", "voir_offre_commerciale"),
    ("mon offre actuelle", "voir_offre_commerciale"),
    ("quel est mon plan", "voir_offre_commerciale"),
    ("mon offre télécom", "voir_offre_commerciale"),
    ("détail de mon forfait", "voir_offre_commerciale"),
    ("que contient mon offre", "voir_offre_commerciale"),
    ("mon offre mobile", "voir_offre_commerciale"),
    ("mon abonnement téléphonique", "voir_offre_commerciale"),
    ("mon pack actuel", "voir_offre_commerciale"),

    # consulter_offre
    ("détail de mon offre", "consulter_offre"),
    ("consulter mon offre", "consulter_offre"),
    ("voir le détail de mon offre", "consulter_offre"),
    ("afficher mon offre complète", "consulter_offre"),
    ("informations sur mon offre", "consulter_offre"),
    ("détails de mon abonnement", "consulter_offre"),
    ("que comprend mon offre", "consulter_offre"),
    ("afficher mon offre", "consulter_offre"),
    ("description de mon offre", "consulter_offre"),
    ("caractéristiques détaillées", "consulter_offre"),
    ("mon offre en détail", "consulter_offre"),
    ("voir les détails", "consulter_offre"),
    ("consulter mon abonnement", "consulter_offre"),
    ("détails du forfait", "consulter_offre"),
    ("infos sur mon pack", "consulter_offre"),

    # voir_segment_client
    ("mon segment", "voir_segment_client"),
    ("quelle catégorie de client", "voir_segment_client"),
    ("suis-je résidentiel ou entreprise", "voir_segment_client"),
    ("type de client", "voir_segment_client"),
    ("ma catégorie", "voir_segment_client"),
    ("quel type de client je suis", "voir_segment_client"),
    ("segment client", "voir_segment_client"),
    ("catégorie client", "voir_segment_client"),
    ("suis-je un client pro", "voir_segment_client"),
    ("client particulier ou pro", "voir_segment_client"),
    ("ma classification", "voir_segment_client"),
    ("dans quelle catégorie", "voir_segment_client"),

    # voir_date_activation
    ("date d'activation", "voir_date_activation"),
    ("depuis quand je suis client", "voir_date_activation"),
    ("ma date de souscription", "voir_date_activation"),
    ("quand ai-je activé ma ligne", "voir_date_activation"),
    ("date de début de contrat", "voir_date_activation"),
    ("depuis combien de temps", "voir_date_activation"),
    ("quand j'ai souscrit", "voir_date_activation"),
    ("date d'ouverture", "voir_date_activation"),
    ("activation de la ligne", "voir_date_activation"),
    ("depuis quelle date", "voir_date_activation"),
    ("mon ancienneté", "voir_date_activation"),
    ("date d'activation de ma ligne", "voir_date_activation"),

    # voir_statut_client
    ("mon statut", "voir_statut_client"),
    ("mon compte est-il actif", "voir_statut_client"),
    ("statut de ma ligne", "voir_statut_client"),
    ("ma ligne est-elle active", "voir_statut_client"),
    ("état de mon compte", "voir_statut_client"),
    ("mon compte est-il suspendu", "voir_statut_client"),
    ("ma ligne est-elle résiliée", "voir_statut_client"),
    ("suis-je toujours client", "voir_statut_client"),
    ("statut du compte", "voir_statut_client"),
    ("activation de mon compte", "voir_statut_client"),
    ("ma ligne fonctionne-t-elle", "voir_statut_client"),
    ("compte actif ou inactif", "voir_statut_client"),

    # info_client
    ("mes informations", "info_client"),
    ("mon profil", "info_client"),
    ("toutes mes infos", "info_client"),
    ("afficher mon profil", "info_client"),
    ("informations sur mon compte", "info_client"),
    ("mes données personnelles", "info_client"),
    ("voir mon profil", "info_client"),
    ("mes infos client", "info_client"),
    ("détails de mon compte", "info_client"),
    ("mon dossier client", "info_client"),
    ("qui suis-je comme client", "info_client"),
    ("profil client", "info_client"),

    # voir_offre_commerciale_detail
    ("détail commercial", "voir_offre_commerciale_detail"),
    ("description commerciale", "voir_offre_commerciale_detail"),
    ("mon offre en détail commercial", "voir_offre_commerciale_detail"),
    ("caractéristiques commerciales", "voir_offre_commerciale_detail"),
    ("offre détaillée", "voir_offre_commerciale_detail"),
    ("détails vente", "voir_offre_commerciale_detail"),
    ("infos commerciales", "voir_offre_commerciale_detail"),
    ("mon pack description", "voir_offre_commerciale_detail"),
    ("offre description", "voir_offre_commerciale_detail"),
    ("détail de vente", "voir_offre_commerciale_detail"),

    # voir_date_resiliation
    ("date de résiliation", "voir_date_resiliation"),
    ("quand ma ligne sera résiliée", "voir_date_resiliation"),
    ("fin de contrat", "voir_date_resiliation"),
    ("date de fin", "voir_date_resiliation"),
    ("quand mon abonnement se termine", "voir_date_resiliation"),
    ("échéance contrat", "voir_date_resiliation"),
    ("fin d'abonnement", "voir_date_resiliation"),
    ("date de clôture", "voir_date_resiliation"),
    ("quand mon offre se termine", "voir_date_resiliation"),
    ("date de fin de service", "voir_date_resiliation"),
    ("quand est-ce que ça se termine", "voir_date_resiliation"),
    ("jusqu'à quand", "voir_date_resiliation"),

    # ========================================================
    # RECHARGES
    # ========================================================

    # voir_nbr_recharge
    ("nombre de recharges", "voir_nbr_recharge"),
    ("combien de fois j'ai rechargé", "voir_nbr_recharge"),
    ("total des recharges", "voir_nbr_recharge"),
    ("compteur de recharges", "voir_nbr_recharge"),
    ("nombre de rechargements", "voir_nbr_recharge"),
    ("combien de recharges", "voir_nbr_recharge"),
    ("fréquence de recharge", "voir_nbr_recharge"),
    ("combien de fois par mois", "voir_nbr_recharge"),
    ("total recharges effectuées", "voir_nbr_recharge"),
    ("nombre de fois rechargé", "voir_nbr_recharge"),
    ("mes recharges compteur", "voir_nbr_recharge"),
    ("combien de recharges ce mois", "voir_nbr_recharge"),
    ("recharges compteur", "voir_nbr_recharge"),
    ("nombre d'opérations recharge", "voir_nbr_recharge"),
    ("fréquence des recharges", "voir_nbr_recharge"),
    ("combien de rechargements", "voir_nbr_recharge"),
    ("total des opérations", "voir_nbr_recharge"),
    ("nombre de transactions", "voir_nbr_recharge"),
    ("compteur de rechargements", "voir_nbr_recharge"),
    ("combien de fois j'ai mis du crédit", "voir_nbr_recharge"),

    # voir_montant_recharge
    ("montant des recharges", "voir_montant_recharge"),
    ("combien d'argent j'ai rechargé", "voir_montant_recharge"),
    ("total rechargé en dinars", "voir_montant_recharge"),
    ("somme des recharges", "voir_montant_recharge"),
    ("valeur des recharges", "voir_montant_recharge"),
    ("montant total rechargé", "voir_montant_recharge"),
    ("combien d'argent j'ai mis", "voir_montant_recharge"),
    ("total en dinars", "voir_montant_recharge"),
    ("somme totale", "voir_montant_recharge"),
    ("argent rechargé", "voir_montant_recharge"),
    ("crédit total", "voir_montant_recharge"),
    ("montant des rechargements", "voir_montant_recharge"),
    ("combien j'ai dépensé en recharges", "voir_montant_recharge"),
    ("total des montants", "voir_montant_recharge"),
    ("valeur totale", "voir_montant_recharge"),
    ("combien de dinars rechargés", "voir_montant_recharge"),
    ("somme d'argent", "voir_montant_recharge"),
    ("capital rechargé", "voir_montant_recharge"),
    ("montant cumulé", "voir_montant_recharge"),
    ("total financier", "voir_montant_recharge"),

    # voir_bonus_recharge
    ("bonus des recharges", "voir_bonus_recharge"),
    ("combien de bonus j'ai eu", "voir_bonus_recharge"),
    ("mes avantages recharge", "voir_bonus_recharge"),
    ("montant du bonus", "voir_bonus_recharge"),
    ("total des bonus", "voir_bonus_recharge"),
    ("bonus obtenus", "voir_bonus_recharge"),
    ("avantages reçus", "voir_bonus_recharge"),
    ("crédit bonus", "voir_bonus_recharge"),
    ("combien de bonus", "voir_bonus_recharge"),
    ("bonus cumulés", "voir_bonus_recharge"),
    ("points bonus", "voir_bonus_recharge"),
    ("récompenses recharge", "voir_bonus_recharge"),
    ("gains recharge", "voir_bonus_recharge"),
    ("bonus en dinars", "voir_bonus_recharge"),
    ("avantages en argent", "voir_bonus_recharge"),
    ("bonus total", "voir_bonus_recharge"),
    ("combien de cadeaux", "voir_bonus_recharge"),
    ("bonus reçus", "voir_bonus_recharge"),
    ("montant des bonus", "voir_bonus_recharge"),
    ("total avantages", "voir_bonus_recharge"),

    # voir_type_recharge
    ("type de recharge", "voir_type_recharge"),
    ("comment j'ai rechargé", "voir_type_recharge"),
    ("moyen de recharge", "voir_type_recharge"),
    ("méthode de recharge", "voir_type_recharge"),
    ("recharge par carte ou cash", "voir_type_recharge"),
    ("comment j'ai payé", "voir_type_recharge"),
    ("mode de recharge", "voir_type_recharge"),
    ("par quel moyen", "voir_type_recharge"),
    ("type de paiement", "voir_type_recharge"),
    ("recharge en ligne ou physique", "voir_type_recharge"),
    ("méthode utilisée", "voir_type_recharge"),
    ("comment j'ai mis du crédit", "voir_type_recharge"),
    ("moyen de paiement", "voir_type_recharge"),
    ("recharge par carte", "voir_type_recharge"),
    ("recharge par injection", "voir_type_recharge"),

    # voir_recharges
    ("mes recharges", "voir_recharges"),
    ("historique recharges", "voir_recharges"),
    ("afficher mes recharges", "voir_recharges"),
    ("toutes mes recharges", "voir_recharges"),
    ("liste des recharges", "voir_recharges"),
    ("détail des recharges", "voir_recharges"),
    ("récapitulatif recharges", "voir_recharges"),
    ("mes opérations recharge", "voir_recharges"),
    ("historique complet recharges", "voir_recharges"),
    ("tous mes rechargements", "voir_recharges"),
    ("mes transactions recharge", "voir_recharges"),
    ("recharges effectuées", "voir_recharges"),
    ("liste de mes recharges", "voir_recharges"),
    ("détail de mes recharges", "voir_recharges"),
    ("mes ajouts de crédit", "voir_recharges"),

    # voir_recharge_moyenne
    ("recharge moyenne", "voir_recharge_moyenne"),
    ("montant moyen par recharge", "voir_recharge_moyenne"),
    ("combien en moyenne", "voir_recharge_moyenne"),
    ("moyenne des recharges", "voir_recharge_moyenne"),
    ("valeur moyenne", "voir_recharge_moyenne"),
    ("recharge typique", "voir_recharge_moyenne"),
    ("moyenne par opération", "voir_recharge_moyenne"),
    ("montant habituel", "voir_recharge_moyenne"),
    ("combien je recharge d'habitude", "voir_recharge_moyenne"),
    ("moyenne de mes recharges", "voir_recharge_moyenne"),

    # voir_plus_grosse_recharge
    ("plus grosse recharge", "voir_plus_grosse_recharge"),
    ("recharge maximum", "voir_plus_grosse_recharge"),
    ("montant le plus élevé", "voir_plus_grosse_recharge"),
    ("ma plus grande recharge", "voir_plus_grosse_recharge"),
    ("record de recharge", "voir_plus_grosse_recharge"),
    ("maximum rechargé", "voir_plus_grosse_recharge"),
    ("recharge la plus importante", "voir_plus_grosse_recharge"),
    ("top recharge", "voir_plus_grosse_recharge"),
    ("montant max", "voir_plus_grosse_recharge"),
    ("quand j'ai le plus rechargé", "voir_plus_grosse_recharge"),

    # voir_derniere_recharge
    ("dernière recharge", "voir_derniere_recharge"),
    ("dernier rechargement", "voir_derniere_recharge"),
    ("ma dernière recharge", "voir_derniere_recharge"),
    ("recharge récente", "voir_derniere_recharge"),
    ("dernière opération", "voir_derniere_recharge"),
    ("quand j'ai rechargé la dernière fois", "voir_derniere_recharge"),
    ("dernier ajout", "voir_derniere_recharge"),
    ("recharge la plus récente", "voir_derniere_recharge"),
    ("date dernière recharge", "voir_derniere_recharge"),
    ("montant dernière recharge", "voir_derniere_recharge"),

    # ========================================================
    # APPELS
    # ========================================================

    # voir_nbr_appel
    ("nombre d'appels", "voir_nbr_appel"),
    ("combien d'appels j'ai passé", "voir_nbr_appel"),
    ("total appels", "voir_nbr_appel"),
    ("combien d'appels émis", "voir_nbr_appel"),
    ("nombre de communications", "voir_nbr_appel"),
    ("combien de fois j'ai téléphoné", "voir_nbr_appel"),
    ("total des appels", "voir_nbr_appel"),
    ("combien d'appels ce mois", "voir_nbr_appel"),
    ("nombre d'appels effectués", "voir_nbr_appel"),
    ("fréquence des appels", "voir_nbr_appel"),
    ("combien d'appels par jour", "voir_nbr_appel"),
    ("total communications", "voir_nbr_appel"),
    ("nombre d'appels passés", "voir_nbr_appel"),
    ("combien d'appels reçus", "voir_nbr_appel"),
    ("appels émis", "voir_nbr_appel"),
    ("compteur d'appels", "voir_nbr_appel"),
    ("nombre d'appels sortants", "voir_nbr_appel"),
    ("nombre d'appels entrants", "voir_nbr_appel"),
    ("total des communications", "voir_nbr_appel"),
    ("combien d'appels j'ai", "voir_nbr_appel"),

    # voir_duree_appel
    ("durée des appels", "voir_duree_appel"),
    ("temps total en communication", "voir_duree_appel"),
    ("combien de minutes j'ai parlé", "voir_duree_appel"),
    ("temps d'appel", "voir_duree_appel"),
    ("durée de mes communications", "voir_duree_appel"),
    ("temps passé au téléphone", "voir_duree_appel"),
    ("combien de temps j'ai téléphoné", "voir_duree_appel"),
    ("durée totale", "voir_duree_appel"),
    ("minutes de communication", "voir_duree_appel"),
    ("temps total d'appel", "voir_duree_appel"),
    ("combien d'heures au téléphone", "voir_duree_appel"),
    ("durée moyenne des appels", "voir_duree_appel"),
    ("temps de conversation", "voir_duree_appel"),
    ("durée des communications", "voir_duree_appel"),
    ("combien de minutes", "voir_duree_appel"),
    ("total minutes", "voir_duree_appel"),
    ("temps cumulé", "voir_duree_appel"),
    ("durée cumulée", "voir_duree_appel"),
    ("temps passé en appel", "voir_duree_appel"),
    ("durée totale des appels", "voir_duree_appel"),

    # voir_cout_appel
    ("coût des appels", "voir_cout_appel"),
    ("prix de mes appels", "voir_cout_appel"),
    ("facture appels", "voir_cout_appel"),
    ("montant des appels", "voir_cout_appel"),
    ("ce que m'ont coûté mes appels", "voir_cout_appel"),
    ("combien j'ai payé pour les appels", "voir_cout_appel"),
    ("dépenses appels", "voir_cout_appel"),
    ("tarif des appels", "voir_cout_appel"),
    ("coût des communications", "voir_cout_appel"),
    ("prix des communications", "voir_cout_appel"),
    ("facture de mes appels", "voir_cout_appel"),
    ("montant à payer pour les appels", "voir_cout_appel"),
    ("combien coûtent mes appels", "voir_cout_appel"),
    ("dépenses téléphoniques", "voir_cout_appel"),
    ("coût total des appels", "voir_cout_appel"),
    ("prix total appels", "voir_cout_appel"),
    ("montant total appels", "voir_cout_appel"),
    ("ce que je paye en appels", "voir_cout_appel"),
    ("facturation appels", "voir_cout_appel"),
    ("frais téléphoniques", "voir_cout_appel"),

    # voir_type_trafic_appel
    ("type de trafic appel", "voir_type_trafic_appel"),
    ("nature de mes communications", "voir_type_trafic_appel"),
    ("type d'appel", "voir_type_trafic_appel"),
    ("voix ou sms", "voir_type_trafic_appel"),
    ("appels ou messages", "voir_type_trafic_appel"),
    ("type de communication", "voir_type_trafic_appel"),
    ("nature du trafic appel", "voir_type_trafic_appel"),
    ("quel type d'appels", "voir_type_trafic_appel"),
    ("appels voix ou données", "voir_type_trafic_appel"),
    ("trafic voix", "voir_type_trafic_appel"),
    ("trafic sms", "voir_type_trafic_appel"),
    ("type de service appel", "voir_type_trafic_appel"),
    ("catégorie d'appel", "voir_type_trafic_appel"),
    ("nature des appels", "voir_type_trafic_appel"),
    ("type de trafic utilisé appel", "voir_type_trafic_appel"),

    # voir_taxation_appel
    ("taxation appels", "voir_taxation_appel"),
    ("comment sont taxés mes appels", "voir_taxation_appel"),
    ("frais d'appel", "voir_taxation_appel"),
    ("tarification appels", "voir_taxation_appel"),
    ("taxe sur les appels", "voir_taxation_appel"),
    ("combien de taxes appels", "voir_taxation_appel"),
    ("frais de communication", "voir_taxation_appel"),
    ("taxation des communications", "voir_taxation_appel"),
    ("règles de taxation appels", "voir_taxation_appel"),
    ("comment sont facturés mes appels", "voir_taxation_appel"),
    ("tarif de taxation appels", "voir_taxation_appel"),
    ("coût selon taxation", "voir_taxation_appel"),
    ("frais supplémentaires appels", "voir_taxation_appel"),
    ("taxes appliquées appels", "voir_taxation_appel"),
    ("mode de taxation appels", "voir_taxation_appel"),

    # voir_reseau_appel
    ("réseau d'appel", "voir_reseau_appel"),
    ("vers quel réseau j'appelle", "voir_reseau_appel"),
    ("opérateur appelé", "voir_reseau_appel"),
    ("destination réseau appel", "voir_reseau_appel"),
    ("appels vers quel opérateur", "voir_reseau_appel"),
    ("réseau destinataire", "voir_reseau_appel"),
    ("appels vers TT", "voir_reseau_appel"),
    ("appels vers Ooredoo", "voir_reseau_appel"),
    ("appels vers Orange", "voir_reseau_appel"),
    ("vers quel réseau je téléphone", "voir_reseau_appel"),
    ("opérateur de destination", "voir_reseau_appel"),
    ("réseau appelé", "voir_reseau_appel"),
    ("appels fixes ou mobiles", "voir_reseau_appel"),
    ("réseau fixe appels", "voir_reseau_appel"),
    ("réseau mobile appels", "voir_reseau_appel"),

    # voir_destination_appel
    ("destination appels", "voir_destination_appel"),
    ("qui j'appelle le plus", "voir_destination_appel"),
    ("appels vers fixes ou mobiles", "voir_destination_appel"),
    ("vers qui j'appelle", "voir_destination_appel"),
    ("type de destination", "voir_destination_appel"),
    ("appels vers quel type", "voir_destination_appel"),
    ("mes correspondants", "voir_destination_appel"),
    ("qui sont mes appels", "voir_destination_appel"),
    ("destinations fréquentes", "voir_destination_appel"),
    ("appels vers particuliers", "voir_destination_appel"),
    ("appels vers professionnels", "voir_destination_appel"),
    ("numéros appelés", "voir_destination_appel"),
    ("répartition des appels", "voir_destination_appel"),
    ("appels nationaux", "voir_destination_appel"),
    ("appels internationaux", "voir_destination_appel"),

    # historique_appels
    # IMPORTANT : PAS de "appels effectués" ici → doublon avec voir_appels_sortants
    ("mes appels", "historique_appels"),
    ("historique appels", "historique_appels"),
    ("mes communications", "historique_appels"),
    ("tous mes appels", "historique_appels"),
    ("liste des appels", "historique_appels"),
    ("récapitulatif appels", "historique_appels"),
    ("mes conversations téléphoniques", "historique_appels"),
    ("historique des appels", "historique_appels"),
    ("mes appels passés et reçus", "historique_appels"),
    ("historique complet appels", "historique_appels"),
    ("mes communications téléphoniques", "historique_appels"),
    ("détail de mes appels", "historique_appels"),
    ("résumé de mes appels", "historique_appels"),
    ("bilan de mes appels", "historique_appels"),
    ("journal des appels", "historique_appels"),

    # ============================================================
    # voir_appels_sortants — CORRIGÉ v7.0 (était F1:0.67)
    # DOUBLÉ : 10 → 20 exemples
    # SUPPRIMÉ "appels effectués" (doublon avec historique_appels)
    # RÈGLE : toujours préciser ÉMIS / PASSÉS / SORTANTS
    # ============================================================
    ("appels sortants", "voir_appels_sortants"),
    ("appels que j'ai passés", "voir_appels_sortants"),
    ("mes appels émis", "voir_appels_sortants"),
    ("appels émis uniquement", "voir_appels_sortants"),
    ("communications sortantes", "voir_appels_sortants"),
    ("appels vers l'extérieur", "voir_appels_sortants"),
    ("mes appels sortants seulement", "voir_appels_sortants"),
    ("liste appels sortants", "voir_appels_sortants"),
    ("détail appels sortants", "voir_appels_sortants"),
    ("appels que j'ai initiés", "voir_appels_sortants"),
    ("j'ai appelé combien de fois", "voir_appels_sortants"),
    ("appels sortants ce mois", "voir_appels_sortants"),
    ("nombre d'appels que j'ai passés", "voir_appels_sortants"),
    ("mes appels vers d'autres", "voir_appels_sortants"),
    ("appels émis ce mois", "voir_appels_sortants"),
    ("total appels sortants", "voir_appels_sortants"),
    ("coût appels sortants", "voir_appels_sortants"),
    ("durée appels sortants", "voir_appels_sortants"),
    ("appels sortants uniquement", "voir_appels_sortants"),
    ("uniquement les appels que j'ai passés", "voir_appels_sortants"),

    # voir_appels_entrants
    ("appels entrants", "voir_appels_entrants"),
    ("appels que j'ai reçus", "voir_appels_entrants"),
    ("mes appels reçus", "voir_appels_entrants"),
    ("nombre d'appels reçus", "voir_appels_entrants"),
    ("communications entrantes", "voir_appels_entrants"),
    ("appels que j'ai eus", "voir_appels_entrants"),
    ("liste appels entrants", "voir_appels_entrants"),
    ("détail appels entrants", "voir_appels_entrants"),
    ("qui m'a appelé", "voir_appels_entrants"),
    ("appels reçus ce mois", "voir_appels_entrants"),

    # voir_sms
    ("mes sms", "voir_sms"),
    ("nombre de sms", "voir_sms"),
    ("sms envoyés", "voir_sms"),
    ("sms reçus", "voir_sms"),
    ("messages texte", "voir_sms"),
    ("combien de sms", "voir_sms"),
    ("historique sms", "voir_sms"),
    ("liste des sms", "voir_sms"),
    ("mes messages", "voir_sms"),
    ("textos", "voir_sms"),

    # voir_appels_longue_duree
    ("appels longs", "voir_appels_longue_duree"),
    ("plus longs appels", "voir_appels_longue_duree"),
    ("appels de plus de 10 minutes", "voir_appels_longue_duree"),
    ("communications longues", "voir_appels_longue_duree"),
    ("durée maximale", "voir_appels_longue_duree"),
    ("appels les plus longs", "voir_appels_longue_duree"),
    ("record de durée", "voir_appels_longue_duree"),
    ("appels exceptionnels", "voir_appels_longue_duree"),
    ("grandes conversations", "voir_appels_longue_duree"),
    ("appels interminables", "voir_appels_longue_duree"),

    # voir_appels_courte_duree
    ("appels courts", "voir_appels_courte_duree"),
    ("appels de moins d'une minute", "voir_appels_courte_duree"),
    ("brefs appels", "voir_appels_courte_duree"),
    ("communications courtes", "voir_appels_courte_duree"),
    ("durée minimale", "voir_appels_courte_duree"),
    ("appels les plus courts", "voir_appels_courte_duree"),
    ("appels éclairs", "voir_appels_courte_duree"),
    ("micro-appels", "voir_appels_courte_duree"),
    ("appels rapides", "voir_appels_courte_duree"),
    ("courtes conversations", "voir_appels_courte_duree"),

    # ========================================================
    # COÛTS
    # ========================================================

    # cout_total
    ("mon coût total", "cout_total"),
    ("combien j'ai dépensé", "cout_total"),
    ("total de mes dépenses", "cout_total"),
    ("ma facture totale", "cout_total"),
    ("tout ce que j'ai payé", "cout_total"),
    ("dépenses globales", "cout_total"),
    ("combien j'ai payé en tout", "cout_total"),
    ("somme de mes dépenses", "cout_total"),
    ("total facture", "cout_total"),
    ("montant total", "cout_total"),
    ("addition globale", "cout_total"),
    ("ce que je dois", "cout_total"),
    ("total à payer", "cout_total"),
    ("ma consommation en argent", "cout_total"),
    ("récapitulatif financier", "cout_total"),

    # cout_total_mois
    ("coût du mois", "cout_total_mois"),
    ("dépenses ce mois", "cout_total_mois"),
    ("facture du mois", "cout_total_mois"),
    ("combien ce mois-ci", "cout_total_mois"),
    ("total mensuel", "cout_total_mois"),
    ("dépenses mensuelles", "cout_total_mois"),
    ("coût de ce mois", "cout_total_mois"),
    ("facturation mensuelle", "cout_total_mois"),
    ("ce mois j'ai payé", "cout_total_mois"),
    ("montant du mois", "cout_total_mois"),
    ("facture de ce mois", "cout_total_mois"),
    ("dépenses de ce mois", "cout_total_mois"),

    # comparaison_cout
    ("comparaison avec mois dernier", "comparaison_cout"),
    ("évolution de mes dépenses", "comparaison_cout"),
    ("plus ou moins que le mois dernier", "comparaison_cout"),
    ("différence de facture", "comparaison_cout"),
    ("variation des coûts", "comparaison_cout"),
    ("augmentation ou baisse", "comparaison_cout"),
    ("comparer mes factures", "comparaison_cout"),
    ("mois dernier vs ce mois", "comparaison_cout"),
    ("tendance des dépenses", "comparaison_cout"),
    ("évolution mensuelle", "comparaison_cout"),
]


# ========================================================
# DATA AUGMENTATION
# ========================================================

def augmenter_donnees_ultime(donnees):
    prefixes = ["", "s'il vous plaît, ", "je voudrais ", "je veux ", "j'aimerais "]
    suffixes = ["", " s'il vous plaît", " merci", " svp"]

    transformations = [
        lambda t: t,
        lambda t: t.capitalize(),
        lambda t: t + "?",
        lambda t: "Est-ce que " + t.lower() + " ?",
        lambda t: "Peux-tu me donner " + t.lower(),
        lambda t: "Pouvez-vous me dire " + t.lower(),
    ]

    nouvelles = []
    for texte, intention in donnees:
        nouvelles.append((texte, intention))
        for prefix in prefixes[:2]:
            for suffix in suffixes[:2]:
                if prefix or suffix:
                    nouveau = prefix + texte + suffix
                    if nouveau != texte:
                        nouvelles.append((nouveau, intention))
        for trans in transformations[1:4]:
            try:
                nouveau = trans(texte)
                if nouveau != texte:
                    nouvelles.append((nouveau, intention))
            except Exception:
                pass

    uniques = []
    seen = set()
    for texte, intention in nouvelles:
        key = f"{texte.lower()}|{intention}"
        if key not in seen:
            seen.add(key)
            uniques.append((texte, intention))

    print(f"✅ Data augmentation ULTIME: {len(donnees)} → {len(uniques)} exemples")
    return uniques


DONNEES = augmenter_donnees_ultime(DONNEES)


# ========================================================
# FONCTIONS UTILITAIRES
# ========================================================

def compter_intentions():
    compteurs = {}
    for _, intention in DONNEES:
        compteurs[intention] = compteurs.get(intention, 0) + 1
    return compteurs


def afficher_resume_intentions():
    compteurs = compter_intentions()
    print("=" * 70)
    print("  RÉSUMÉ DES INTENTIONS")
    print("=" * 70)
    print(f"📊 {len(compteurs)} intentions différentes")
    print(f"📝 {len(DONNEES)} exemples au total")
    print("-" * 70)

    categories = {
        "Générales":   ["saluer", "au_revoir", "remercier", "affirmer", "nier", "donner_id_client"],
        "Activations": [i for i in compteurs if "activation" in i or i in ["forfaits_actives", "voir_services_actives", "voir_code_ussd", "voir_option_ussd"]],
        "Internet":    [i for i in compteurs if "internet" in i or "session" in i or "conso" in i or (i == "voir_type_trafic")],
        "Profil":      [i for i in compteurs if "offre" in i or "segment" in i or "statut" in i or "date_" in i or i == "info_client" or i == "consulter_offre"],
        "Recharges":   [i for i in compteurs if "recharge" in i],
        "Appels":      [i for i in compteurs if "appel" in i or i in ["voir_sms", "historique_appels"]],
        "Coûts":       [i for i in compteurs if "cout" in i or "comparaison" in i],
    }

    for cat, intentions in categories.items():
        intentions_valides = [i for i in intentions if i in compteurs]
        if intentions_valides:
            total_cat = sum(compteurs[i] for i in intentions_valides)
            print(f"\n📁 {cat}: {total_cat} exemples")
            for i in sorted(intentions_valides):
                print(f"   {i:<35} : {compteurs[i]:3d} exemples")
    print("=" * 70)


def diviser_donnees(donnees, train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)
    donnees_melangees = donnees.copy()
    random.shuffle(donnees_melangees)
    textes = [t for t, _ in donnees_melangees]
    labels = [l for _, l in donnees_melangees]
    try:
        X_train, X_temp, y_train, y_temp = train_test_split(
            textes, labels, test_size=(val_ratio + test_ratio),
            random_state=seed, stratify=labels
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size),
            random_state=seed, stratify=y_temp
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except ValueError:
        print("⚠️ Fallback: division sans stratification")
        X_train, X_temp, y_train, y_temp = train_test_split(
            textes, labels, test_size=(val_ratio + test_ratio), random_state=seed
        )
        val_size = val_ratio / (val_ratio + test_ratio)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1 - val_size), random_state=seed
        )
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def afficher_statistiques(train, val, test):
    X_train, y_train = train
    X_val, y_val = val
    X_test, y_test = test
    total = len(X_train) + len(X_val) + len(X_test)
    print("=" * 60)
    print("  STATISTIQUES DES DONNÉES")
    print("=" * 60)
    print(f"Total exemples  : {total}")
    print(f"Train           : {len(X_train)} ({len(X_train)/total*100:.0f}%)")
    print(f"Validation      : {len(X_val)} ({len(X_val)/total*100:.0f}%)")
    print(f"Test            : {len(X_test)} ({len(X_test)/total*100:.0f}%)")
    print(f"Intentions      : {len(set(y_train))}")
    print("=" * 60)


if __name__ == "__main__":
    afficher_resume_intentions()
    train, val, test = diviser_donnees(DONNEES)
    afficher_statistiques(train, val, test)