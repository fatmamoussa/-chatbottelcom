# ============================================================
# actions.py - Actions pour chaque champ
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 5.0 - TOUTES les intentions couvertes
# ============================================================

import pandas as pd
import os
from config import config

# ========================================================
# CHARGEMENT DES DONNÉES
# ========================================================

def charger_csv(nom_fichier):
    chemin = os.path.join(config.CHEMIN_DATASET, nom_fichier)
    try:
        df = pd.read_csv(chemin, sep="|", low_memory=False)
        print(f"✅ {nom_fichier}: {len(df)} lignes")
        return df
    except Exception as e:
        print(f"⚠️ Erreur {nom_fichier}: {e}")
        return pd.DataFrame()


print("\n" + "="*60)
print("  CHARGEMENT DES DONNÉES")
print("="*60)

DF_PARC       = charger_csv(config.DATASET.FICHIERS["parc"])
DF_ACTIVATION = charger_csv(config.DATASET.FICHIERS["activation"])
DF_DATA       = charger_csv(config.DATASET.FICHIERS["data"])
DF_REFIL      = charger_csv(config.DATASET.FICHIERS["refil"])
DF_TRAFIC     = charger_csv(config.DATASET.FICHIERS["trafic"])

print("="*60)


# ========================================================
# FONCTIONS UTILITAIRES
# ========================================================

def client_existe(cc):
    if DF_PARC.empty:
        return False
    return cc in DF_PARC["CONTRAT_CLIENT"].values


def filtrer_par_mois(df, mois, colonne_mois="MOIS_YEAR"):
    if df.empty or not mois:
        return df
    if isinstance(mois, str) and "/" in mois:
        return df[df[colonne_mois].astype(str).str.contains(mois[:7])]
    return df


def formater_reponse(titre, donnees):
    lignes = [f"📌 {titre}"]
    for key, value in donnees.items():
        lignes.append(f"   • {key}: {value}")
    return "\n".join(lignes)


def _get_cc(entites, slots):
    """Raccourci pour récupérer le numéro client"""
    return entites.get("contrat_client") or slots.get("contrat_client")


# ========================================================
# ACTIONS POUR data_activation.csv
# ========================================================

def action_voir_nbr_activation(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune activation pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    total = int(df["NBR_ACTIVATION"].sum())
    slots["contrat_client"] = cc
    return f"📊 Nombre d'activations pour {cc} : **{total}**", slots


def action_voir_cout_activation(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune activation pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    total = df["COUT_TTC"].sum()
    slots["contrat_client"] = cc
    return f"💰 Coût total des activations pour {cc} : **{total:.3f} DT**", slots


def action_voir_services_actives(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun service activé pour {cc}.", slots
    services = df["USSD_SERVICE"].dropna().unique().tolist()
    if not services: return f"Aucun service nommé pour {cc}.", slots
    slots["contrat_client"] = cc
    liste = "\n".join([f"   • {s}" for s in services[:10]])
    return f"📱 Services activés pour {cc} :\n{liste}", slots


def action_voir_code_ussd(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune activation pour {cc}.", slots
    codes = df["CODE_USSD_SERVICE"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"📞 Codes USSD utilisés : {', '.join(map(str, codes[:5]))}", slots


def action_voir_option_ussd(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune activation pour {cc}.", slots
    options = df["USSD_SERVICE_OPTION"].dropna().unique().tolist()
    if not options: return f"Aucune option USSD pour {cc}.", slots
    slots["contrat_client"] = cc
    return f"⚙️ Options USSD : {', '.join(options[:5])}", slots


def action_voir_mois_activation(entites, slots):
    """Mois où des activations ont eu lieu"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune activation pour {cc}.", slots
    mois_list = df["MOIS_YEAR"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"📅 Mois d'activation pour {cc} : {', '.join(map(str, sorted(mois_list)[:6]))}", slots


def action_voir_offre_activation(entites, slots):
    """Offres pour lesquelles des activations ont été faites"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune activation pour {cc}.", slots
    col = "OFFRE_COMMERCIAL" if "OFFRE_COMMERCIAL" in df.columns else None
    if not col:
        return action_voir_services_actives(entites, slots)
    offres = df[col].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"📦 Offres activées pour {cc} : {', '.join(offres[:5])}", slots


# ========================================================
# ACTIONS POUR data_data.csv (Internet)
# ========================================================

def action_voir_volume_internet(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    volume = df["VOLUME"].sum()
    volume_aff = f"{volume/1024:.2f} Go" if volume > 1024 else f"{volume:.2f} Mo"
    slots["contrat_client"] = cc
    return f"🌐 Volume internet consommé pour {cc} : **{volume_aff}**", slots


def action_voir_nbr_sessions(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    sessions = int(df["NBRE"].sum())
    slots["contrat_client"] = cc
    return f"📱 Nombre de sessions internet pour {cc} : **{sessions}**", slots


def action_voir_cout_internet(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    cout = df["COUT_TOTAL"].sum()
    slots["contrat_client"] = cc
    return f"💰 Coût internet pour {cc} : **{cout:.3f} DT**", slots


def action_voir_type_trafic(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    types = df["TYPE_TRAFIC"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"📊 Types de trafic internet : {', '.join(map(str, types))}", slots


def action_voir_taxation_internet(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    taxations = df["TAXATION"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"💵 Taxation internet : {', '.join(taxations)}", slots


def action_voir_reseau_internet(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    reseaux = df["RESEAU_APPEL"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"📶 Réseaux utilisés pour internet : {', '.join(reseaux)}", slots


def action_voir_heure_connexion(entites, slots):
    """Heures de connexion internet (si colonne disponible)"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    col_heure = next((c for c in df.columns if "HEURE" in c.upper() or "HOUR" in c.upper()), None)
    slots["contrat_client"] = cc
    if col_heure:
        heures = df[col_heure].dropna().value_counts().head(3)
        heures_str = ", ".join([f"{h}h ({v} sessions)" for h, v in heures.items()])
        return f"🕐 Heures de connexion les plus fréquentes : {heures_str}", slots
    # Fallback : utiliser les sessions
    sessions = int(df["NBRE"].sum())
    return f"🕐 Vous avez effectué {sessions} sessions internet au total.", slots


def action_voir_duree_session(entites, slots):
    """Durée des sessions internet"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    col_duree = next((c for c in df.columns if "DUREE" in c.upper() or "DURATION" in c.upper()), None)
    slots["contrat_client"] = cc
    if col_duree:
        duree_total = df[col_duree].sum()
        heures = int(duree_total // 60)
        minutes = int(duree_total % 60)
        duree_aff = f"{heures}h {minutes}min" if heures > 0 else f"{minutes}min"
        return f"⏱️ Durée totale de vos sessions internet : **{duree_aff}**", slots
    # Fallback : volume / sessions
    volume = df["VOLUME"].sum()
    sessions = int(df["NBRE"].sum())
    volume_aff = f"{volume/1024:.2f} Go" if volume > 1024 else f"{volume:.2f} Mo"
    return f"📊 Vous avez consommé {volume_aff} en {sessions} sessions internet.", slots


def action_voir_conso_quotidienne(entites, slots):
    """Consommation internet moyenne par jour"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune donnée internet pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    slots["contrat_client"] = cc
    volume_total = df["VOLUME"].sum()
    # Estimer le nombre de jours distincts
    col_date = next((c for c in df.columns if "DATE" in c.upper() or "JOUR" in c.upper()), None)
    if col_date:
        nb_jours = df[col_date].nunique()
        nb_jours = max(nb_jours, 1)
    else:
        nb_jours = 30  # Estimation mensuelle
    conso_jour = volume_total / nb_jours
    conso_aff = f"{conso_jour/1024:.2f} Go" if conso_jour > 1024 else f"{conso_jour:.0f} Mo"
    return f"📅 Consommation internet moyenne par jour : **{conso_aff}/jour**", slots


# ========================================================
# ACTIONS POUR data_parc.csv (Profil)
# ========================================================

def action_voir_offre_commerciale(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    if DF_PARC.empty: return "Données client non disponibles.", slots
    df = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Client {cc} non trouvé.", slots
    slots["contrat_client"] = cc
    offre = df.iloc[0]["OFFRE_COMMERCIAL"]
    return f"📦 Votre offre commerciale : **{offre}**", slots


def action_voir_offre_commerciale_detail(entites, slots):
    """Détail complet de l'offre commerciale"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    if DF_PARC.empty: return "Données client non disponibles.", slots
    df = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Client {cc} non trouvé.", slots
    slots["contrat_client"] = cc
    row = df.iloc[0]
    donnees = {
        "Offre": row.get("OFFRE_COMMERCIAL", "N/A"),
        "Description": row.get("DESC_OFFRE", row.get("OFFRE_COMMERCIAL", "N/A")),
        "Segment": row.get("DESC_SEGMENT_CLIENT", "N/A"),
        "Date activation": row.get("DATE_ACTIVATION", "N/A"),
        "Statut": "Actif" if row.get("STATUT") == "A" else "Inactif",
    }
    return formater_reponse(f"Détail offre {cc}", donnees), slots


def action_voir_segment_client(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Client {cc} non trouvé.", slots
    segment = df.iloc[0]["DESC_SEGMENT_CLIENT"]
    slots["contrat_client"] = cc
    return f"👥 Votre segment client : **{segment}**", slots


def action_voir_date_activation(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    if DF_PARC.empty: return "Données client non disponibles.", slots
    df = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Client {cc} non trouvé.", slots
    slots["contrat_client"] = cc
    date = df.iloc[0]["DATE_ACTIVATION"]
    return f"📅 Date d'activation de votre ligne : **{date}**", slots


def action_voir_date_resiliation(entites, slots):
    """Date de résiliation ou fin de contrat"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    if DF_PARC.empty: return "Données client non disponibles.", slots
    df = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Client {cc} non trouvé.", slots
    slots["contrat_client"] = cc
    row = df.iloc[0]
    # Chercher une colonne de résiliation
    col_resil = next((c for c in df.columns if "RESIL" in c.upper() or "FIN" in c.upper() or "END" in c.upper()), None)
    statut = row.get("STATUT", "")
    if statut == "D" and col_resil:
        date_resil = row[col_resil]
        return f"📅 Date de résiliation : **{date_resil}**", slots
    elif statut == "D":
        return f"📅 Votre ligne est résiliée. Contactez le service client pour la date exacte.", slots
    elif statut == "A":
        return f"✅ Votre ligne est active — aucune résiliation prévue.", slots
    else:
        return f"ℹ️ Statut actuel : **{statut}**. Contactez le service client pour plus d'informations.", slots


def action_voir_statut_client(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    if DF_PARC.empty: return "Données client non disponibles.", slots
    df = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Client {cc} non trouvé.", slots
    slots["contrat_client"] = cc
    statut = df.iloc[0]["STATUT"]
    statut_map = {"A": "✅ Actif", "I": "⏸️ Inactif", "D": "❌ Résilié"}
    statut_texte = statut_map.get(statut, statut)
    return f"⚡ Statut de votre ligne : **{statut_texte}**", slots


# ========================================================
# ACTIONS POUR data_refil.csv (Recharges)
# ========================================================

def action_voir_nbr_recharge(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_REFIL[DF_REFIL["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune recharge pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    total = int(df["NBR_RECHARGE"].sum())
    slots["contrat_client"] = cc
    return f"🔢 Nombre de recharges pour {cc} : **{total}**", slots


def action_voir_montant_recharge(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_REFIL[DF_REFIL["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune recharge pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    total = df["MONTANT_RECHARGE"].sum()
    slots["contrat_client"] = cc
    return f"💰 Montant total rechargé pour {cc} : **{total:.3f} DT**", slots


def action_voir_bonus_recharge(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_REFIL[DF_REFIL["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune recharge pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    total = df["MONTANT_BONUS_RECHARGE"].sum()
    slots["contrat_client"] = cc
    return f"🎁 Bonus total des recharges pour {cc} : **{total:.3f} DT**", slots


def action_voir_type_recharge(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_REFIL[DF_REFIL["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune recharge pour {cc}.", slots
    types = df["TYPE_RECHARGE"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"💳 Types de recharge utilisés : {', '.join(types)}", slots


def action_voir_recharge_moyenne(entites, slots):
    """Montant moyen par recharge"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_REFIL[DF_REFIL["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune recharge pour {cc}.", slots
    slots["contrat_client"] = cc
    montant_total = df["MONTANT_RECHARGE"].sum()
    nb_recharges = int(df["NBR_RECHARGE"].sum())
    if nb_recharges == 0:
        return f"Aucune recharge enregistrée pour {cc}.", slots
    moyenne = montant_total / nb_recharges
    return f"📊 Montant moyen par recharge pour {cc} : **{moyenne:.3f} DT**", slots


def action_voir_plus_grosse_recharge(entites, slots):
    """Recharge la plus importante"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_REFIL[DF_REFIL["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune recharge pour {cc}.", slots
    slots["contrat_client"] = cc
    max_recharge = df["MONTANT_RECHARGE"].max()
    # Trouver le mois correspondant
    row_max = df.loc[df["MONTANT_RECHARGE"].idxmax()]
    mois_max = row_max.get("MOIS_YEAR", "N/A")
    return f"🏆 Plus grosse recharge pour {cc} : **{max_recharge:.3f} DT** (en {mois_max})", slots


def action_voir_derniere_recharge(entites, slots):
    """Dernière recharge effectuée"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_REFIL[DF_REFIL["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune recharge pour {cc}.", slots
    slots["contrat_client"] = cc
    # Trier par mois pour avoir la plus récente
    col_date = "MOIS_YEAR"
    if col_date in df.columns:
        df_sorted = df.sort_values(col_date, ascending=False)
        derniere = df_sorted.iloc[0]
        montant = derniere["MONTANT_RECHARGE"]
        mois = derniere[col_date]
        return f"🕐 Dernière recharge pour {cc} : **{montant:.3f} DT** en {mois}", slots
    # Fallback
    montant_total = df["MONTANT_RECHARGE"].sum()
    return f"💰 Total des recharges pour {cc} : **{montant_total:.3f} DT**", slots


def action_voir_recharges(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_REFIL[DF_REFIL["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucune recharge pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    slots["contrat_client"] = cc
    donnees = {
        "Nombre": int(df["NBR_RECHARGE"].sum()),
        "Montant": f"{df['MONTANT_RECHARGE'].sum():.3f} DT",
        "Bonus": f"{df['MONTANT_BONUS_RECHARGE'].sum():.3f} DT",
        "Période": mois if mois else "Total"
    }
    return formater_reponse(f"Recharges {cc}", donnees), slots


# ========================================================
# ACTIONS POUR data_trafic.csv (Appels)
# ========================================================

def action_voir_nbr_appel(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    total = int(df["NBR_APPEL"].sum())
    slots["contrat_client"] = cc
    return f"🔢 Nombre d'appels pour {cc} : **{total}**", slots


def action_voir_duree_appel(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    duree = df["DUREE_APPEL"].sum()
    heures = int(duree // 60)
    minutes = int(duree % 60)
    duree_aff = f"{heures}h {minutes}min" if heures > 0 else f"{minutes}min"
    slots["contrat_client"] = cc
    return f"⏱️ Durée totale d'appels pour {cc} : **{duree_aff}**", slots


def action_voir_cout_appel(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    cout = df["COUT_TTC"].sum()
    slots["contrat_client"] = cc
    return f"💰 Coût total des appels pour {cc} : **{cout:.3f} DT**", slots


def action_voir_type_trafic_appel(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    types = df["CODE_TYPE_TRAFIC"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"📞 Types de trafic d'appels : {', '.join(map(str, types))}", slots


def action_voir_taxation_appel(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    taxations = df["TAXATION"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"💵 Taxation des appels : {', '.join(taxations)}", slots


def action_voir_reseau_appel(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    reseaux = df["RESEAU_APPEL"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"📡 Réseaux appelés : {', '.join(reseaux)}", slots


def action_voir_destination_appel(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    destinations = df["DES_DESTINATION_TRAFIC"].dropna().unique().tolist()
    slots["contrat_client"] = cc
    return f"🎯 Destinations des appels : {', '.join(destinations[:5])}", slots


def action_voir_appels_sortants(entites, slots):
    """Appels émis (sortants)"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    slots["contrat_client"] = cc
    # Filtrer les appels sortants
    col_sens = next((c for c in df.columns if "SENS" in c.upper() or "DIRECT" in c.upper() or "TYPE" in c.upper()), None)
    if col_sens:
        sortants = df[df[col_sens].astype(str).str.upper().isin(["S", "OUT", "SORTANT", "EMIS"])]
        if not sortants.empty:
            nb = int(sortants["NBR_APPEL"].sum())
            duree = sortants["DUREE_APPEL"].sum()
            cout = sortants["COUT_TTC"].sum()
            heures = int(duree // 60)
            minutes = int(duree % 60)
            duree_aff = f"{heures}h {minutes}min" if heures > 0 else f"{minutes}min"
            donnees = {"Nombre": nb, "Durée": duree_aff, "Coût": f"{cout:.3f} DT"}
            return formater_reponse(f"Appels sortants {cc}", donnees), slots
    # Fallback : total appels
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    nb = int(df["NBR_APPEL"].sum())
    duree = df["DUREE_APPEL"].sum()
    cout = df["COUT_TTC"].sum()
    heures = int(duree // 60)
    minutes = int(duree % 60)
    duree_aff = f"{heures}h {minutes}min" if heures > 0 else f"{minutes}min"
    donnees = {"Nombre total d'appels": nb, "Durée totale": duree_aff, "Coût total": f"{cout:.3f} DT"}
    return formater_reponse(f"Appels {cc}", donnees), slots


def action_voir_appels_entrants(entites, slots):
    """Appels reçus (entrants)"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    slots["contrat_client"] = cc
    col_sens = next((c for c in df.columns if "SENS" in c.upper() or "DIRECT" in c.upper()), None)
    if col_sens:
        entrants = df[df[col_sens].astype(str).str.upper().isin(["E", "IN", "ENTRANT", "RECU"])]
        if not entrants.empty:
            nb = int(entrants["NBR_APPEL"].sum())
            duree = entrants["DUREE_APPEL"].sum()
            heures = int(duree // 60)
            minutes = int(duree % 60)
            duree_aff = f"{heures}h {minutes}min" if heures > 0 else f"{minutes}min"
            donnees = {"Nombre": nb, "Durée": duree_aff}
            return formater_reponse(f"Appels entrants {cc}", donnees), slots
    # Fallback
    nb = int(df["NBR_APPEL"].sum())
    return f"📞 Total des appels enregistrés pour {cc} : **{nb}** (données détaillées entrants/sortants non disponibles)", slots


def action_voir_sms(entites, slots):
    """SMS envoyés/reçus"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    slots["contrat_client"] = cc
    # Chercher dans trafic avec filtre SMS
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if not df.empty:
        col_type = next((c for c in df.columns if "TYPE" in c.upper() and "TRAFIC" in c.upper()), None)
        if col_type:
            sms_df = df[df[col_type].astype(str).str.upper().isin(["SMS", "2", "MMS"])]
            if not sms_df.empty:
                nb = int(sms_df["NBR_APPEL"].sum())
                cout = sms_df["COUT_TTC"].sum()
                donnees = {"Nombre de SMS": nb, "Coût": f"{cout:.3f} DT"}
                return formater_reponse(f"SMS {cc}", donnees), slots
    # Chercher dans activation
    df_act = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if not df_act.empty:
        sms_act = df_act[df_act["USSD_SERVICE"].astype(str).str.upper().str.contains("SMS", na=False)]
        if not sms_act.empty:
            nb = int(sms_act["NBR_ACTIVATION"].sum())
            return f"📩 Activations SMS pour {cc} : **{nb}**", slots
    return f"📩 Aucune donnée SMS disponible pour {cc}. Vérifiez votre forfait.", slots


def action_voir_appels_longue_duree(entites, slots):
    """Appels de longue durée (> 10 minutes)"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    slots["contrat_client"] = cc
    # Appels > 10 minutes
    col_duree = "DUREE_APPEL"
    if col_duree in df.columns:
        longs = df[df[col_duree] > 10]
        nb = int(longs["NBR_APPEL"].sum()) if not longs.empty else 0
        duree_max = df[col_duree].max()
        donnees = {
            "Appels > 10 min": nb,
            "Durée maximale": f"{int(duree_max)} min",
            "Total appels": int(df["NBR_APPEL"].sum())
        }
        return formater_reponse(f"Appels longs {cc}", donnees), slots
    return f"📞 Données de durée non disponibles pour {cc}.", slots


def action_voir_appels_courte_duree(entites, slots):
    """Appels de courte durée (< 1 minute)"""
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    slots["contrat_client"] = cc
    col_duree = "DUREE_APPEL"
    if col_duree in df.columns:
        courts = df[df[col_duree] < 1]
        nb = int(courts["NBR_APPEL"].sum()) if not courts.empty else 0
        total = int(df["NBR_APPEL"].sum())
        pct = (nb / total * 100) if total > 0 else 0
        donnees = {
            "Appels < 1 min": nb,
            "Pourcentage": f"{pct:.1f}%",
            "Total appels": total
        }
        return formater_reponse(f"Appels courts {cc}", donnees), slots
    return f"📞 Données de durée non disponibles pour {cc}.", slots


def action_historique_appels(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun appel pour {cc}.", slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if mois: df = filtrer_par_mois(df, mois)
    slots["contrat_client"] = cc
    duree = df["DUREE_APPEL"].sum()
    heures = int(duree // 60)
    minutes = int(duree % 60)
    duree_aff = f"{heures}h {minutes}min" if heures > 0 else f"{minutes}min"
    donnees = {
        "Nombre d'appels": int(df["NBR_APPEL"].sum()),
        "Durée totale": duree_aff,
        "Coût total": f"{df['COUT_TTC'].sum():.3f} DT",
        "Période": mois if mois else "Total"
    }
    return formater_reponse(f"Historique appels {cc}", donnees), slots


# ========================================================
# ACTIONS GÉNÉRALES
# ========================================================

def action_consulter_offre(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Client {cc} non trouvé.", slots
    row = df.iloc[0]
    slots["contrat_client"] = cc
    statut_map = {"A": "Actif", "I": "Inactif", "D": "Résilié"}
    donnees = {
        "Offre": row.get("OFFRE_COMMERCIAL", "N/A"),
        "Segment": row.get("DESC_SEGMENT_CLIENT", "N/A"),
        "Statut": statut_map.get(row.get("STATUT", ""), row.get("STATUT", "N/A")),
        "Date activation": row.get("DATE_ACTIVATION", "N/A")
    }
    return formater_reponse(f"Offre {cc}", donnees), slots


def action_info_client(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_PARC[DF_PARC["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Client {cc} non trouvé.", slots
    row = df.iloc[0]
    slots["contrat_client"] = cc
    statut_map = {"A": "Actif ✅", "I": "Inactif ⏸️", "D": "Résilié ❌"}
    donnees = {
        "Offre": row.get("OFFRE_COMMERCIAL", "N/A"),
        "Segment": row.get("DESC_SEGMENT_CLIENT", "N/A"),
        "Statut": statut_map.get(row.get("STATUT", ""), row.get("STATUT", "N/A")),
        "Date activation": row.get("DATE_ACTIVATION", "N/A")
    }
    return formater_reponse(f"Profil client {cc}", donnees), slots


def action_forfaits_actives(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
    if df.empty: return f"Aucun forfait actif pour {cc}.", slots
    slots["contrat_client"] = cc
    services = df["USSD_SERVICE"].dropna().unique().tolist()
    liste = "\n".join([f"   • {s}" for s in services[:10]])
    return f"📱 Forfaits actifs {cc}:\n{liste}", slots


def action_cout_total(entites, slots):
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    cout_data = cout_trafic = cout_activ = 0
    if not DF_DATA.empty:
        df = DF_DATA[DF_DATA["CONTRAT_CLIENT"] == cc]
        if mois: df = filtrer_par_mois(df, mois)
        if not df.empty: cout_data = df["COUT_TOTAL"].sum()
    if not DF_TRAFIC.empty:
        df = DF_TRAFIC[DF_TRAFIC["CONTRAT_CLIENT"] == cc]
        if mois: df = filtrer_par_mois(df, mois)
        if not df.empty: cout_trafic = df["COUT_TTC"].sum()
    if not DF_ACTIVATION.empty:
        df = DF_ACTIVATION[DF_ACTIVATION["CONTRAT_CLIENT"] == cc]
        if mois: df = filtrer_par_mois(df, mois)
        if not df.empty: cout_activ = df["COUT_TTC"].sum()
    total = cout_data + cout_trafic + cout_activ
    slots["contrat_client"] = cc
    donnees = {
        "Internet": f"{cout_data:.3f} DT",
        "Appels": f"{cout_trafic:.3f} DT",
        "Forfaits": f"{cout_activ:.3f} DT",
        "Total": f"{total:.3f} DT",
        "Période": mois if mois else "Toutes périodes"
    }
    return formater_reponse(f"Coût total {cc}", donnees), slots


def action_cout_total_mois(entites, slots):
    """Coût total du mois en cours ou d'un mois précis"""
    from datetime import datetime
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    # Utiliser le mois fourni ou le mois actuel
    mois = entites.get("mois_annee") or slots.get("mois_annee")
    if not mois:
        mois = datetime.now().strftime("%m/%Y")
        entites = dict(entites)
        entites["mois_annee"] = mois
    return action_cout_total(entites, slots)


def action_comparaison_cout(entites, slots):
    """Compare le coût du mois actuel vs mois précédent"""
    from datetime import datetime
    cc = _get_cc(entites, slots)
    if not cc: return config.DIALOGUE.DEMANDE_CC[0], slots
    slots["contrat_client"] = cc
    now = datetime.now()
    mois_actuel = now.strftime("%m/%Y")
    mois_precedent_dt = datetime(now.year if now.month > 1 else now.year - 1,
                                  now.month - 1 if now.month > 1 else 12, 1)
    mois_precedent = mois_precedent_dt.strftime("%m/%Y")

    def get_cout(mois):
        total = 0
        for df, col in [(DF_DATA, "COUT_TOTAL"), (DF_TRAFIC, "COUT_TTC"), (DF_ACTIVATION, "COUT_TTC")]:
            if df.empty: continue
            d = df[df["CONTRAT_CLIENT"] == cc]
            d = filtrer_par_mois(d, mois)
            if not d.empty: total += d[col].sum()
        return total

    cout_actuel = get_cout(mois_actuel)
    cout_prec = get_cout(mois_precedent)
    diff = cout_actuel - cout_prec
    tendance = "📈 Augmentation" if diff > 0 else "📉 Baisse" if diff < 0 else "➡️ Stable"

    donnees = {
        f"Mois actuel ({mois_actuel})": f"{cout_actuel:.3f} DT",
        f"Mois précédent ({mois_precedent})": f"{cout_prec:.3f} DT",
        "Différence": f"{diff:+.3f} DT",
        "Tendance": tendance
    }
    return formater_reponse(f"Comparaison coût {cc}", donnees), slots


# ========================================================
# MAP DES ACTIONS — COMPLET (55 intentions)
# ========================================================

ACTIONS_CSV = {
    # data_activation.csv
    "voir_nbr_activation":          action_voir_nbr_activation,
    "voir_cout_activation":         action_voir_cout_activation,
    "voir_services_actives":        action_voir_services_actives,
    "voir_code_ussd":               action_voir_code_ussd,
    "voir_option_ussd":             action_voir_option_ussd,
    "voir_mois_activation":         action_voir_mois_activation,
    "voir_offre_activation":        action_voir_offre_activation,

    # data_data.csv (Internet)
    "voir_volume_internet":         action_voir_volume_internet,
    "voir_nbr_sessions":            action_voir_nbr_sessions,
    "voir_cout_internet":           action_voir_cout_internet,
    "voir_type_trafic":             action_voir_type_trafic,
    "voir_taxation_internet":       action_voir_taxation_internet,
    "voir_reseau_internet":         action_voir_reseau_internet,
    "voir_heure_connexion":         action_voir_heure_connexion,
    "voir_duree_session":           action_voir_duree_session,
    "voir_conso_quotidienne":       action_voir_conso_quotidienne,

    # data_parc.csv (Profil)
    "voir_offre_commerciale":           action_voir_offre_commerciale,
    "voir_offre_commerciale_detail":    action_voir_offre_commerciale_detail,
    "voir_segment_client":              action_voir_segment_client,
    "voir_date_activation":             action_voir_date_activation,
    "voir_date_resiliation":            action_voir_date_resiliation,
    "voir_statut_client":               action_voir_statut_client,
    "consulter_offre":                  action_consulter_offre,
    "info_client":                      action_info_client,

    # data_refil.csv (Recharges)
    "voir_nbr_recharge":            action_voir_nbr_recharge,
    "voir_montant_recharge":        action_voir_montant_recharge,
    "voir_bonus_recharge":          action_voir_bonus_recharge,
    "voir_type_recharge":           action_voir_type_recharge,
    "voir_recharge_moyenne":        action_voir_recharge_moyenne,
    "voir_plus_grosse_recharge":    action_voir_plus_grosse_recharge,
    "voir_derniere_recharge":       action_voir_derniere_recharge,
    "voir_recharges":               action_voir_recharges,

    # data_trafic.csv (Appels)
    "voir_nbr_appel":               action_voir_nbr_appel,
    "voir_duree_appel":             action_voir_duree_appel,
    "voir_cout_appel":              action_voir_cout_appel,
    "voir_type_trafic_appel":       action_voir_type_trafic_appel,
    "voir_taxation_appel":          action_voir_taxation_appel,
    "voir_reseau_appel":            action_voir_reseau_appel,
    "voir_destination_appel":       action_voir_destination_appel,
    "voir_appels_sortants":         action_voir_appels_sortants,
    "voir_appels_entrants":         action_voir_appels_entrants,
    "voir_sms":                     action_voir_sms,
    "voir_appels_longue_duree":     action_voir_appels_longue_duree,
    "voir_appels_courte_duree":     action_voir_appels_courte_duree,
    "historique_appels":            action_historique_appels,

    # Générales
    "forfaits_actives":             action_forfaits_actives,
    "cout_total":                   action_cout_total,
    "cout_total_mois":              action_cout_total_mois,
    "comparaison_cout":             action_comparaison_cout,
}