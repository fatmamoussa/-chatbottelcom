# ============================================================
# ner.py - Reconnaissance d'entités
# Projet : Chatbot Tunisie Telecom — From Scratch
# Version : 3.0 — Gestion complète des mois / dates
# ============================================================

import re
from datetime import datetime
from config import config


class ModeleNER:
    """Reconnaissance d'entités par regex."""

    def __init__(self):
        self.patterns = {
            "contrat_client": re.compile(config.NER.PATTERN_CONTRAT),
            "mois_annee":     re.compile(config.NER.PATTERN_MOIS_ANNEE),
            "montant":        re.compile(config.NER.PATTERN_MONTANT, re.IGNORECASE),
        }

        # ── Mois en texte français ────────────────────────────
        self.mois_fr = {
            "janvier": "01", "fevrier": "02", "février": "02",
            "mars": "03", "avril": "04", "mai": "05", "juin": "06",
            "juillet": "07", "aout": "08", "août": "08",
            "septembre": "09", "octobre": "10",
            "novembre": "11", "decembre": "12", "décembre": "12",
        }

        # ── Expressions relatives → fonction qui retourne MM/YYYY ──
        self.expressions_relatives = {
            "ce mois":           self._mois_actuel,
            "ce mois-ci":        self._mois_actuel,
            "mois actuel":       self._mois_actuel,
            "mois courant":      self._mois_actuel,
            "mois en cours":     self._mois_actuel,
            "mois dernier":      self._mois_dernier,
            "mois précédent":    self._mois_dernier,
            "mois precedent":    self._mois_dernier,
            "mois passé":        self._mois_dernier,
            "mois passe":        self._mois_dernier,
            "le mois dernier":   self._mois_dernier,
            "le mois passé":     self._mois_dernier,
            "mois prochain":     self._mois_prochain,
            "mois suivant":      self._mois_prochain,
            "le mois prochain":  self._mois_prochain,
            "trimestre dernier": self._trimestre_dernier,
            "dernier trimestre": self._trimestre_dernier,
        }

        # Pattern regex mois texte
        mois_list = sorted(self.mois_fr.keys(), key=len, reverse=True)
        self.pattern_mois_texte = re.compile(
            r'\b(' + '|'.join(re.escape(m) for m in mois_list) + r')\b',
            re.IGNORECASE
        )

        # Pattern regex expressions relatives
        expr_list = sorted(self.expressions_relatives.keys(), key=len, reverse=True)
        self.pattern_relatif = re.compile(
            r'(' + '|'.join(re.escape(e) for e in expr_list) + r')',
            re.IGNORECASE
        )

        self.mois_map = config.NER.MOIS_MAP

    # ── Dates relatives ───────────────────────────────────────

    def _mois_actuel(self):
        return datetime.now().strftime("%m/%Y")

    def _mois_dernier(self):
        now = datetime.now()
        if now.month == 1:
            return f"12/{now.year - 1}"
        return f"{now.month - 1:02d}/{now.year}"

    def _mois_prochain(self):
        now = datetime.now()
        if now.month == 12:
            return f"01/{now.year + 1}"
        return f"{now.month + 1:02d}/{now.year}"

    def _trimestre_dernier(self):
        now = datetime.now()
        trimestre = (now.month - 1) // 3
        if trimestre == 0:
            trimestre, annee = 4, now.year - 1
        else:
            annee = now.year
        mois_debut = (trimestre - 1) * 3 + 1
        return f"{mois_debut:02d}/{annee}"

    # ── Extraction ────────────────────────────────────────────

    def extraire(self, texte):
        """Extrait toutes les entités d'un texte."""
        entites = {}
        texte_lower = texte.lower().strip()

        # 1. Numéro client CC_XXXXXXXX
        match = self.patterns["contrat_client"].search(texte)
        if match:
            entites["contrat_client"] = match.group(0).upper()

        # 2. Date format MM/YYYY
        match = self.patterns["mois_annee"].search(texte)
        if match:
            entites["mois_annee"] = match.group(0)

        # 3. Expressions relatives (mois dernier, ce mois…)
        if "mois_annee" not in entites:
            match = self.pattern_relatif.search(texte_lower)
            if match:
                expr = match.group(0).lower().strip()
                for cle, fn in self.expressions_relatives.items():
                    if cle.lower() == expr:
                        entites["mois_annee"] = fn()
                        entites["mois_texte"] = expr
                        break

        # 4. Mois en texte (janvier, février…)
        if "mois_annee" not in entites:
            match = self.pattern_mois_texte.search(texte_lower)
            if match:
                mois_texte = match.group(0).lower()
                entites["mois_texte"] = mois_texte
                mois_num = self.mois_fr.get(mois_texte)
                if mois_num:
                    annee_match = re.search(r'\b(20\d{2})\b', texte)
                    annee = annee_match.group(1) if annee_match else str(datetime.now().year)
                    entites["mois_annee"] = f"{mois_num}/{annee}"
                    entites["mois_numero"] = int(mois_num)

        # 5. Montant en DT
        match = self.patterns["montant"].search(texte)
        if match:
            try:
                entites["montant"] = float(match.group(1).replace(',', '.'))
            except Exception:
                pass

        return entites

    def valider_numero(self, cc):
        if not cc:
            return False
        return bool(re.match(r'^CC_[0-9]{8}$', str(cc)))

    def formater_mois(self, mois_annee):
        """'01/2024' → 'janvier 2024'"""
        if not mois_annee or '/' not in str(mois_annee):
            return str(mois_annee)
        mois_num, annee = str(mois_annee).split('/')
        noms = {
            "01": "janvier", "02": "février", "03": "mars",
            "04": "avril",   "05": "mai",     "06": "juin",
            "07": "juillet", "08": "août",    "09": "septembre",
            "10": "octobre", "11": "novembre","12": "décembre",
        }
        return f"{noms.get(mois_num, mois_num)} {annee}"


if __name__ == "__main__":
    ner = ModeleNER()
    tests = [
        "CC_52099260 du mois 11/2024",
        "mes recharges en janvier",
        "appels de novembre 2023",
        "mes appels du mois dernier",
        "ma consommation ce mois-ci",
        "recharges du mois précédent",
        "coût total ce mois",
        "dépenses du trimestre dernier",
        "combien j'ai rechargé le mois passé",
        "CC_12345678 recharges janvier 2024",
    ]
    print("=" * 55)
    for msg in tests:
        e = ner.extraire(msg)
        mois = f" → {ner.formater_mois(e['mois_annee'])}" if "mois_annee" in e else ""
        print(f"'{msg}'\n  {e}{mois}\n")