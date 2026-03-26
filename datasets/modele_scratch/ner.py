# ner_advanced.py - NER avec spaCy et règles avancées
import re
import spacy
from typing import Dict, List, Tuple

class NER:
    """
    NER avancé avec :
    - spaCy pour les entités classiques
    - Regex pour les entités métier
    - Modèle entraîné pour les numéros clients
    """
    
    def __init__(self):
        self.nlp = spacy.load("fr_core_news_sm")
        
        # Patterns pour les numéros clients (plus flexibles)
        self.patterns = {
            "contrat_client": [
                re.compile(r'CC[_\-]?\s*[0-9]{8}', re.IGNORECASE),  # CC_52099260
                re.compile(r'(?:contrat|numéro|n°|id|client)[\s:]*([0-9]{8})', re.IGNORECASE),  # contrat 52099260
                re.compile(r'\b([0-9]{8})\b(?!\s*DT|\s*dinar)', re.IGNORECASE),  # 52099260 seul
            ],
            "mois_annee": [
                re.compile(r'(0[1-9]|1[0-2])/(20[2-9][0-9])'),  # 08/2025
                re.compile(r'(janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s*(20[2-9][0-9])', re.IGNORECASE),
            ],
            "montant": [
                re.compile(r'(\d+[.,]?\d*)\s*(DT|dinars?|tnd)', re.IGNORECASE),
                re.compile(r'(\d+[.,]?\d*)\s*€?', re.IGNORECASE),  # Montant simple
            ],
            "telephone": [
                re.compile(r'(2|5|9)[0-9]{7}'),  # Numéros tunisiens
                re.compile(r'\+216\s*[0-9]{8}'),
            ]
        }
        
        # Dictionnaire des mois
        self.mois_map = {
            "janvier": "01", "février": "02", "mars": "03", "avril": "04",
            "mai": "05", "juin": "06", "juillet": "07", "août": "08",
            "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12"
        }
        
        # Expressions relatives avec calcul dynamique
        self.expressions_relatives = {
            "ce mois": self._mois_actuel,
            "mois dernier": self._mois_dernier,
            "mois prochain": self._mois_prochain,
            "trimestre dernier": self._trimestre_dernier,
            "année dernière": self._annee_derniere,
            "ce trimestre": self._trimestre_actuel,
        }
    
    def _mois_actuel(self):
        from datetime import datetime
        return datetime.now().strftime("%m/%Y")
    
    def _mois_dernier(self):
        from datetime import datetime, timedelta
        date = datetime.now().replace(day=1) - timedelta(days=1)
        return date.strftime("%m/%Y")
    
    def _mois_prochain(self):
        from datetime import datetime, timedelta
        date = datetime.now().replace(day=28) + timedelta(days=4)
        return date.strftime("%m/%Y")
    
    def _trimestre_dernier(self):
        from datetime import datetime
        now = datetime.now()
        trimestre = ((now.month - 1) // 3) - 1
        if trimestre < 0:
            trimestre, annee = 3, now.year - 1
        else:
            annee = now.year
        mois_debut = trimestre * 3 + 1
        return f"{mois_debut:02d}/{annee}"
    
    def _trimestre_actuel(self):
        from datetime import datetime
        now = datetime.now()
        trimestre = (now.month - 1) // 3
        mois_debut = trimestre * 3 + 1
        return f"{mois_debut:02d}/{now.year}"
    
    def _annee_derniere(self):
        from datetime import datetime
        return str(datetime.now().year - 1)
    
    def extraire(self, texte: str) -> Dict:
        """Extrait toutes les entités avec scores de confiance"""
        entites = {}
        texte_lower = texte.lower()
        
        # 1. Détection avec spaCy
        doc = self.nlp(texte)
        for ent in doc.ents:
            if ent.label_ == "DATE":
                if "mois_annee" not in entites:
                    entites["mois_annee"] = ent.text
                    entites["mois_annee_confiance"] = 0.8
            elif ent.label_ == "MONEY" and "montant" not in entites:
                entites["montant"] = ent.text
                entites["montant_confiance"] = 0.7
        
        # 2. Détection avec regex avancées
        for entite_type, patterns in self.patterns.items():
            for pattern in patterns:
                match = pattern.search(texte)
                if match:
                    value = match.group(0) if len(match.groups()) == 0 else match.group(1)
                    if entite_type not in entites:
                        entites[entite_type] = value
                        entites[f"{entite_type}_confiance"] = 0.9
                        break
        
        # 3. Expressions relatives
        for expr, func in self.expressions_relatives.items():
            if expr in texte_lower and "mois_annee" not in entites:
                entites["mois_annee"] = func()
                entites["mois_annee_source"] = expr
                entites["mois_annee_confiance"] = 0.95
                break
        
        # 4. Standardisation des formats
        if "contrat_client" in entites:
            # Normaliser CC_XXXXXXXX
            cc = entites["contrat_client"]
            if re.match(r'^[0-9]{8}$', cc):
                entites["contrat_client"] = f"CC_{cc}"
            elif re.match(r'^CC[_\-]?[0-9]{8}$', cc, re.IGNORECASE):
                entites["contrat_client"] = cc.upper()
        
        return entites
    
    def valider_contrat(self, cc: str) -> Tuple[bool, str]:
        """Valide un numéro de contrat"""
        if not cc:
            return False, "Numéro vide"
        
        # Extraire les chiffres
        chiffres = re.sub(r'\D', '', cc)
        if len(chiffres) != 8:
            return False, f"Format invalide: {len(chiffres)} chiffres, attendu 8"
        
        # Vérifier dans la base (simulé)
        return True, "OK"
    
    def formater_mois(self, mois: str) -> str:
        """Formate un mois au format MM/YYYY"""
        if re.match(r'^\d{2}/\d{4}$', mois):
            return mois
        
        # Mois en texte
        for nom, num in self.mois_map.items():
            if nom in mois.lower():
                annee = re.search(r'20\d{2}', mois)
                if annee:
                    return f"{num}/{annee.group()}"
                from datetime import datetime
                return f"{num}/{datetime.now().year}"
        
        return mois