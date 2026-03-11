# ============================================================
# utils.py - Utilitaires
# Projet : Chatbot Tunisie Telecom - From Scratch
# Version : 1.0
# ============================================================

import json
import time
from datetime import datetime
from pathlib import Path


class Logger:
    """Logger pour tracer les conversations"""
    
    def __init__(self, log_file="conversations.log"):
        self.log_file = log_file
    
    def log(self, message, reponse, duree):
        """Enregistre une conversation"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "reponse": reponse.get("reponse"),
            "intention": reponse.get("intention"),
            "confiance": reponse.get("confiance"),
            "duree_ms": round(duree * 1000, 2)
        }
        
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


class Timer:
    """Chronomètre simple"""
    
    def __init__(self):
        self.debut = None
    
    def start(self):
        self.debut = time.time()
        return self
    
    def stop(self):
        if self.debut:
            duree = time.time() - self.debut
            self.debut = None
            return duree
        return 0


def formater_duree(secondes):
    """Formate une durée"""
    if secondes < 60:
        return f"{secondes:.0f}s"
    elif secondes < 3600:
        minutes = secondes // 60
        return f"{minutes:.0f}m {secondes%60:.0f}s"
    else:
        heures = secondes // 3600
        return f"{heures:.0f}h {(secondes%3600)//60:.0f}m"