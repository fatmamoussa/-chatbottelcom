# Chatbot Tunisie Telecom - PFE Data Science

## 🎯 Problématique
Assistance client automatisée pour Tunisie Telecom

## 🏗️ Architecture
- NLU: TF-IDF + SVM (from scratch)
- NER: Regex personnalisées
- Dialogue: Gestionnaire à états
- API: Flask REST
- Frontend: HTML/CSS/JS

## 📊 Performances
- Accuracy: 90.34%
- Intentions: 55
- Validation croisée: 91.90% ±0.79%

## 🚀 Déploiement
```bash
python api.py
http://localhost:5005

### 2. **Graphiques pour la soutenance** (déjà faits)
Les 7 graphiques que nous avons créés sont parfaits !

### 3. **Préparer les réponses aux questions**

| Question du jury | Votre réponse |
|-----------------|---------------|
| "Pourquoi from scratch ?" | "Pour maîtriser chaque brique et démontrer ma compréhension" |
| "Pourquoi SVM ?" | "Meilleur compromis performance/données, validé par Joachims (1998)" |
| "Limites ?" | "Plus de données NLU amélioreraient les intentions faibles" |
| "Passage à l'échelle ?" | "L'API Flask tient 100+ requêtes/sec, optimisable avec Redis" |

---

## 🏆 **CE QUE LE JURY VA RETENIR**

### ✅ **Points forts qui impressionnent**