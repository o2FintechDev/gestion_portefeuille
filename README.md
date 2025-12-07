# Gestion de Portefeuille – Application d’Analyse Financière et d’Optimisation Markowitz

Application Streamlit complète permettant la gestion, l’analyse et l’optimisation de portefeuilles financiers.  
Le projet intègre un pipeline de données robuste, des modèles d’optimisation modernes, des visualisations interactives et une architecture modulaire professionnelle.

---

## 1. Objectifs du projet

- Construire une application opérationnelle d’aide à la décision pour investisseurs.  
- Automatiser la collecte, le nettoyage et la normalisation de données financières (actions, indices, taux).  
- Implémenter des modèles de performance et de risque : rendements, volatilités, matrices de corrélation.  
- Intégrer l’optimisation de portefeuilles :
  - Portefeuille à variance minimale (GMV)
  - Portefeuille de tangence
  - Frontière efficiente (Markowitz)
  - CML
- Offrir une interface utilisateur simple, dynamique et intuitive.
- Ajouter des modules avancés : taux sans risque, inflation, diagnostics de données, indicateurs techniques.

---

## 2. Fonctionnalités principales

### **2.1 Gestion des données**
- Récupération via **yfinance**, **Eurostat**, **ECB Data Portal**, **FRED**.  
- Cache intelligent pour limiter les appels API.  
- Rotation de proxies et headers pour réduire les blocages.  
- Diagnostics :
  - NaN
  - Alignement des index
  - Volume de données

### **2.2 Analyse financière**
- Rendements (linéaires/log).  
- Volatilités, covariance, corrélation.  
- Statistiques du portefeuille :
  - Rendement moyen
  - Risque
  - Ratio de Sharpe
  - Beta CAPM
  - Classification marché

### **2.3 Optimisation Markowitz**
- Contraintes configurables : short selling, limites de pondération.  
- Modèles disponibles :
  - GMV
  - Tangency portfolio
  - Frontière efficiente  
- Résolution robuste via `scipy.optimize`.  
- Gestion d’erreurs :
  - Matrices singulières
  - Solutions impossibles

### **2.4 Interface Streamlit**
- Ajout / suppression dynamique de tickers.  
- Slider pour contraintes de pondération.  
- Visualisations interactives :
  - Cours
  - Heatmap de corrélation
  - Frontière efficiente + CML
  - Répartition des poids  
- Messages conditionnels (minimum 2 actifs).

