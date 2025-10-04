# 📌 Système de Recommandation Multi-modèles pour Amazon

## 📖 Description du projet

Ce projet propose un **système de recommandation multi-modèles** appliqué aux produits Amazon.  
L’objectif est d’aider les utilisateurs à découvrir des articles pertinents en combinant plusieurs techniques de recommandation et en intégrant une **analyse de sentiment des avis clients**.

Le système inclut :  

- **Filtrage par popularité** → recommandé pour les cas de **cold-start**.  
- **Filtrage basé sur le contenu (Content-Based)** → recommandations à partir des **descriptions produits**.  
- **Filtrage collaboratif (SVD)** → basé sur les interactions utilisateurs-produits.  
- **Approche hybride** → combinaison des modèles pour plus de robustesse.  
- **Analyse de sentiment (NLP)** des avis clients avec **Bernoulli Naïve Bayes** (70,43% de précision).  

Le projet est déployé sous forme d’une **application interactive Streamlit**, permettant aux utilisateurs de tester en temps réel les différents modèles de recommandation.

---

## 🏗️ Architecture du projet

1. **Collecte & Prétraitement des données**  
   - Données issues du dataset **Amazon Reviews 2023** (McAuley Lab).  
   - Utilisation de **PySpark** pour le traitement distribué (128 GB de données).  
   - Sauvegarde et gestion via **MongoDB**.  
   - Nettoyage, vectorisation et enrichissement avec des scripts Python modulaires.  

2. **Modélisation**  
   - Popularity Filtering  
   - Content-Based Filtering (TF-IDF + Similarité Cosinus)  
   - Collaborative Filtering (SVD – Surprise library)  
   - Hybrid Filtering  
   - NLP Sentiment Analysis (Naïve Bayes, SVC)  

3. **Évaluation**  
   - Métriques : **Précision@K, Rappel, Diversité**  
   - Résultats :  
     - Content-Based → meilleure précision (~47,4%)  
     - Hybride → meilleure couverture  
     - Bernoulli Naïve Bayes → meilleur modèle de sentiment (70,43%)  

4. **Déploiement**  
   - Application **Streamlit** :  
     - Sélection d’un produit  
     - Choix du modèle de recommandation  
     - Recommandations affichées avec images, prix, note, etc.  
     - Téléchargement des résultats en **CSV**  

---

## ⚙️ Technologies utilisées

- **Langage** : Python  
- **Data Processing** : PySpark, Pandas, NumPy  
- **Base de données** : MongoDB  
- **Machine Learning** : Scikit-learn, Surprise (SVD)  
- **NLP** : CountVectorizer, TF-IDF, Naïve Bayes, SVC  
- **Visualisation** : Power BI, Matplotlib  
- **Interface** : Streamlit  

---

## 📊 Résultats clés

- **Content-Based Filtering** → Précision : 47,4%  
- **Collaborative (SVD)** → Bonne performance à grande échelle  
- **Hybride** → Meilleure couverture & robustesse  
- **NLP Bernoulli Naïve Bayes** → 70,43% de précision sur l’analyse de sentiment  

---

## 🔮 Perspectives

- Améliorer le NLP avec des modèles modernes (**BERT, RoBERTa, DistilBERT**)  
- Ajouter plus de signaux utilisateurs (clics, temps passé, paniers abandonnés)  
- Déployer sur le **cloud** avec une API REST  
- Intégrer des **Graph Neural Networks (GNN)** pour la recommandation  
- Tester en conditions réelles avec des **A/B tests**  

---

## 👩‍💻 Auteurs

- **Chaimaa Ahnin**  
- **Mouna Bnyiche**  
- **Aya Eddaoudy**  
- **Nihad El Alami**  

Projet encadré par **Pr. Imad Sassi**  
Année Universitaire : **2024-2025**

