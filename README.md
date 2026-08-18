# mod-le_pr-diction_-pid-mie
Modèle de prédiction d'épidémie de paludisme et de dengue au Burkina Faso

# SysSurv BF — Epidemic Tracking BF

**Plateforme d'intelligence épidémiologique pour la prédiction et la surveillance du paludisme et de la dengue au Burkina Faso.**

Système de prédiction basé sur des modèles **Random Forest** optimisés, restitué via un tableau de bord web interactif destiné aux acteurs de la santé publique.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-000000?logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?logo=scikitlearn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly.js-Dashboard-3F4F75?logo=plotly&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Sommaire

- [Aperçu](#-aperçu)
- [Fonctionnalités](#-fonctionnalités)
- [Architecture du projet](#-architecture-du-projet)
- [Modèles de prédiction](#-modèles-de-prédiction)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [API](#-api)
- [Stack technique](#-stack-technique)
- [Jeu de données](#-jeu-de-données)
- [Auteurs](#-auteurs)
- [Licence](#-licence)

---

## Aperçu

Le paludisme et la dengue restent deux maladies vectorielles majeures au Burkina Faso. **SysSurv BF** exploite des données climatiques (pluviométrie, température), sanitaires (couverture MILDA, campagnes vaccinales) et temporelles pour anticiper l'évolution hebdomadaire des cas dans **7 régions sanitaires**, sur la base de **13 ans de données** (2013-2026).

Le projet couvre l'ensemble de la chaîne :
1. Génération / prétraitement d'un jeu de données de surveillance hebdomadaire ;
2. Analyse exploratoire (saisonnalité, corrélations, valeurs aberrantes) ;
3. Entraînement et optimisation de deux modèles Random Forest indépendants (paludisme, dengue) ;
4. Restitution via une application web Flask avec tableau de bord interactif et système d'alerte.

---

## Fonctionnalités

Le tableau de bord s'organise en **7 modules** :

| Module | Description |
|---|---|
| **Vue d'ensemble** | KPIs en temps réel (cas agrégés, couverture MILDA, régions en alerte), évolution mensuelle filtrée par région, répartition géographique des cas |
| **Carte régionale** | Visualisation géographique interactive des foyers épidémiques, codage couleur par niveau d'alerte |
| **Tendances épidémiologiques** | Saisonnalité, corrélations climatiques, heatmap région × mois, évolution annuelle comparée des deux maladies |
| **Prédiction ML** | Projection à 6 mois à partir des conditions climatiques et de la couverture MILDA saisies par l'utilisateur |
| **Alertes actives** | Classification automatique en 3 niveaux (stable / alerte / critique) avec actions recommandées différenciées |
| **Rapports & Exports** | Génération de rapports CSV (synthèse annuelle, tendances mensuelles, dataset complet) |
| **Données brutes** | Exploration filtrable (région, période, mois) avec statistiques descriptives dynamiques |

### Seuils d'alerte

| Maladie | Seuil « alerte » | Seuil « critique » |
|---|---|---|
| Paludisme (cas/semaine) | 800 | 1 500 |
| Dengue (cas/semaine) | 50 | 100 |

---

## Architecture du projet

```
syssurv/
├── app.py                          # Application Flask principale
├── donnee.py                       # Génération des données simulées
├── requirements.txt
├── model/
│   ├── model_palu_optimise.pkl     # Random Forest — Paludisme
│   ├── model_dengue_optimise.pkl   # Random Forest — Dengue
│   ├── features_palu.pkl           # Liste ordonnée des features
│   └── features_dengue.pkl
├── templates/
│   ├── index.html                  # Page d'accueil vitrine
│   └── dashboard.html              # Tableau de bord (7 modules)
├── static/
│   ├── css/style.css
│   ├── js/script.js
│   └── images/favicon.svg
└── notebooks/
    ├── pretraitement_et_analyse_exploratoire.ipynb
    └── entrainement.ipynb
```

L'application repose sur une architecture web classique en trois couches :
- **Présentation** : templates HTML/Jinja2, CSS, JavaScript, graphiques Plotly.js ;
- **Métier** : routes Flask, chargement des modèles, calcul des indicateurs agrégés ;
- **Données** : jeu de données de surveillance + artefacts de modélisation (`.pkl`).

---

## Modèles de prédiction

Deux modèles **Random Forest Regressor** indépendants, optimisés par `GridSearchCV` combiné à une validation croisée temporelle (`TimeSeriesSplit`, 5 découpages) :

| | Modèle Paludisme | Modèle Dengue |
|---|---|---|
| **n_estimators** | 150 | 150 |
| **max_depth** | 10 | 10 |
| **min_samples_split** | 2 | 2 |
| **min_samples_leaf** | 1 | 1 |
| **MAE (test 2025-2026)** | 155,98 cas | 13,22 cas |
| **RMSE (test 2025-2026)** | 252,20 cas | 38,33 cas |
| **Variables** | pluviométrie, température, couverture MILDA, campagne vaccinale, lags pluie/temp (1-4), mois cyclique, région | température, lags pluie/temp (1-4), mois cyclique, région |

> La pluviométrie est le facteur climatique le plus corrélé au paludisme (r = 0,45). La dengue est en revanche davantage pilotée par la saisonnalité et l'urbanisation (Centre et Hauts-Bassins) que par le climat instantané.

Grille d'hyperparamètres testée :

```python
param_grid = {
    "n_estimators": [100, 150],
    "max_depth": [10, 15, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
}
```

---

## Installation

**Prérequis** : Python 3.10+

```bash
# 1. Cloner le dépôt
git clone https://github.com/<votre-compte>/syssurv-bf.git
cd syssurv-bf

# 2. Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate     # Linux / macOS
venv\Scripts\activate        # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

`requirements.txt` (dépendances principales) :

```
flask
pandas
numpy
joblib
scikit-learn
```

---

## Utilisation

```bash
python app.py
```

L'application est accessible sur **http://localhost:5000**.

- `/` — Page d'accueil de présentation ;
- `/dashboard` — Tableau de bord interactif.

Pour régénérer le jeu de données simulées ou réentraîner les modèles, se référer aux notebooks du dossier `notebooks/` :
1. `pretraitement_et_analyse_exploratoire.ipynb`
2. `entrainement.ipynb`

---

## 🔌 API

| Route | Méthode | Description |
|---|---|---|
| `/` | GET | Page de présentation |
| `/dashboard` | GET | Tableau de bord interactif |
| `/api/predict` | POST | Prédiction à 6 mois pour une région donnée |
| `/api/data/export` | GET | Export du jeu de données au format CSV |

**Exemple de requête `/api/predict`** :

```json
{
  "region": "Centre (Ouaga)",
  "mois": 7,
  "pluie": 120,
  "temperature": 27,
  "milda": 85,
  "vaccination": 1
}
```

---

## Stack technique

| Domaine | Outils |
|---|---|
| Traitement des données | Python, Pandas, NumPy |
| Machine Learning | scikit-learn (Random Forest, GridSearchCV, TimeSeriesSplit) |
| Backend | Flask |
| Frontend / Visualisation | HTML, CSS, JavaScript, Plotly.js |
| Environnement de développement | Jupyter Notebook |

---

## Jeu de données

Faute d'accès aux données réelles du système national d'information sanitaire (DHIS2) pour des raisons de confidentialité, un jeu de données a été **reconstitué à partir de paramètres épidémiologiques et climatiques réalistes** du contexte burkinabè :

- **5 096 observations** (7 régions × 14 années × 52 semaines épidémiologiques)
- Période : **2013 – 2026**
- Régions couvertes : Centre (Ouaga), Hauts-Bassins (Bobo), Sahel, Est, Boucle du Mouhoun, Cascades, Centre-Nord
- Variables : pluviométrie, température, taux de couverture MILDA, campagne vaccinale, cas de paludisme, cas de dengue

---

## Auteurs

Projet réalisé dans le cadre du mémoire de fin de cycle — **Université Virtuelle du Burkina Faso (UVBF)**
*Modèle de prédiction d'épidémie au Burkina Faso : cas du paludisme et de la dengue*

- **TIEMTORÉ Emmanuel Landry**
---

## Licence

Ce projet est distribué sous licence [MIT](LICENSE).

