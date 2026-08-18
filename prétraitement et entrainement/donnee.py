import pandas as pd
import numpy as np

# 1. Reproductibilité et Configuration
np.random.seed(2026)

# Extension des variables pour atteindre ~5000 lignes 
# (7 régions * 14 ans * 52 semaines = 5096 entrées)
regions = ['Centre (Ouaga)', 'Hauts-Bassins (Bobo)', 'Sahel', 'Est', 'Boucle du Mouhoun', 'Cascades', 'Centre-Nord']
annees = list(range(2013, 2027)) # De 2013 à 2026 inclus
semaines = list(range(1, 53)) 

data_rows = []

# 2. Boucle pour générer l'historique par Région / Année / Semaine
for region in regions:
    for annee in annees:
        for semaine in semaines:
            
            # --- FACTEURS CLIMATIQUES (Variables explicatives) ---
            # Pluviométrie (mm) : Saison des pluies forte entre juin (S22) et octobre (S40)
            if 22 <= semaine <= 40:
                pluviometrie = np.random.uniform(20, 120) if region != 'Sahel' else np.random.uniform(5, 45)
            else:
                pluviometrie = np.random.uniform(0, 5)
                
            # Température moyenne (°C) : Forte chaleur en mars/avril (S10 à S18)
            if 10 <= semaine <= 18:
                temperature = np.random.uniform(34, 41)
            else:
                temperature = np.random.uniform(25, 33)
                
            # --- DYNAMIQUE PALUDISME ---
            base_palu = 500 if region in ['Centre (Ouaga)', 'Hauts-Bassins (Bobo)'] else 300
            # Tendances de fond : baisse progressive avec les années due au déploiement des soins
            facteur_temporel = 1.0 - (annee - 2013) * 0.02 
            
            if 26 <= semaine <= 46: # Pic saisonnier (Juillet à Novembre)
                cas_palu = int(base_palu * np.random.uniform(3.5, 6.0) * facteur_temporel)
            else:
                cas_palu = int(base_palu * np.random.uniform(0.6, 1.2) * facteur_temporel)
            
            cas_palu = max(0, cas_palu) # Éviter les nombres négatifs
                
            # --- DYNAMIQUE DENGUE ---
            # La dengue émerge fortement en milieu urbain à partir de 2016-2017 au Burkina
            if region in ['Centre (Ouaga)', 'Hauts-Bassins (Bobo)']:
                if annee == 2023 and (35 <= semaine <= 48): # Grande épidémie historique de 2023
                    cas_dengue = int(np.random.uniform(400, 1500))
                elif annee >= 2020 and (38 <= semaine <= 46): # Pics annuels récents
                    cas_dengue = int(np.random.uniform(120, 400))
                elif 38 <= semaine <= 46: # Pics annuels plus anciens (2013-2019)
                    cas_dengue = int(np.random.uniform(30, 100))
                else:
                    cas_dengue = int(np.random.uniform(5, 30))
            else:
                # Hors grandes villes, la dengue reste très sporadique et isolée
                cas_dengue = int(np.random.uniform(0, 15))

            # --- RIPOSTE INSTITUTIONNELLE ---
            taux_milda = np.random.uniform(75, 95)
            # Campagne vaccinale active (simulée à partir de 2024 pour le paludisme)
            campagne_vaccination = 1 if (annee >= 2024 and semaine in [15, 16, 25, 26]) else 0

            # Ajout de la ligne
            data_rows.append({
                'Region': region,
                'Annee': annee,
                'Semaine_Epi': semaine,
                'Pluviometrie_mm': round(pluviometrie, 1),
                'Temperature_Moy_C': round(temperature, 1),
                'Taux_Couverture_MILDA': round(taux_milda, 1),
                'Campagne_Vaccinale_Active': campagne_vaccination,
                'Cas_Paludisme': cas_palu,
                'Cas_Dengue': cas_dengue
            })

# 3. Création du DataFrame et Sauvegarde
df_surveillance_large = pd.DataFrame(data_rows)
df_surveillance_large.to_csv('surveillance_sanitaire_bf_5000.csv', index=False)

print(f"Jeu de données étendu généré avec succès : {df_surveillance_large.shape[0]} lignes.")