"""
SysSurv BF — Application Flask
Plateforme de Surveillance Épidémiologique du Burkina Faso
Paludisme & Dengue · Modèles Random Forest optimisés
"""

import json
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# INITIALISATION FLASK
# ═══════════════════════════════════════════════════════════════
app = Flask(__name__)

# ═══════════════════════════════════════════════════════════════
# CONSTANTES & SEUILS
# ═══════════════════════════════════════════════════════════════
REGIONS = [
    "Centre (Ouaga)", "Hauts-Bassins (Bobo)", "Sahel",
    "Est", "Boucle du Mouhoun", "Cascades", "Centre-Nord",
]
REGIONS_OHE = sorted(REGIONS)

SEUIL_PALU_CRITIQUE   = 1500
SEUIL_PALU_ALERTE     = 800
SEUIL_DENGUE_CRITIQUE = 100
SEUIL_DENGUE_ALERTE   = 50

MOIS_LABELS = {
    1:"Janvier",2:"Février",3:"Mars",4:"Avril",5:"Mai",6:"Juin",
    7:"Juillet",8:"Août",9:"Septembre",10:"Octobre",11:"Novembre",12:"Décembre"
}
MOIS_ABBR = ["Jan","Fév","Mar","Avr","Mai","Jun","Jul","Aoû","Sep","Oct","Nov","Déc"]

ACTIONS_PALU = {
    "critique": ["Distribution massive de MILDA en urgence",
                 "Renforcement immédiat du dépistage",
                 "Activation des centres de santé de référence",
                 "Pulvérisation intra-domiciliaire"],
    "warning":  ["Sensibilisation active des populations",
                 "Renforcement des stocks antipaludéens",
                 "Surveillance hebdomadaire renforcée"],
    "stable":   ["Maintien de la surveillance standard"],
}
ACTIONS_DENGUE = {
    "critique": ["Pulvérisation intra-domiciliaire d'urgence",
                 "Élimination des gîtes larvaires",
                 "Sensibilisation communautaire intensive",
                 "Renforcement du dépistage sérologique"],
    "warning":  ["Surveillance des cas suspects",
                 "Campagne d'élimination des eaux stagnantes"],
    "stable":   ["Maintien de la surveillance standard"],
}

# ═══════════════════════════════════════════════════════════════
# CHARGEMENT DES MODÈLES
# ═══════════════════════════════════════════════════════════════
def load_models():
    mp = joblib.load("model/model_palu_optimise.pkl")
    md = joblib.load("model/model_dengue_optimise.pkl")
    fp = joblib.load("model/features_palu.pkl")
    fd = joblib.load("model/features_dengue.pkl")
    sp = sd = None
    try:
        sp = joblib.load("model/scaler_palu.pkl")
        sd = joblib.load("model/scaler_dengue.pkl")
    except FileNotFoundError:
        pass
    return mp, md, fp, fd, sp, sd

try:
    model_palu, model_dengue, features_palu, features_dengue, scaler_p, scaler_d = load_models()
    print("Modèles ML chargés avec succès.")
except FileNotFoundError as e:
    print(f"Modèles introuvables ({e}). Prédictions désactivées.")
    model_palu = model_dengue = features_palu = features_dengue = None
    scaler_p = scaler_d = None

# ═══════════════════════════════════════════════════════════════
# DONNÉES HISTORIQUES
# ═══════════════════════════════════════════════════════════════
def generate_historical_data():
    np.random.seed(2026)
    rows = []
    for region in REGIONS:
        for annee in range(2013, 2027):
            for semaine in range(1, 53):
                mois = min(12, max(1, int((semaine - 1) / 4.33) + 1))
                if 22 <= semaine <= 40:
                    pluie = np.random.uniform(20,120) if region != "Sahel" else np.random.uniform(5,45)
                else:
                    pluie = np.random.uniform(0, 5)
                temp = np.random.uniform(34,41) if 10 <= semaine <= 18 else np.random.uniform(25,33)
                base_palu = 500 if region in ["Centre (Ouaga)","Hauts-Bassins (Bobo)"] else 300
                facteur_t = 1.0 - (annee - 2013) * 0.02
                cas_palu = (int(base_palu * np.random.uniform(3.5,6.0) * facteur_t)
                            if 26 <= semaine <= 46
                            else int(base_palu * np.random.uniform(0.6,1.2) * facteur_t))
                cas_palu = max(0, cas_palu)
                if region in ["Centre (Ouaga)","Hauts-Bassins (Bobo)"]:
                    if annee == 2023 and 35 <= semaine <= 48:
                        cas_dengue = int(np.random.uniform(400,1500))
                    elif annee >= 2020 and 38 <= semaine <= 46:
                        cas_dengue = int(np.random.uniform(120,400))
                    elif 38 <= semaine <= 46:
                        cas_dengue = int(np.random.uniform(30,100))
                    else:
                        cas_dengue = int(np.random.uniform(5,30))
                else:
                    cas_dengue = int(np.random.uniform(0,15))
                rows.append({
                    "Region": region, "Annee": annee, "Semaine": semaine, "Mois": mois,
                    "Pluviometrie_mm": round(pluie, 1),
                    "Temperature_Moy_C": round(temp, 1),
                    "Taux_Couverture_MILDA": round(np.random.uniform(75,95), 1),
                    "Campagne_Vaccinale_Active": 1 if annee >= 2024 and semaine in [15,16,25,26] else 0,
                    "Cas_Paludisme": cas_palu,
                    "Cas_Dengue": cas_dengue,
                })
    return pd.DataFrame(rows)

print("Génération des données historiques…")
df_historical = generate_historical_data()
print(f"Données prêtes : {len(df_historical):,} enregistrements.")

# ═══════════════════════════════════════════════════════════════
# UTILITAIRES ML
# ═══════════════════════════════════════════════════════════════
def build_input_df(region, mois, pluie, temp, milda, vaccin, pluie_lags, temp_lags):
    mois_sin = float(np.sin(2 * np.pi * mois / 12))
    mois_cos = float(np.cos(2 * np.pi * mois / 12))
    row = {
        "pluviometrie_mm": pluie, "temperature_moy_c": temp,
        "taux_couverture_milda": milda, "campagne_vaccinale_active": vaccin,
        "pluie_lag_1": pluie_lags[0], "pluie_lag_2": pluie_lags[1],
        "pluie_lag_3": pluie_lags[2], "pluie_lag_4": pluie_lags[3],
        "temp_lag_1": temp_lags[0],   "temp_lag_2": temp_lags[1],
        "temp_lag_3": temp_lags[2],   "temp_lag_4": temp_lags[3],
        "mois_sin": mois_sin, "mois_cos": mois_cos,
    }
    for r in REGIONS_OHE:
        row[f"region_{r}"] = 1 if r == region else 0
    return pd.DataFrame([row])

def run_predict(df_in):
    Xp = df_in[features_palu]
    Xd = df_in[features_dengue]
    if scaler_p:
        Xp = scaler_p.transform(Xp)
        Xd = scaler_d.transform(Xd)
    return (max(0, int(model_palu.predict(Xp)[0])),
            max(0, int(model_dengue.predict(Xd)[0])))

def niveau_palu(v):
    if v >= SEUIL_PALU_CRITIQUE: return "critique"
    if v >= SEUIL_PALU_ALERTE:   return "warning"
    return "stable"

def niveau_dengue(v):
    if v >= SEUIL_DENGUE_CRITIQUE: return "critique"
    if v >= SEUIL_DENGUE_ALERTE:   return "warning"
    return "stable"

def build_projection(region, mois_start, pluie, temp, milda, vaccin, pl, tl, n=6):
    results = []
    for i in range(n):
        m = (mois_start + i - 1) % 12 + 1
        df_tmp = build_input_df(region, m, pluie * (0.9 + 0.04*i), temp, milda, vaccin, pl, tl)
        pp, pd_ = run_predict(df_tmp)
        results.append({"mois": MOIS_ABBR[m-1], "palu": pp, "dengue": pd_})
    return results

# ═══════════════════════════════════════════════════════════════
# HELPER DASHBOARD DATA
# ═══════════════════════════════════════════════════════════════
def compute_dashboard_data(annee_sel, region_active):
    df_annee = df_historical[df_historical["Annee"] == annee_sel]
    df_prev  = df_historical[df_historical["Annee"] == annee_sel - 1]

    total_palu   = int(df_annee["Cas_Paludisme"].sum())
    total_dengue = int(df_annee["Cas_Dengue"].sum())
    milda_moy    = round(float(df_annee["Taux_Couverture_MILDA"].mean()), 1)
    palu_prev    = int(df_prev["Cas_Paludisme"].sum())  if not df_prev.empty else total_palu
    dengue_prev  = int(df_prev["Cas_Dengue"].sum())    if not df_prev.empty else total_dengue
    delta_palu   = round((total_palu - palu_prev) / palu_prev * 100, 1)   if palu_prev   else 0.0
    delta_dengue = round((total_dengue - dengue_prev) / dengue_prev * 100, 1) if dengue_prev else 0.0
    reg_sum = df_annee.groupby("Region")["Cas_Paludisme"].sum()
    regions_alerte = int((reg_sum >= SEUIL_PALU_ALERTE * 52).sum())

    # évolution mensuelle
    df_mois = (df_annee[df_annee["Region"] == region_active]
               .groupby("Mois")
               .agg(Palu=("Cas_Paludisme","sum"), Dengue=("Cas_Dengue","sum"))
               .reset_index())
    chart_evolution = {
        "mois":   [MOIS_ABBR[int(m)-1] for m in df_mois["Mois"]],
        "palu":   df_mois["Palu"].tolist(),
        "dengue": df_mois["Dengue"].tolist(),
    }

    # répartition régionale
    df_reg = (df_annee.groupby("Region")
              .agg(Palu=("Cas_Paludisme","sum"), Dengue=("Cas_Dengue","sum"))
              .reset_index().sort_values("Palu", ascending=True))
    chart_regions = {
        "labels": df_reg["Region"].tolist(),
        "palu":   df_reg["Palu"].tolist(),
        "dengue": df_reg["Dengue"].tolist(),
    }

    # saisonnalité
    df_sais = (df_historical[df_historical["Region"] == region_active]
               .groupby("Mois")
               .agg(Palu=("Cas_Paludisme","mean"), Dengue=("Cas_Dengue","mean"))
               .reset_index())
    chart_saisonnalite = {
        "mois":   [MOIS_ABBR[int(m)-1] for m in df_sais["Mois"]],
        "palu":   [round(v,0) for v in df_sais["Palu"]],
        "dengue": [round(v,0) for v in df_sais["Dengue"]],
    }

    # annuel par région
    df_ann = (df_historical.groupby(["Annee","Region"])
              .agg(Palu=("Cas_Paludisme","sum"), Dengue=("Cas_Dengue","sum"))
              .reset_index())
    chart_annuel = {}
    for r in REGIONS:
        sub = df_ann[df_ann["Region"] == r].sort_values("Annee")
        chart_annuel[r] = {"annees": sub["Annee"].tolist(),
                           "palu":   sub["Palu"].tolist(),
                           "dengue": sub["Dengue"].tolist()}

    # corrélation
    df_c = df_historical[df_historical["Region"] == region_active]
    chart_corr = {
        "pluie": df_c["Pluviometrie_mm"].tolist(),
        "temp":  df_c["Temperature_Moy_C"].tolist(),
        "palu":  df_c["Cas_Paludisme"].tolist(),
    }

    # heatmap
    df_heat = (df_historical.groupby(["Region","Mois"])["Cas_Paludisme"]
               .mean().reset_index())
    pivot = df_heat.pivot(index="Region", columns="Mois", values="Cas_Paludisme").fillna(0)
    chart_heatmap = {
        "regions": pivot.index.tolist(),
        "mois":    [MOIS_ABBR[int(c)-1] for c in pivot.columns],
        "values":  [[round(v,0) for v in row] for row in pivot.values.tolist()],
    }

    # carte
    coords = {
        "Centre (Ouaga)":       {"lat": 12.365,"lon": -1.534},
        "Hauts-Bassins (Bobo)": {"lat": 11.177,"lon": -4.297},
        "Sahel":                {"lat": 14.297,"lon":  0.053},
        "Est":                  {"lat": 12.363,"lon":  0.352},
        "Boucle du Mouhoun":    {"lat": 12.363,"lon": -3.452},
        "Cascades":             {"lat": 10.617,"lon": -4.770},
        "Centre-Nord":          {"lat": 13.100,"lon": -1.067},
    }
    df_carte = (df_annee.groupby("Region")
                .agg(Palu=("Cas_Paludisme","sum"),
                     Dengue=("Cas_Dengue","sum"),
                     MILDA=("Taux_Couverture_MILDA","mean"))
                .reset_index())
    carte_data = []
    for _, row in df_carte.iterrows():
        niv = ("Critique" if row["Palu"] >= SEUIL_PALU_CRITIQUE*52
               else ("Alerte" if row["Palu"] >= SEUIL_PALU_ALERTE*52 else "Stable"))
        carte_data.append({
            "region": row["Region"],
            "lat": coords[row["Region"]]["lat"],
            "lon": coords[row["Region"]]["lon"],
            "palu":   int(row["Palu"]),
            "dengue": int(row["Dengue"]),
            "milda":  round(float(row["MILDA"]),1),
            "niveau": niv,
        })

    # tableau
    df_table = (df_annee.groupby("Region")
                .agg(Palu=("Cas_Paludisme","sum"),
                     Dengue=("Cas_Dengue","sum"),
                     MILDA=("Taux_Couverture_MILDA","mean"),
                     Pluie=("Pluviometrie_mm","mean"),
                     Temp=("Temperature_Moy_C","mean"))
                .reset_index().sort_values("Palu", ascending=False))
    table_rows = []
    for _, row in df_table.iterrows():
        niv = ("critique" if row["Palu"] >= SEUIL_PALU_CRITIQUE*52
               else ("warning" if row["Palu"] >= SEUIL_PALU_ALERTE*52 else "stable"))
        table_rows.append({
            "region": row["Region"],
            "palu":   f"{int(row['Palu']):,}",
            "dengue": f"{int(row['Dengue']):,}",
            "milda":  f"{row['MILDA']:.1f}",
            "pluie":  f"{row['Pluie']:.1f}",
            "temp":   f"{row['Temp']:.1f}",
            "niveau": niv,
            "badge":  ("Critique" if niv=="critique" else ("Alerte" if niv=="warning" else "Stable")),
        })

    # alertes
    alertes = []
    for _, row in df_table.iterrows():
        palu_sem = int(row["Palu"]) // 52
        deng_sem = int(row["Dengue"]) // 52
        niv_p = niveau_palu(palu_sem)
        niv_d = niveau_dengue(deng_sem)
        if niv_p != "stable":
            alertes.append({"region": row["Region"], "maladie": "Paludisme",
                            "cas": palu_sem, "niveau": niv_p, "actions": ACTIONS_PALU[niv_p]})
        if niv_d != "stable":
            alertes.append({"region": row["Region"], "maladie": "Dengue",
                            "cas": deng_sem, "niveau": niv_d, "actions": ACTIONS_DENGUE[niv_d]})
    alertes.sort(key=lambda a: (0 if a["niveau"]=="critique" else 1, -a["cas"]))

    return {
        "total_palu": total_palu, "total_dengue": total_dengue,
        "milda_moy": milda_moy,  "regions_alerte": regions_alerte,
        "delta_palu": delta_palu, "delta_dengue": delta_dengue,
        "regions": REGIONS, "annee_sel": annee_sel, "region_active": region_active,
        "current_date": datetime.now().strftime("%d/%m/%Y"),
        "modeles_ok": model_palu is not None,
        "chart_evolution":    json.dumps(chart_evolution),
        "chart_regions":      json.dumps(chart_regions),
        "chart_saisonnalite": json.dumps(chart_saisonnalite),
        "chart_annuel":       json.dumps(chart_annuel),
        "chart_corr":         json.dumps(chart_corr),
        "chart_heatmap":      json.dumps(chart_heatmap),
        "carte_data":         json.dumps(carte_data),
        "table_rows": table_rows,
        "alertes": alertes,
        "n_critique": sum(1 for a in alertes if a["niveau"]=="critique"),
        "n_warning":  sum(1 for a in alertes if a["niveau"]=="warning"),
        "n_stable":   7 - len({a["region"] for a in alertes}),
        "mois_labels": MOIS_LABELS,
        "mois_abbr": MOIS_ABBR,
    }

# ═══════════════════════════════════════════════════════════════
# ROUTES
# ═══════════════════════════════════════════════════════════════
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    annee_sel     = request.args.get("annee",  default=2026,             type=int)
    region_active = request.args.get("region", default="Centre (Ouaga)", type=str)
    ctx = compute_dashboard_data(annee_sel, region_active)
    return render_template("dashboard.html", **ctx)

@app.route("/api/predict", methods=["POST"])
def api_predict():
    if model_palu is None:
        return jsonify({"error": "Modèles ML indisponibles. Vérifiez le dossier /model/"}), 503
    try:
        data   = request.get_json(force=True)
        region = data.get("region", "Centre (Ouaga)")
        mois   = int(data.get("mois",  6))
        pluie  = float(data.get("pluie",  80.0))
        temp   = float(data.get("temp",   30.0))
        milda  = float(data.get("milda",  85.0))
        vaccin = int(data.get("vaccin",   0))
        pl     = [float(x) for x in data.get("pluie_lags", [70,60,50,40])]
        tl     = [float(x) for x in data.get("temp_lags",  [29,28,27,27])]

        df_in = build_input_df(region, mois, pluie, temp, milda, vaccin, pl, tl)
        pred_palu, pred_dengue = run_predict(df_in)
        niv_p = niveau_palu(pred_palu)
        niv_d = niveau_dengue(pred_dengue)
        saison = "Pluvieuse" if 5 <= mois <= 10 else "Sèche"
        projection = build_projection(region, mois, pluie, temp, milda, vaccin, pl, tl)

        return jsonify({
            "status": "success",
            "region": region,
            "mois": MOIS_LABELS.get(mois, str(mois)),
            "saison": saison,
            "paludisme": {"prediction": pred_palu, "niveau": niv_p, "actions": ACTIONS_PALU[niv_p]},
            "dengue":    {"prediction": pred_dengue,"niveau": niv_d, "actions": ACTIONS_DENGUE[niv_d]},
            "projection": projection,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/data/export")
def api_data_export():
    csv_data = df_historical.to_csv(index=False)
    return Response(csv_data, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment;filename=surveillance_bf.csv"})

# ═══════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n" + "═"*58)
    print("  🩺  SysSurv BF — Serveur Flask")
    print("═"*58)
    print("  Accueil    : http://127.0.0.1:5000/")
    print("  Dashboard  : http://127.0.0.1:5000/dashboard")
    print("  API predict: POST http://127.0.0.1:5000/api/predict")
    print("  Export CSV : http://127.0.0.1:5000/api/data/export")
    print("═"*58 + "\n")
    app.run(debug=True, port=5000)