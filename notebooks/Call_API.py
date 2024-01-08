# Databricks notebook source

import os
import requests
import numpy as np
import pandas as pd
import json

def create_tf_serving_json(data):
  return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset):
  url = 'https://adb-8468160794787762.2.azuredatabricks.net/model/To_API/1/invocations'
  headers = {'Authorization': f'Bearer {os.environ.get("DATABRICKS_TOKEN")}', 'Content-Type': 'application/json'}
  ds_dict = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

dataset = {
  "columns": [
    "Emission_GES_éclairage",
    "Conso_chauffage_dépensier_installation_chauffage_n°1",
    "Etiquette_GES",
    "Classe_altitude",
    "Conso_5_usages/m²_é_finale",
    "Conso_5_usages_é_finale",
    "Hauteur_sous-plafond",
    "Qualité_isolation_enveloppe",
    "Qualité_isolation_menuiseries",
    "Qualité_isolation_murs",
    "Qualité_isolation_plancher_bas",
    "Qualité_isolation_plancher_haut_comble_perdu",
    "Surface_habitable_logement",
    "Type_bâtiment"
  ],
  "data": [
    [
      23.8,
      1,
      "D",
      "400-800m",
      158,
      29359.9,
      2.5,
      "bonne",
      "moyenne",
      "bonne",
      "moyenne",
      "bonne",
      185.3,
      "maison"
    ],
    [
      6.1,
      5254.1,
      "B",
      "inférieur à 400m",
      127.3,
      6003.5,
      2.5,
      "bonne",
      "moyenne",
      "bonne",
      "moyenne",
      "très bonne",
      47.2,
      "maison"
    ],
    [
      2.6,
      16518.3,
      "E",
      "inférieur à 400m",
      246.6,
      17261.4,
      2.5,
      "insuffisante",
      "moyenne",
      "insuffisante",
      "très bonne",
      None,
      70,
      "appartement"
    ],
    [
      20,
      1,
      "C",
      "inférieur à 400m",
      126,
      19664.3,
      2.5,
      "bonne",
      "bonne",
      "bonne",
      "insuffisante",
      None,
      156,
      "maison"
    ],
    [
      12.1,
      8354,
      "C",
      "inférieur à 400m",
      94.8,
      8811.9,
      2.5,
      "bonne",
      "très bonne",
      "moyenne",
      "moyenne",
      "très bonne",
      93,
      "maison"
    ]
  ]
}

# COMMAND ----------

"""columns = [
    "Emission_GES_éclairage",
    "Conso_chauffage_dépensier_installation_chauffage_n°1",
    "Etiquette_GES",
    "Classe_altitude",
    "Conso_5_usages/m²_é_finale",
    "Conso_5_usages_é_finale",
    "Hauteur_sous-plafond",
    "Qualité_isolation_enveloppe",
    "Qualité_isolation_menuiseries",
    "Qualité_isolation_murs",
    "Qualité_isolation_plancher_bas",
    "Qualité_isolation_plancher_haut_comble_perdu",
    "Surface_habitable_logement",
    "Type_bâtiment"
  ]

# Sélectionner uniquement les colonnes spécifiées
val_selected = val.select(*selected_cols)"""

# COMMAND ----------

from mlflow.models.signature import infer_signature
import mlflow
val = spark.read.options(inferSchema=True).table("valtable")
val = val.toPandas()
score = score_model(val)
print(score)

