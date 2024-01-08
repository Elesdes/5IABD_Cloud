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
  headers = {'Authorization': f'Bearer dapi75b8d077aa285f8b0703a62695a8295e', 'Content-Type': 'application/json'}
  ds_dict = dataset.to_dict(orient='split') if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
  data_json = json.dumps(ds_dict, allow_nan=True)
  response = requests.request(method='POST', headers=headers, url=url, data=data_json)
  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

# COMMAND ----------

from mlflow.models.signature import infer_signature
import mlflow
columns = [
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
val = spark.read.options(inferSchema=True).table("valtable")
val = val.select(*columns)
val = val.limit(5).toPandas()
score = score_model(val)
print(score)
print(type(score))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Now the full dataset

# COMMAND ----------

columns = [
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
val = spark.read.options(inferSchema=True).table("valtable")
val = val.select(*columns)
batch_size = 100
offset = 0
while True:
    batch = val.limit(batch_size).toPandas().iloc[offset:]
    print(batch)
    if batch.empty:
        break
    score = score_model(batch)
    print(score)
    offset += batch_size
    break
    
"""    
df_predictions = pd.DataFrame(predictions)
df_val_pandas = df_val.toPandas()

df_predictions.to_csv('../pred/dump_pred.csv', index=False)
df_val_pandas["N°DPE"].to_csv('../pred/dump_val.csv', index=False)

df2 = pd.read_csv('../pred/dump_pred.csv')
df1 = pd.read_csv('../pred/dump_val.csv')

concatenated_df = pd.concat([df1, df2], ignore_index=True, axis=1)
print(concatenated_df)

concatenated_df.to_csv('../pred/concatenated_file.csv', index=False)
"""
