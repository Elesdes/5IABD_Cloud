# Databricks notebook source
import mlflow
from azure.storage.blob import BlobServiceClient

model = mlflow.pyfunc.load_model(model_uri="dbfs:/databricks/mlflow-tracking/1388524137587605/a270422aed714d279a8468158a0ce90d/artifacts/model")
#model = mlflow.pyfunc.load_model(model_uri="dbfs:/databricks/mlflow-tracking/2944145529935381/2893f639eb4d4865b0c0189028264143/artifacts/model")


# COMMAND ----------

df_val = spark.read.options(inferSchema=True).table("valtable")
print(df_val.select("N°DPE").show())

# COMMAND ----------

#type(df_val.toPandas())
predictions = model.predict(df_val.toPandas())

# COMMAND ----------

print(predictions)
print(type(predictions))

# COMMAND ----------

import pandas as pd
import numpy as np
#df_answer = spark.createDataFrame([df_val.select("N°DPE"),predictions])
df_predictions = pd.DataFrame(predictions)
df_val_pandas = df_val.toPandas()

df_predictions.to_csv('../pred/dump_pred.csv', index=False)
df_val_pandas["N°DPE"].to_csv('../pred/dump_val.csv', index=False)

df2 = pd.read_csv('../pred/dump_pred.csv')
df1 = pd.read_csv('../pred/dump_val.csv')

concatenated_df = pd.concat([df1, df2], ignore_index=True, axis=1)
print(concatenated_df)

concatenated_df.to_csv('../pred/concatenated_file.csv', index=False)
