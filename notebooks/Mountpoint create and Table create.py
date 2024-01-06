# Databricks notebook source
dbutils.fs.ls("/mnt/mntgrp1")

# COMMAND ----------

"""
dbutils.fs.mount(
source = "wasbs://stockcsv@azuregrp1.blob.core.windows.net",
mount_point = "/mnt/mntgrp1",
extra_configs = {"fs.azure.account.key.azuregrp1.blob.core.windows.net":dbutils.secrets.get(scope = "scopegrp1", key = "victoria")})"""

# COMMAND ----------

# dbutils.fs.unmount("/mnt/mntgrp1")

# COMMAND ----------

tables = spark.catalog.listTables()
for table in tables:
    spark.sql("DROP TABLE IF EXISTS " + table.name + ";")


# COMMAND ----------

df = spark.read.csv("/mnt/mntgrp1/train.csv", header=True, inferSchema=True)
df = df.drop("_c0")

# COMMAND ----------

df = spark.read.csv("/mnt/mntgrp1/train.csv", header=True, inferSchema=True)
df = df.drop("_c0")

# COMMAND ----------

display(df)

# COMMAND ----------

print(df.columns)

# COMMAND ----------

df = df.withColumnRenamed("N°_département_(BAN)","N°_département_BAN")
df = df.withColumnRenamed("Code_postal_(BAN)","Code_postal_BAN")
df = df.withColumnRenamed("Nom__commune_(Brut)","Nom__commune_Brut")
df = df.withColumnRenamed("Code_INSEE_(BAN)","Code_INSEE_BAN")
df = df.withColumnRenamed("Code_postal_(brut)","Code_postal_brut")

# COMMAND ----------

print(df.columns)

# COMMAND ----------

df.write.mode("overwrite").saveAsTable("TrainTable")

# COMMAND ----------

df = spark.read.csv("/mnt/mntgrp1/test.csv", header=True, inferSchema=True)
df = df.drop("_c0")

# COMMAND ----------

display(df)

# COMMAND ----------

print(df.columns)

# COMMAND ----------

df = df.withColumnRenamed("N°_département_(BAN)","N°_département_BAN")
df = df.withColumnRenamed("Code_postal_(BAN)","Code_postal_BAN")
df = df.withColumnRenamed("Nom__commune_(Brut)","Nom__commune_Brut")
df = df.withColumnRenamed("Code_INSEE_(BAN)","Code_INSEE_BAN")
df = df.withColumnRenamed("Code_postal_(brut)","Code_postal_brut")

# COMMAND ----------

df.write.mode("overwrite").saveAsTable("TestTable")

# COMMAND ----------

df = spark.read.csv("/mnt/mntgrp1/val.csv", header=True, inferSchema=True)
df = df.drop("_c0")

# COMMAND ----------

display(df)

# COMMAND ----------

df = df.withColumnRenamed("N°_département_(BAN)","N°_département_BAN")
df = df.withColumnRenamed("Code_postal_(BAN)","Code_postal_BAN")
df = df.withColumnRenamed("Nom__commune_(Brut)","Nom__commune_Brut")
df = df.withColumnRenamed("Code_INSEE_(BAN)","Code_INSEE_BAN")
df = df.withColumnRenamed("Code_postal_(brut)","Code_postal_brut")

# COMMAND ----------

df.write.mode("overwrite").saveAsTable("ValTable")
