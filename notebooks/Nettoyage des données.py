# Databricks notebook source
tables = spark.catalog.listTables()
for table in tables:
   print(table.name)

# COMMAND ----------

spark.read.options(inferSchema=True).table("traintable").count()

# COMMAND ----------

# Colonnes que l'on va garder
cols_with_y = ['Emission_GES_éclairage',
       'Conso_chauffage_dépensier_installation_chauffage_n°1', 'Etiquette_GES',
       'Classe_altitude', 'Conso_5_usages/m²_é_finale',
       'Conso_5_usages_é_finale', 'Etiquette_DPE', 'Hauteur_sous-plafond',
       'Qualité_isolation_enveloppe', 'Qualité_isolation_menuiseries',
       'Qualité_isolation_murs', 'Qualité_isolation_plancher_bas',
       'Qualité_isolation_plancher_haut_comble_perdu',
       'Surface_habitable_logement', 'Type_bâtiment']


# COMMAND ----------

# Colonnes que l'on va garder
"""
cols_with_y = ['Conso_chauffage_dépensier_installation_chauffage_n°1',
       'Conso_5_usages/m²_é_finale', 'Conso_5_usages_é_finale', 'Etiquette_DPE',
       'Qualité_isolation_enveloppe', 'Qualité_isolation_menuiseries',
       'Qualité_isolation_murs', 'Qualité_isolation_plancher_bas',
       'Qualité_isolation_plancher_haut_comble_perdu',
       'Surface_habitable_logement', 'Type_bâtiment']"""

# COMMAND ----------

df_train = spark.read.options(inferSchema=True).table("traintable")
df_test = spark.read.options(inferSchema=True).table("testtable")

df_train = df_train.select(cols_with_y)
df_test = df_test.select(cols_with_y)

df_train = df_train.na.drop()
df_test = df_test.na.drop()

df_train = df_train.sample(fraction=50000 / df_train.count(), seed=404)
df_test = df_test.sample(fraction=50000 / df_test.count(), seed=404)


# COMMAND ----------

df_train.count()

# COMMAND ----------

display(df_train)

# COMMAND ----------

from pyspark.sql.functions import count, col

# Categories name
categories = df_train.select("Etiquette_DPE").distinct()

# Count occurrences of each value
value_counts = df_train.groupBy('Etiquette_DPE').agg(count(col('Etiquette_DPE')).alias('count'))

# Save min length
min_length = value_counts.orderBy('count').first()["count"]


# COMMAND ----------

# MAGIC %md
# MAGIC # Normalize data

# COMMAND ----------

from functools import reduce
import pyspark

# Assuming your DataFrame is named df and the column is 'Etiquette_GES'
total_rows = df_train.count()
dfs = []

# Calculate the fraction for each category
for category in categories:
    dfs.append(df_train.filter(df_train["Etiquette_DPE"] == category).limit(min_length))

sampled_data = reduce(pyspark.sql.DataFrame.union, dfs)

# COMMAND ----------

sampled_data.groupBy('Etiquette_DPE').agg(count(col('Etiquette_DPE')).alias('count2')).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Utile si l'on veut mettre les catégories au même nombre

# COMMAND ----------

"""
categories = ["A", "B", "C", "D", "E", "F", "G"]
max_categories = "12470"
for categorie in categories:
    df_limited = spark.sql("SELECT " + " * FROM traintable WHERE Etiquette_DPE = \""+ categorie + "\" LIMIT " + max_categories)
display(df_limited)
"""

# COMMAND ----------

"""
nb_exemples_classe_minoritaire = df_train.filter(df_train.col("Etiquette_DPE") == "A").count()
df_equilibre = df_train.groupby("Etiquette_DPE").\
    agg(*[df_train.col("Etiquette_DPE")] + [df_train.col("Etiquette_DPE")] * (nb_exemples_classe_minoritaire - 1)).\
    select(*[df_train.col("Etiquette_DPE")]).\
    sample(True, 1.0).\
    orderBy("Etiquette_DPE")

print(len(df_equilibre))
"""

# COMMAND ----------

#min_length_df = df_train.groupBy("Etiquette_DPE").agg(min(len("")))

# COMMAND ----------

df_train.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("TrainCleanTable")
df_test.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable("TestCleanTable")
