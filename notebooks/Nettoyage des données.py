# Databricks notebook source
tables = spark.catalog.listTables()
for table in tables:
   print(table.name)

# COMMAND ----------

spark.read.table("traintable").count()

# COMMAND ----------

# Colonnes que l'on va garder
cols_with_y = ["Surface_habitable_desservie_par_installation_ECS", "Emission_GES_éclairage", "Conso_5_usages_é_finale_énergie_n°2", "Conso_chauffage_dépensier_installation_chauffage_n°1", "Emission_GES_chauffage_énergie_n°2", "Etiquette_GES", "Année_construction", "Conso_5_usages/m²_é_finale", "Conso_5_usages_é_finale","Etiquette_DPE", "Qualité_isolation_enveloppe","Qualité_isolation_menuiseries","Qualité_isolation_murs","Qualité_isolation_plancher_bas","Surface_habitable_logement","Type_bâtiment"]


# COMMAND ----------

df_train = spark.read.table("traintable")
df_test = spark.read.table("testtable")

df_train = df_train.select(cols_with_y)
df_test = df_test.select(cols_with_y)

#df_train = df_train.na.drop()
#df_test = df_test.na.drop()

# COMMAND ----------

df_train.count()

# COMMAND ----------

display(df_train)

# COMMAND ----------

df_train.write.mode("overwrite").saveAsTable("TrainCleanTable")
df_test.write.mode("overwrite").saveAsTable("TestCleanTable")
