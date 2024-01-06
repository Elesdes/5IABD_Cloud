# Databricks notebook source
df = spark.read.options(inferSchema=True).table("traintable")

# COMMAND ----------

display(df)

# COMMAND ----------

df = spark.read.options(inferSchema=True).table("traincleantable")
display(df)
