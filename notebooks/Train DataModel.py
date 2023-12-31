# Databricks notebook source
# MAGIC %md
# MAGIC On prend notre modèle métier

# COMMAND ----------

import mlflow
experiment_id = mlflow.create_experiment("/Users/erwan.duprey@free.fr/Experiment_Test")
with mlflow.start_run(
    run_name="PARENT_RUN",
    experiment_id=experiment_id,
    tags={"version": "v1", "priority": "P1"},
    description="Exp_train_model",
) as parent_run:
    mlflow.log_param("parent", "yes")
    with mlflow.start_run(
        run_name="CHILD_RUN",
        experiment_id=experiment_id,
        description="child",
        nested=True,
    ) as child_run:
        mlflow.log_param("child", "yes")

print("parent run:")

print(f"run_id: {parent_run.info.run_id}")
print("description: {}".format(parent_run.data.tags.get("mlflow.note.content")))
print("version tag value: {}".format(parent_run.data.tags.get("version")))
print("priority tag value: {}".format(parent_run.data.tags.get("priority")))
print("--")

# Search all child runs with a parent id
query = f"tags.mlflow.parentRunId = '{parent_run.info.run_id}'"
results = mlflow.search_runs(experiment_ids=[experiment_id], filter_string=query)
print("child runs:")
print(results[["run_id", "params.child", "tags.mlflow.runName"]])
