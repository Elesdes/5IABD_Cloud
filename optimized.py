import pandas as pd
import tensorflow as tf
from keras.models import load_model
import autokeras as ak
import csv

cols = ["Surface_habitable_desservie_par_installation_ECS", "Emission_GES_éclairage", "Conso_5_usages_é_finale_énergie_n°2", "Conso_chauffage_dépensier_installation_chauffage_n°1", "Emission_GES_chauffage_énergie_n°2", "Etiquette_GES", "Année_construction", "Conso_5_usages/m²_é_finale", "Conso_5_usages_é_finale","Qualité_isolation_enveloppe","Qualité_isolation_menuiseries","Qualité_isolation_murs","Qualité_isolation_plancher_bas","Qualité_isolation_plancher_haut_comble_perdu","Surface_habitable_logement","Type_bâtiment"]
cols_with_y = ["Surface_habitable_desservie_par_installation_ECS", "Emission_GES_éclairage", "Conso_5_usages_é_finale_énergie_n°2", "Conso_chauffage_dépensier_installation_chauffage_n°1", "Emission_GES_chauffage_énergie_n°2", "Etiquette_GES", "Année_construction", "Conso_5_usages/m²_é_finale", "Conso_5_usages_é_finale","Etiquette_DPE", "Qualité_isolation_enveloppe","Qualité_isolation_menuiseries","Qualité_isolation_murs","Qualité_isolation_plancher_bas","Qualité_isolation_plancher_haut_comble_perdu","Surface_habitable_logement","Type_bâtiment"]


x_train = pd.read_csv("train.csv", usecols=cols_with_y)
x_test = pd.read_csv("test.csv", usecols=cols_with_y)
x_val = pd.read_csv("val.csv", usecols=cols)

x_train = x_train.dropna()
x_test = x_test.dropna()


if __name__ == '__main__':
    y_train = x_train.pop("Etiquette_DPE")
    y_test = x_test.pop("Etiquette_DPE")

    # It tries 10 different models.
    clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
    clf.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), validation_batch_size=32)
    # Predict with the best model.
    predicted_y = clf.predict(x_val)
    # Evaluate the best model with testing data.
    print(clf.evaluate(x_test, y_test))
    model = clf.export_model()
    try:
        model.save("model_autokeras", save_format="tf")
    except Exception:
        model.save("model_autokeras.h5")
    """
    loaded_model = load_model("Models/model_autokeras", custom_objects=ak.CUSTOM_OBJECTS)

    predicted_y = loaded_model.predict(x_val)
    """
    with open('output.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["N°DPE", "Etiquette_DPE"])
        writer.writerow([pd.read_csv("val.csv", usecols=["N°DPE"]), predicted_y])

