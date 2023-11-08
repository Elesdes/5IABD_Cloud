import pandas as pd
import tensorflow as tf
import autokeras as ak

x_train = pd.read_csv("train.csv")[:25000]
x_test = pd.read_csv("test.csv")[:25000]

x_val = pd.read_csv("val.csv")

if __name__ == '__main__':
    # x_train as pandas.DataFrame, y_train as pandas.Series
    y_train = x_train.pop("Etiquette_DPE")

    # Preparing testing data.
    y_test = x_test.pop("Etiquette_DPE")

    # It tries 10 different models.
    clf = ak.StructuredDataClassifier(overwrite=True, max_trials=3)
    # Feed the structured data classifier with training data.
    clf.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test), validation_batch_size=32)
    # Predict with the best model.
    predicted_y = clf.predict(x_val)
    # Evaluate the best model with testing data.
    print(clf.evaluate(x_test, y_test))
