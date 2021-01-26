from dagster import IOManager, ModeDefinition, io_manager, pipeline, repository, solid

from tune_sklearn import TuneSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from scipy.stats import randint
import numpy as np

@solid
def load_data(context):
    return datasets.load_digits()

@solid
def process_data(context, digits):
    x = digits.data
    y = digits.target
    return train_test_split(x, y, test_size=.2)

@solid
def tune_model(context, data):
    x_train, x_test, y_train, y_test = data
    clf = RandomForestClassifier()
    param_distributions = {
        "n_estimators": randint(20, 80),
        "max_depth": randint(2, 10)
    }

    tune_search = TuneSearchCV(clf, param_distributions, n_trials=3)

    tune_search.fit(x_train, y_train)

    pred = tune_search.predict(x_test)
    accuracy = np.count_nonzero(np.array(pred) == np.array(y_test)) / len(pred)
    return accuracy

@pipeline
def simple_pipeline():
    digits = load_data()
    data = process_data(digits)
    tune_model(data)

