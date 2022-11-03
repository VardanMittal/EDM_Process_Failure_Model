from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
import pandas as pd


def Data_processing(X, y):
    """This function helps in process the given data helps in encoding data into numerical data."""
    num_pipeline = Pipeline([
        ("scalar", StandardScaler())
    ])
    cat_pipeline = Pipeline([
        ("encoding", OrdinalEncoder())
    ])
    X_prep = pd.DataFrame(num_pipeline.fit_transform(X))
    y_prep = pd.DataFrame(cat_pipeline.fit_transform(y))

    return X_prep, y_prep
