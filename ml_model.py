import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load

def load_data():
    # Load the housing datapip install joblib
    housing = pd.read_csv("HousingData.csv")

    # Train-Test Splitting
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

    # Removing NaN values
    imputer = SimpleImputer(strategy="median")
    imputer.fit(housing)
    x = imputer.transform(housing)
    housing_tr = pd.DataFrame(x, columns=housing.columns)
    housing.fillna(housing.mean(), inplace=True)

    # Stratified Shuffle Split
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['CHAS']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    return strat_train_set, strat_test_set

def train_model(data):
    # Preparing the data for training
    housing_labels = data["MEDV"].copy()
    housing = data.drop("MEDV", axis=1)

    # Creating a pipeline for data preprocessing
    my_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    # Transforming numerical data using the pipeline
    housing_num_tr = my_pipeline.fit_transform(housing)

    # Selecting a desired model for Dragon Real Estate
    model = RandomForestRegressor()
    model.fit(housing_num_tr, housing_labels)

    return model

def save_model(model, filename):
    # Saving the model
    dump(model, filename)

def load_model(filename):
    # Loading the model
    return load(filename)

def predict_house_price(features):
    # Load the trained model
    model = load_model('Dragon.joblib')

    # Perform any necessary data preprocessing on the input features (if required)
    # For example, you can use the same pipeline used during training

    # Make predictions
    predicted_price = model.predict([features])
    return predicted_price[0]
