""" Model train module ( Train and register the Model ) il manque le test """

import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib
import pathlib
import mlflow 


def train_model(
        training_data: str,
        model_name: str,
        tracking_uri:str,

) -> None:
    """ Model training function """

    mlflow.set_tracking_uri(uri=tracking_uri)
    mlflow.set_experiment("MLflow Integration Autolog 3")
    mlflow.sklearn.autolog(registered_model_name=model_name)

    # Load training dataset
    train_df = pd.read_csv(training_data, sep=',')

    # Extract features and labels 
    x_train = train_df.iloc[:,:-1] # First column is indexed 0 but it is the one that Pandas added with indexes, so we start at 1
    y_train = train_df.iloc[:, -1] # As before, we use all lines and here last column

    model = RandomForestClassifier(random_state=42)
    model.fit(
        X=x_train, y=y_train
    )



if __name__ == "__main__":

    # Parseur pour que python puisse lire les arguments du bash
    import argparse
    parser = argparse.ArgumentParser(description="training data and model registry parser")
    parser.add_argument("--training_data", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tracking_uri", type=str)

    args = parser.parse_args()


    # Appel de la fonction pour appel depuis bash
    train_model(training_data=args.training_data, model_name=args.model_name, tracking_uri=args.tracking_uri)


# Pour executer sous bash
# python3 scripts/train_model.py --training_data "data/processed/train.csv" --model_name "integration-autolog" --tracking_uri "http://127.0.0.1:8080"
