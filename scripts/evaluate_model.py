""" Model evaluation module (il manque l'evaluation, on utilise test car on l'a pas mis dans train_model.py) """

import joblib
import pathlib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import mlflow

def evaluate_model(
        test_data: str,
        model_name: str,
        tracking_uri:str,

):
    """ Evaluate model """

    # Load model
    #mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    mlflow.set_tracking_uri(uri=tracking_uri)
    model_version = mlflow.search_model_versions(filter_string=f"name='{model_name}'")[-1].version
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Load test dataset 
    test_df = pd.read_csv(test_data, sep=',')

    # Extract labels and features
    x_test  = test_df.iloc[:,:-1]
    y_test = test_df.iloc[:, -1]

    # Evaluate model 
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_true= y_test, y_pred=y_pred)
    print(f"Validation accuracy: {accuracy * 100}")


if __name__ == "__main__":

    # Parseur pour que python puisse lire les arguments du bash
    import argparse
    parser = argparse.ArgumentParser(description="test data and model registry parser")
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--tracking_uri", type=str)


    args = parser.parse_args()


    # Appel de la fonction pour appel depuis bash
    evaluate_model(test_data=args.test_data, model_name=args.model_name, tracking_uri=args.tracking_uri)


# Pour executer sous bash
# python3 scripts/evaluate_model.py --test_data "data/processed/test.csv" --model_name "integration-autolog" --tracking_uri "http://127.0.0.1:8080"
