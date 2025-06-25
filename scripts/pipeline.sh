#!/bin/bash

# Option pour les erreurs: "Exit immediatly if a command exists with a non-zero status"
set -e

# Setting variables 
input_data=${input_data:="data/raw/iris.csv"}
output_folder=${output_folder:="data/processed"}
training_data=${training_data:="data/processed/train.csv"}
tracking_uri=${tracking_uri:=http://10.140.107.51:8080}
model_name=${model_name:=iris_model}
test_data=${test_data:="data/processed/test.csv"}


# Demande des arguments 
if [ "$#" -lt 1 ] ; then 
    echo "Need input data and output folder"
    exit 1
fi

# Step 1: Prepare data 
echo "Preparing data..."
python3 scripts/prepare_data.py --input_data $input_data --output_folder $output_folder

# Step 2: Train model
python3 scripts/train_model.py --training_data $training_data --model_name $model_name --tracking_uri $tracking_uri

# Step 3: Evaluate model
python3 scripts/evaluate_model.py --test_data $test_data --model_name $model_name --tracking_uri $tracking_uri
