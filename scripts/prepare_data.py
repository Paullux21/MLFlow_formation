""" Data Preparation Module """

import pandas as pd
import numpy as np

def prepare_data(
        input_data: str,
        output_folder: str
) -> None:
    """
    Apply preparation process on inputdata and save output in out folder

    Args:
        input_data (str): _description_
        output_folder (str): _description_
    """

    iris_df = pd.read_csv(input_data, sep=',')

    # cleaning example
    # ...

    train, validate, test = np.split(iris_df.sample(frac=1, random_state=42),
                                    [
                                    int(0.6*len(iris_df)), 
                                    int(0.8*len(iris_df))
                                    ])

    # Saving
    # train.to_csv(f'{output_folder}/train.csv')
    # validate.to_csv(f'{output_folder}/validate.csv')
    # test.to_csv(f'{output_folder}/test.csv')

    # Saving with pathlib for Windows compatibility 
    import pathlib
    train.to_csv(pathlib.Path(output_folder).joinpath('train.csv'))
    validate.to_csv(pathlib.Path(output_folder).joinpath('validate.csv'))
    test.to_csv(pathlib.Path(output_folder).joinpath('test.csv'))


if __name__ == "__main__":

    # Parseur pour que python puisse lire les arguments du bash
    import argparse
    parser = argparse.ArgumentParser(description="input data parser")
    parser.add_argument("--input_data", type=str)
    parser.add_argument("--output_folder", type=str)

    args = parser.parse_args()


    # Appel de la fonction pour appel depuis bash
    prepare_data(input_data=args.input_data, output_folder=args.output_folder)



# Pour executer sous bash
# python3 scripts/prepare_data.py --input_data "data/raw/iris.csv" --output_folder "data/processed"




