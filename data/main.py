import pandas as pd
import numpy as np

def get_clean_data():
    df = pd.read_csv('adult.data.csv')
    print("===============================================")

    # replace "?" symbols by "none"
    for col in df.columns:
        if type(df[col].iloc[0]) == str:
            print("here", col)
            df.loc[df[col].str.contains('?', regex=False)] = 'none'

    # drop any row containing the word "none"
    df = df[(df != 'none').all(axis=1)]
    return df
    
if __name__ == '__main__':
    df = get_clean_data()
    # df.to_csv('output_test.csv', index=False, header=True)

    # data = df.values
    # features = list(df.columns)
    # print(features)