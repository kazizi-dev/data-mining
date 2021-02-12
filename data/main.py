import pandas as pd
import numpy as np

class Node():
    value = ""
    children = []
    
    def __init__(self, v, d):
        self.value = v
        if(isinstance(d, dict)):
            self.children = d.keys()

# apply preprocessing techniques
def get_clean_data():
    df = pd.read_csv('adult.data.csv')
    print("===============================================")

    # replace "?" symbols by "none"
    for col in df.columns:
        if type(df[col].iloc[0]) == str:
            df.loc[df[col].str.contains('?', regex=False)] = 'none'

    # drop any row containing the word "none"
    df = df[(df != 'none').all(axis=1)]
    
    # rename columns
    df.rename(
        columns={
            'marital-status': 'marital_status',
            'education-num': 'education_num',
            'capital-gain': 'capital_gain', 
            'capital-loss': 'capital_loss', 
            'native-country': 'country',
            'hours-per-week': 'hours_per_week'
        }, 
        inplace=True
    )
    
    # encode income data with numerics
    df['income'] = df['income'].map({'<=50K': 0, '>50K': 1}).astype(int)
    df['marital_status'] = df['marital_status'].astype(str)
    df['marital_status'] = df['marital_status'].replace(
        [' Never-married',' Divorced',' Separated',' Widowed'], 'Single')
    df['marital_status'] = df['marital_status'].replace([
        ' Married-civ-spouse',' Married-spouse-absent',' Married-AF-spouse'], 
        'Married')
    df['marital_status'] = df['marital_status'].map(
        {'Married' : 1, 'Single' : 0}).astype(int)
    
    
    # cast types from object to int
    df['age'] = df['age'].astype(int)
    # df['fnlwgt'] = df['fnlwgt'].astype(int)
    df['education_num'] = df['education_num'].astype(int)
    df['capital_gain'] = df['capital_gain'].astype(int)
    df['capital_loss'] = df['capital_loss'].astype(int)
    df['hours_per_week'] = df['hours_per_week'].astype(int)
    df['marital_status'] = df['marital_status'].astype(int)
    
    df.drop('fnlwgt', axis =1, inplace = True)

    return df


def get_entropy(target_col_data):
    _, counts = np.unique(target_col_data, return_counts = True)
    probabilities = counts / counts.sum()
    entropy = np.sum(probabilities * np.log2(probabilities))
    return -entropy


def get_info_gain(df, split_attr, target_attr='income'):    
    entropy_t = get_entropy(df[target_attr])
    vals, counts = np.unique(df[split_attr], return_counts=True)
    
    entropy_t1 = 0.0
    for i in range(len(vals)):
        target_data = df.where(df[split_attr] == vals[i]).dropna()[target_attr]
        entropy_t1 = np.sum((counts[i]/np.sum(counts)) * get_entropy(target_data))
    
    info_gain = entropy_t - entropy_t1
    print(split_attr, ' and ', info_gain)    
    return info_gain

        
# choose the best attribute which has max info gain
def get_best_attr(df, attributes, target_attr):
    best_attr = attributes[0]
    max_info_gain = 0
    
    for attr in attributes:
        res = get_info_gain(df, attr, target_attr)
        if res > max_info_gain:
            max_info_gain = res
            best_attr = attr
    
    return best_attr

def grow():
    pass

def prune():
    pass

def test():
    pass

# build and train the decision tree 
def learn(data, attributes, target):
    grow(data, attributes, target)
    
if __name__ == '__main__':
    #categorical_attr, numerical_attr = get_attributes(df)
    df = get_clean_data()
    print(">> 1. Preprocessing: complete")
    
    attributes = df.columns
    print('Best attribute: ', get_best_attr(df, attributes, attributes[-1]))
    print(">> 2. Calculate Entropy: complete")
    
    
    
    
# def get_attributes(df):
#     categorical_attr = []
#     numerical_attr = []
    
#     for col in df.columns:
#         if df[col].dtype.name == 'object':
#             categorical_attr.append(col)
#         else:
#             numerical_attr.append(col)
#     return categorical_attr, numerical_attr


# def get_unique_vals(df, col):
#     data = df.values
#     return set(row[col] for row in data)


# def get_entropy(df, target_attr='income'):
#     elements,counts = np.unique(target_attr, return_counts = True)  
    
#     entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])  
    
#     return entropy
