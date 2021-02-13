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
    elements, counts = np.unique(df[split_attr], return_counts=True)
    
    entropy_t1 = 0.0
    for i in range(len(elements)):
        target_data = df.where(df[split_attr] == elements[i]).dropna()[target_attr]
        entropy_t1 = np.sum((counts[i]/np.sum(counts)) * get_entropy(target_data))
    
    info_gain = entropy_t - entropy_t1
    # print(split_attr, ' and ', info_gain)    
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


def get_data_for_val(df, best_attr, val):
    return df.where(df[best_attr] == val).dropna()
    

# build and grow the tree 
def grow(df, attributes, target_attr):
    
    # get multi-dimensional array
    # data = df.reset_index().values
        
    # get info for the best attribute
    best_attr = get_best_attr(df, attributes, target_attr)
    print("best attribute is: ", best_attr)
    # best_attr = 'workclass'
    
    
    # best_attr_idx = attributes.get_loc(best_attr)
    best_attr_idx = 0
    for attr in attributes:
        if attr == best_attr:
            break
        else:
            best_attr_idx += 1
            
    unique_vals = (df[best_attr].unique()).tolist()
    
    if len(df) == 0:
        return np.unique(df[target_attr])[np.argmax(np.unique(df[target_attr], return_counts=True)[1])]
    if (len(attributes) - 1) <= 0:
        return None
    elif unique_vals.count(unique_vals[0]) == len(unique_vals):
        return unique_vals[0]
    else:
        # remove attribute from future options
        for val in unique_vals:
            # select best attribute rows with unique value
            new_df = get_data_for_val(df, best_attr, val)
            # remove attribute from future options
            new_attr = attributes.delete(best_attr_idx)
            
            # recursively build the subtree
            subtree = grow(new_df, new_attr, target_attr)
            tree = {best_attr:{}}
            tree[best_attr][val] = subtree
            # print(tree[best_attr][val])
    return tree

def prune():
    pass


def test():
    pass

# build and train the decision tree 
def learn(df, attributes, target_attr):
    grow(df, attributes, target_attr)
    
if __name__ == '__main__':
    #categorical_attr, numerical_attr = get_attributes(df)
    df = get_clean_data()
    print(">> 1. Preprocessing: complete")
    
    # print('Best attribute: ', get_best_attr(df, attributes, attributes[-1]))
    # print(">> 2. Get Best Attribute: complete")
    
    attributes = df.columns
    tree = {}
    tree = learn(df, attributes, 'income')
    
    print(tree)
    
    
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
