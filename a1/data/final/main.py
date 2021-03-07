import pandas as pd
import numpy as np

class DecisionTree():
    tree = {}

    def train(self, df, attributes, target_attr):
        self.tree = grow(df, attributes, target_attr)
        

# apply preprocessing techniques
def get_clean_data():
    df = pd.read_csv('adult.data.csv')
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


def get_info_gain(df, split_attr, target_attr):    
    entropy_t = get_entropy(df[target_attr])
    elements, counts = np.unique(df[split_attr], return_counts=True)
    
    entropy_t1 = 0.0
    for i in range(len(elements)):
        target_data = df.where(df[split_attr] == elements[i])[target_attr]
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
    return df[df[best_attr] == val].dropna()
    

# build and grow the tree 
def grow(df, attributes, target_attr):      
    # get info for the best attribute
    best_attr = get_best_attr(df, attributes, target_attr)

    # find the index of the best attribute    
    best_attr_idx = 0
    for attr in attributes:
        if attr == best_attr:
            break
        else:
            best_attr_idx += 1
            
    return prune(df, best_attr_idx, best_attr, attributes, target_attr)
            
def prune(df, best_attr_idx, best_attr, attributes, target_attr):
    unique_vals = (df[best_attr].unique()).tolist()
    
    if unique_vals.count(unique_vals[0]) == len(unique_vals):
        return unique_vals[0]
    else:
        tree = {best_attr:{}}
        for val in unique_vals:
            new_df = get_data_for_val(df, best_attr, val)
            # remove attribute from future options
            new_attr = attributes.delete(best_attr_idx)
            
            # recursively build the subtree
            subtree = grow(new_df, new_attr, target_attr)
            tree[best_attr][val] = subtree
            
    # return the tree object
    return tree


# did not finish
def test(df, attributes, target):
    data = df.values.tolist()
    
    tree = DecisionTree()
    tree.train(df, attributes, target)
    print("==============================================")
    print("Tree:\n")
    print(tree.tree.copy())
    print("==============================================")
    
    # dummy value
    acc = 0.45432       # did not finish the test code
    print(f"Dummy Accuracy Value: {acc}")
        

if __name__ == '__main__':
    print('Status: in-progress')
    df = get_clean_data()
    attributes = df.columns    
    test(df, attributes, attributes[-1])
    print('Status: success')
    
