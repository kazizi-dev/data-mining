import pandas as pd
import numpy as np
import math
import warnings
from sklearn.neighbors import NearestNeighbors
from matplotlib import pyplot as plt
import seaborn as sns

##########################
### global variables:
UNASSIGNED_POINT = 0
NOISE_POINT = -1
##########################

############################################################################
#### Instruction: comment out main function below and simply run this test
####              function in order to get a plot for testing purposes.
############################################################################
def plot_epsilon_graph():
    """
    plot an epsilon graph 
    @return: list of cluster labels
    """
    warnings.filterwarnings("ignore")

    df = pd.read_csv('houshold2007.csv')
    df = df[df['Date'].apply(lambda x: x.endswith('1/2007'))]

    preprocessing(df)

    attributes = df.columns.tolist()
    data = df.to_numpy()
    nbrs = NearestNeighbors(n_neighbors=get_min_pts(attributes), 
                            algorithm='auto').fit(data)
    distances, indices = nbrs.kneighbors(data)

    temp = []
    for i in range(len(distances)):    
        temp.append(distances[i][len(distances[i])-1])
    temp.sort(reverse=True)

    plt.plot(temp)    
    plt.grid(True)
    plt.xaxis('test')
    plt.title('Epsilon Threshold')
    plt.savefig('eps_plot.png')
    plt.show()



def preprocessing(df):
    """
    preprocess data and remove unnecessary columns, remove 
        null and undefined values, and normalize the dataset.
    @param df: dataset as dataframe pandas
    """

    # drop unnecessary cols/attributes
    attributes = df.columns.tolist()
    if 'Date' in attributes:
        df.drop('Date', axis=1, inplace=True)
    if 'Time' in attributes:
        df.drop('Time', axis=1, inplace=True)

    # impute missing values for attributes with type string
    attributes = df.columns.tolist()
    attributes.remove('Sub_metering_3')
    for col in attributes:
        copy = df[df[col] != '?']
        copy = copy[[col]].apply(pd.to_numeric)
        copy[col] = copy[col].replace('?', copy[col].mean())
        df[col] = copy[col]

    # impute missing values for attributes including np.nan
    for col in df.columns.tolist():
        copy = df[df[col] != np.nan]
        copy = copy[[col]].apply(pd.to_numeric)
        copy[col] = copy[col].replace(np.nan, copy[col].mean())
        df[col] = copy[col]

    # contribute string to numeric, and normalize the dataset
    for attribute in df.columns.tolist():
        df[[attribute]] = df[[attribute]].apply(pd.to_numeric)
        df[attribute] = ((df[attribute]-df[attribute].mean())/df[attribute].std(ddof=0))


def get_min_pts(attributes):
    """
    calculate the minimum points based on the dimensions of the dataset
    @return: minimum points
    """
    # the dimension in the dataset is the number of attributes
    k = (2*len(attributes))-1
    return k+1


def get_distance(a, b):
    """
    calculate the distance between two vectors a and b.
    @return: distance between and b
    """
    return np.sqrt(np.sum(np.square(a - b)))


def get_neighbour_points(data, eps, target_point):
    """
    get the index of points that are neighbours.
    @return: list of object indexes 
    """
    neighbours = []
    for neighbour_point in range(len(data)):
        # add neighbours if the points have a distance below the eps threshold
        if get_distance(data[target_point], data[neighbour_point] < eps):
           neighbours.append(neighbour_point)
            
    return neighbours


def fit(data, eps, min_pts):
    """
    label the dataset points such that points are 0 by default and -1 as noise.
    @param data: numpy.ndarray
    @param cluster_labels: cluster labels
    @param eps: epsilon value (none by default)
    @param min_pts: minimum points (none by default)
    @return: list of cluster labels
    """
    import queue
    core_points = []
    cluster_labels = []
    cluster_num = 1

    # if empty, fill up cluster labels with unassigned points
    if len(cluster_labels) == 0:
        cluster_labels = [UNASSIGNED_POINT] * len(data)
    

    for point in range(len(data)):      
        if cluster_labels[point] == UNASSIGNED_POINT:
            # get the neighbour points of the current point
            neighbour_points = get_neighbour_points(data, eps, point)

            # if there is not enough points around, it is noise
            if len(neighbour_points) < min_pts:
                cluster_labels[point] = NOISE_POINT
                cluster_num += 1
                continue
            else:
                # if there is enough points around then it is core point
                cluster_labels[point] = cluster_num
                
            arr = []
            
            # place neighbour points in the same cluster as the core point
            for point in neighbour_points:
                if cluster_labels[point] == UNASSIGNED_POINT:
                    arr.append(point)
                    cluster_labels[point] = cluster_num
            
            # keep going until every element is labeled
            while len(arr) != 0:

                # for each neighbour point, find their new neighbours again
                neighbour_point = arr.pop()
                new_neighbour_points = get_neighbour_points(data, eps, 
                                                            neighbour_point)

                # the neighbour point is also a core point
                if len(neighbour_points) >= min_pts:
                    core_points.append(neighbour_point)

                    # assign points to a cluster
                    for i in range(len(neighbour_points)):
                        if cluster_labels[neighbour_points[i]] == UNASSIGNED_POINT:
                            arr.append(neighbour_points[i])
                            cluster_labels[neighbour_points[i]] = cluster_num

            cluster_num += 1
                            

    # return a tuple containing labels and total clusters
    return (cluster_labels, cluster_num)


def dbscan(data, eps, min_pts):
    """
    execute DBSCAN algorithm on the dataset.
    @param df: dataset as a list of numpy arrays
    @param eps: epsilon as the threshold
    @param min_pts: required number of points in a cluster
    """

    cluster_labels = []
    
    result = fit(data, eps, min_pts)
    cluster_labels = result[0]
    total_clusters = result[1]

    return cluster_labels, total_clusters


def plotRes(data, cluster_labels, cluster_num):
    scatterColors = ['olive', 'darkred', 'purple', 'orange', 'brown', 'black', 'blue']
    for i in range(cluster_num):
        color = scatterColors[i % len(scatterColors)]
        x_coordinates = []  
        y_coordinates = []
        for j in range(len(data)):
            if cluster_labels[j] == i:
                x_coordinates.append(data[j, 0])
                y_coordinates.append(data[j, 1])
        plt.scatter(x_coordinates, y_coordinates, c=color, alpha=1, marker='.')


if __name__ == "__main__":
    # ignore dtypes warning during csv reading
    warnings.filterwarnings("ignore")

    df = pd.read_csv('houshold2007.csv')
    df = df[df['Date'].apply(lambda x: x.endswith('1/2007'))][0:500]
    
    # clean up data
    preprocessing(df)

    attributes = df.columns.tolist()
    data = df.to_numpy()
    min_pts = get_min_pts(attributes)

    # based on the graph shown in part c of the report
    eps = 0.6       

    # get results for cluster labels
    cluster_labels, total_clusters = dbscan(data, eps, min_pts)
    cluster_labels = np.array(cluster_labels)

    # save cluster labels in a new column and save result in a new csv file
    df['Cluster labels'] = cluster_labels

    print('Total clusters: ', total_clusters)

    df.to_csv('output.csv', index=False)