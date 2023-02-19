import time
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from typing import List
from collections import Counter
from scipy.spatial.distance import pdist, squareform

#The function calculates the range of radiuses to be checked
#We calculate the distances of the points in space to know how spread out they are
#and thus determine the radius ranges
def get_range_of_radius (X_train_scaled,Y_train):
    distance_matrix = squareform(pdist(X_train_scaled, metric='euclidean'))
    start_radius=distance_matrix.mean() - Y_train.nunique()
    end_radius=distance_matrix.mean()
    return start_radius,end_radius

#The function checks which feature fit to the point according to the radius
def find_nearest_neighbors_by_radius(radius,X_train,x_vld,Y_train) :
    # calculat's the euclidean distance with each point in the train and point from validation set
    distances = np.linalg.norm(X_train - x_vld, axis=1)
    indexes_of_points_in_radius = np.where(distances < radius)

    if len(indexes_of_points_in_radius[0])==0:#if there aren't instance’s neighbors in the radius
        increaced_radius=distances.min()+1#we will increace the radius
        indexes_of_points_in_radius = np.where(distances < increaced_radius)

    labels_of_neighbours_in_radius=Y_train[indexes_of_points_in_radius[0]]
    predicted_label=Counter(labels_of_neighbours_in_radius).most_common(1)[0][0]#get the most common class
    return predicted_label

#The function returns the optimal radius for the data set
def get_the_best_radius (X_val_scaled,X_train_scaled,Y_val,Y_train):
    start_radius, end_radius = get_range_of_radius(X_train_scaled, Y_train)
    best_radius = 0
    best_accuracy = 0
    for radius in np.linspace(start_radius, end_radius, 40):
        predictions_VLD=[]
        for row in range(X_val_scaled.shape[0]):
            predictions_VLD.append(find_nearest_neighbors_by_radius(radius,X_train_scaled,X_val_scaled[row],Y_train))
        curren_accuracy = accuracy_score(Y_val,predictions_VLD)
        if curren_accuracy > best_accuracy:
            best_accuracy = curren_accuracy
            best_radius = radius

    return best_radius

def classify_with_NNR(data_trn, data_vld, data_tst) -> List:
    print(f'starting classification with {data_trn}, {data_vld}, and {data_tst}')
    #frist we scale the the appropriate columns
    scaler=StandardScaler()
    df_train = pd.read_csv(data_trn)
    X_train=df_train.iloc[:,:-1]
    df_vld = pd.read_csv(data_vld)
    X_vld=df_vld.iloc[:,:-1]
    df_tst=pd.read_csv(data_tst)
    X_tst=df_tst.iloc[:,:-1]
    X_train_scaled=scaler.fit_transform(X_train)
    Y_train=df_train.iloc[: , -1]
    X_val_scaled=scaler.transform(X_vld)
    Y_val = df_vld.iloc[: , -1]
    X_tst_scaled=scaler.transform(X_tst)

    predictions = list()
    radius=get_the_best_radius(X_val_scaled,X_train_scaled,Y_val,Y_train)
    for point in range(X_tst_scaled.shape[0]):#here we get the predicted values for the test set
        predictions.append(find_nearest_neighbors_by_radius(radius,X_train_scaled,X_tst_scaled[point],Y_train))
    return predictions


if __name__ == '__main__':
    start = time.time()

    with open('config.json', 'r', encoding='utf8') as json_file:
        config = json.load(json_file)

    predicted = classify_with_NNR(config['data_file_train'],
                                  config['data_file_validation'],
                                  config['data_file_test'])

    df = pd.read_csv(config['data_file_test'])
    labels = df['class'].values

    if not predicted:  # empty prediction, should not happen in your implementation
        predicted = list(range(len(labels)))

    assert(len(labels) == len(predicted))  # make sure you predict label for all test instances
    print(f'test set classification accuracy: {accuracy_score(labels, predicted)}')

    print(f'total time: {round(time.time()-start, 0)} sec')
