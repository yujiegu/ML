import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k, distance_function):
        self.k = k
        self.distance_function = distance_function

    # save features and lable to self
    def train(self, features, labels):
        self.features = features
        self.labels= labels
        return features

    # predict labels of a list of points
    def predict(self, features):
        pointlabel = []
        for i in range(0,len(features)):
            label = self.get_k_neighbors(features[i])
            result = Counter(label)
            if result[0] > result[1]:
                pointlabel.append(0)
            else:
                pointlabel.append(1)
        return pointlabel

    # find KNN of one point
    def get_k_neighbors(self, point):
        dis=[]
        for i in range(0,len(self.features)):
            dis.append(self.distance_function(point,self.features[i]))
        sorted_dis_list = sorted(range(len(dis)), key=lambda x: dis[x])
        labelorder = [self.labels[i] for i in sorted_dis_list]
        labelorder = labelorder[0:self.k]
        return labelorder


