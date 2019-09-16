import numpy as np
from knn import KNN

# implement F1 score
def f1_score(real_labels, predicted_labels):
    TP=0
    FP=0
    FN=0
    TN=0
    for i in range(0,len(real_labels)):
        if real_labels[i] == 1:
            if predicted_labels[i] == 1:
                TP+=1
            else:
                FN+=1
        else:
            if predicted_labels[i] == 1:
                FP+=1
            else:
                TN+=1
    if(TP+FP==0)or (TN+FN==0):
        F1 = 0
    else:
        P = TP/(TP+FP)
        R = TP/(TP+FN)
        if P+R==0:
            F1 = 0
        else:
            F1 = 2*P*R/(P+R)
    return F1

#implement distance functions
class Distances:
    def canberra_distance(point1, point2):
        canberraDistance = 0
        for i in range(0,len(point1)):
            if (point1[i] == 0)&(point2[i]==0):
                canberraDistance = canberraDistance
            else:
                canberraDistance+=abs(point1[i]-point2[i])/(abs(point1[i])+abs(point2[i]))
        return canberraDistance

    # choose p =3
    def minkowski_distance(point1, point2):
        minkowskisum =0
        for i in range(0,len(point1)):
            minkowskisum += abs(point1[i]-point2[i])**3
        minkowskiDistance = minkowskisum ** (1/3)
        return minkowskiDistance

    def euclidean_distance(point1, point2):
        euclideansum = 0
        for i in range(0,len(point1)):
            euclideansum += (point1[i]-point2[i])**2
        euclideanDistance = euclideansum ** (1/2)
        return euclideanDistance

    def inner_product_distance(point1, point2):
        return np.dot(point1,point2)

    def cosine_similarity_distance(point1, point2):
        A = 0
        B = 0
        for i in range(0, len(point1)):
            A += (point1[i] ** 2)
            B += (point2[i] ** 2)
        C = np.dot(point1, point2)
        if (A==0) or (B==0):
            cosDistance=1
        else:
            cosDistance = 1-C / ((A ** 0.5) * (B ** 0.5))
        return cosDistance

    def gaussian_kernel_distance(point1, point2):
        square=0
        for i in range(0,len(point1)):
            square += (point1[i]-point2[i])**2
        square = -0.5*square
        gaussianDistance = -np.exp(square)
        return gaussianDistance

class HyperparameterTuner:
    def __init__(self):
        self.best_k = None
        self.best_distance_function = None
        self.best_scaler = None
        self.best_model = None

    # find parameters with the best f1 score on validation dataset
    # try different distance function
    # find the best k, increase by 2
    # f1-score to compare different models
    # check distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
    # same distance fuction, choose model which has a less k
    def tuning_without_scaling(self, distance_funcs, x_train, y_train, x_val, y_val):
        f1 = -1
        for i in distance_funcs:
            for k in range(1,30,2):
                a = KNN(k, distance_funcs[i])
                a.train(x_train, y_train)
                pointlabel = a.predict(x_val)
                f1current = f1_score(y_val,pointlabel)
                if (f1current>f1):
                    f1 = f1current
                    best_k = k
                    best_distance_function = i
        best_model = KNN(best_k,distance_funcs[best_distance_function])
        best_model.train(x_train, y_train)


        #assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_distance_function
        self.best_model = best_model
        return best_k,best_distance_function,best_model

    # find parameters with the best f1 score on validation dataset, with normalized data
    # choose model based on the following priorities:
    # normalization, [min_max_scale > normalize];
    # distance function  [canberra > minkowski > euclidean > gaussian > inner_prod > cosine_dist]
    # same distance function, choose model which has a less k
    def tuning_with_scaling(self, distance_funcs, scaling_classes, x_train, y_train, x_val, y_val):
        f1 = -1
        for j in scaling_classes:
            scaler1 = scaling_classes[j]()
            x_normal = scaler1(x_train)
            x_val_normal = scaler1(x_val)
            for i in distance_funcs:
                for k in range(1, 30, 2):
                    a = KNN(k, distance_funcs[i])
                    a.train(x_normal, y_train)
                    pointlabel = a.predict(x_val_normal)
                    f1current = f1_score(y_val, pointlabel)
                    if (f1current > f1):
                        f1 = f1current
                        best_k = k
                        best_distance_function = i
                        best_scaler = j
        best_model = KNN(best_k,distance_funcs[best_distance_function])
        best_model.train(x_train, y_train)


        # assign the final values to these variables
        self.best_k = best_k
        self.best_distance_function = best_distance_function
        self.best_scaler = best_scaler
        self.best_model = best_model
        return best_k,best_distance_function,best_scaler,best_model


class NormalizationScaler:
    def __init__(self):
        pass

    # normalize data
    def __call__(self, features):

        normalization = []
        for i in features:
            if (np.linalg.norm(i) == 0):
                normalization.append(i)
            else:
                a = list(i / np.linalg.norm(i))
                normalization.append(a)
        return normalization


class MinMaxScaler:
    def __init__(self):
        self.time=0
        pass

    def __call__(self, features):
        # find min and max in the training case
        # then apply to the validation and test set
        self.time += 1
        if self.time == 1:
            self.min = [float('inf')] * len(features[0])
            self.max = [float('-inf')] * len(features[0])
            for i in range(0, len(features[0])):
                for j in range(0, len(features)):
                    if features[j][i] < self.min[i]:
                        self.min[i] = features[j][i]
                    if features[j][i] > self.max[i]:
                        self.max[i] = features[j][i]

        minmax = [([float('inf')] * len(features[0])) for i in range(len(features))]
        for i in range(0, len(features)):
            for j in range(0, len(features[0])):
                if self.min[j] == self.max[j]:
                    if self.min[j] == 0:
                        minmax[i][j] = 0
                    elif self.min[j] < 1:
                        minmax[i][j] = self.min[j]
                    else:
                        minmax[i][j] = 1
                else:
                    minmax[i][j] = (features[i][j] - self.min[j]) / (self.max[j] - self.min[j])
        minmax = np.array(minmax)

        return minmax
