import numpy as np
import copy
from sklearn.decomposition import NMF, non_negative_factorization


class Layer:
    #Each layer consist of W and H, where H(k) = W(k+1) dot H(k+1)
    def __init__(self, X, n_components, max_iter2):
        self.model = NMF(n_components = n_components, init = 'random', random_state = 0, max_iter = max_iter2)
        self.W = self.model.fit_transform(X)
        self.H = self.model.components_


class Frame:
    def __init__(self, X, each_layer_n_components, max_iter2):
        self.W = []
        self.H = []
        Y = copy.deepcopy(X)
        #Sequential decomposition
        for i in range(len(each_layer_n_components)):
            layer = Layer(Y, each_layer_n_components[i], max_iter2)
            self.W.append(layer.W)
            self.H.append(layer.H)
            Y = layer.H

def DeepMF(X, each_layer_n_components = [4,3], max_iter1 = 1000, max_iter2 = 200):
    #Variables:
    #X: The input matrix for decomposition
    #each_layer_n_components: Arrary, indicating the number of components in each layer;
    #                         should be ordered in a decreasing way; recommended to be less than 3 layers
    #max_iter1: maximum iteration for deep MF
    #max_iter2: maximum iteration for BCD
    #Inital matrices W1...WL, H1...HL for all layers through sequential decomposition
    initial_Frame = Frame(X, each_layer_n_components, max_iter2)
    n_layers = len(each_layer_n_components)

    #Deep semi-NMF
    for k in range(max_iter1):
        #print(k)
        for i in range(n_layers):
            A = np.diag(np.full(X.shape[0],1))
            for j in range(i):
                A = np.dot(A,initial_Frame.W[j])
            if i == (n_layers-1):
                B = initial_Frame.H[i]
            else:
                B = np.dot(initial_Frame.W[i+1],initial_Frame.H[i+1])
            if i == 0:
                Y = X
            else:
                Y = initial_Frame.H[i-1]
            #update W:
            W, H, n_iter = non_negative_factorization(Y, n_components = each_layer_n_components[i], init='custom',
                                                      W = initial_Frame.W[i], update_H=False, H=B, max_iter =max_iter2)
            initial_Frame.W[i] = copy.deepcopy(W)
            #update H:
            temp_W = np.dot(A,W).T
            temp_H = initial_Frame.H[i].T
            W, H, n_iter = non_negative_factorization(X.T, n_components = each_layer_n_components[i], init='custom',
                                                      W = temp_H, update_H=False, H=temp_W, max_iter = max_iter2)
            initial_Frame.H[i] = copy.deepcopy(W.T)

    return initial_Frame



X = np.array([
[7, 8, 10, 8, 9, 4, 1, 2, 3, 2, 4, 1, 3, 5, 2],
[8, 8, 7, 8, 10, 5, 1, 2, 6, 1, 4, 2, 2, 1, 2],
[9, 9, 9, 9, 10, 4, 1, 4, 2, 1, 4, 3, 1, 1, 1],
[3, 1, 2, 1, 3, 8, 8, 7, 8, 10, 3, 4, 1, 2, 2],
[2, 1, 3, 2, 3, 9, 10, 9, 9, 8, 2, 2, 2, 2, 2],
[1, 2, 1, 1, 2, 8, 8, 9, 9, 8, 3, 2, 3, 3, 1],
[4, 1, 1, 2, 2, 2, 5, 1, 1, 5, 9, 7, 8, 9, 7],
[2, 2, 2, 1, 2, 2, 6, 2, 1, 3, 8, 8, 8, 8, 8],
[1, 2, 2, 1, 3, 2, 4, 1, 1, 4, 6, 9, 8, 10, 7]])



Result = DeepMF(X, each_layer_n_components = [4,3], max_iter1 = 1000, max_iter2 = 500)


Result2 = np.dot(Result.W[0],Result.H[0])
Result3 = np.dot(np.dot(Result.W[0],Result.W[1]),Result.H[1])
print(Result2)
print(np.sum((X - Result2)**2))
print(np.sum((X - Result3)**2))
