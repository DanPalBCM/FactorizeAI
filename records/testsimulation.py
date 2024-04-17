import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score, adjusted_rand_score, mean_squared_error, silhouette_score, fowlkes_mallows_score, calinski_harabasz_score
# clusterable dense matrix example
def preprocessing_standarization(X):

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    return X_scaled



def PCA_simulated(number_components = 10):
    """
    Naive simulation data of PCA. Staring with make_blobs from sklearn,
    we run PCA reconstruction inverse transformation, resulting in X_r, this is our
    matrix we will run AutoMF, then resultant matrix should be X_rr which is what should be evaluated.

    Supressing testing code...
    """
    X, clusters = make_blobs(n_samples = 2000,
                  n_features = 100,
                  centers = 2,
                  cluster_std = 0.4,
                  shuffle = True)

    X_scaled = preprocessing_standarization(X)

    model = PCA(n_components=number_components)

    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    # COMMENT >>>
    #W_r = model.fit_transform(X_r)

    #X_rr = model.inverse_transform(W_r)

    #mse = mean_squared_error(X_rr, X_r)

    #print(mse)
    #print(X_rr)
    #print(X_r)
    # <<<
    return X_r


def kawarstesting(X, **kwargs):
    model = PCA(svd_solver = 'auto', **kwargs)
    X = model.fit_transform(X)
    print(X)
    return

#PCA_simulated()

X = np.array([[-1.23,2.43],[-0.23, 3.46]])

kawarstesting(X, n_components = 2, random_state = 0)
#print((X <= 0).all())

#X = np.array([[0, 2.33],[4.3,23.4]])

#print((X >(X <= 0).all()(X <= 0).all()(X <= 0).all()(X <= 0).all()(X <= 0).all()(X <= 0).all()(X <= 0).all()= 0).all())
