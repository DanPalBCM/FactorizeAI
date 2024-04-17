from sklearn.decomposition import NMF, PCA, SparsePCA, KernelPCA, TruncatedSVD, IncrementalPCA, FastICA, \
    MiniBatchSparsePCA, MiniBatchNMF
from scipy.sparse import csr_matrix
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np


#########################
# For preprocessing we consider standarization
#########################

def preprocessing_standarization(X):
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    return X_scaled


#######################
# Matrix Factorization algorithms 
#######################

# NMF

def NMF_sklearn(X):
    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


# Sparse NMF
def NMF_sparse(X):
    X_sparse = csr_matrix(X)

    model = NMF(n_components=2, init='random', random_state=0)
    W = model.fit_transform(X_sparse)

    X_r = model.inverse_transform(W)
    return W, X_r


# PCA

def PCA_sklearn(X):
    model = PCA(n_components=2)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


# Sparse PCA

def Sparse_PCA(X):
    # X_sparse = csr_matrix(X)

    model = SparsePCA(n_components=2)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


# Kernel PCA

def Kernel_PCA(X):
    model = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform=True)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


# Truncated SVD

def Truncated_SVD(X):
    model = TruncatedSVD(n_components=2, algorithm='randomized')
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


# IncrementalPCA

def Incremental_PCA(X):
    # Used when dataset is too large to use PCA
    model = IncrementalPCA(n_components=2, batch_size=10)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


# IndependentCA

def IndependentCA(X):
    model = FastICA(n_components=2)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


def MiniBatch_SparsePCA(X):
    model = MiniBatchSparsePCA(n_components=2)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


def MiniBatch_NMF(X):
    model = MiniBatchNMF(n_components=2)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs, make_sparse_spd_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import rand_score, adjusted_rand_score, mean_squared_error, silhouette_score, \
    fowlkes_mallows_score, calinski_harabasz_score
import warnings

warnings.filterwarnings("ignore")


from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering
from sklearn.datasets import make_moons, make_blobs
from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

#################
# Consensus Ensamble clustering classifier K-means + spectral 
#################

from scipy.spatial.distance import cdist
import numpy as np

class ClusterSimilarityMatrix():
    
    def __init__(self) -> None:
        self._is_fitted = False

    def fit(self, y_clusters):
        if not self._is_fitted:
            self._is_fitted = True
            self.similarity = self.to_binary_matrix(y_clusters)
            return self

        self.similarity += self.to_binary_matrix(y_clusters)

    def to_binary_matrix(self, y_clusters):
        y_reshaped = np.expand_dims(y_clusters, axis=-1)
        return (cdist(y_reshaped, y_reshaped, 'cityblock')==0).astype(int)


class EnsembleCustering():
    def __init__(self, base_estimators, aggregator, distances=False):
        self.base_estimators = base_estimators
        self.aggregator = aggregator
        self.distances = distances

    def fit(self, X):
        X_ = X.copy()

        clt_sim_matrix = ClusterSimilarityMatrix()
        for model in self.base_estimators:
            clt_sim_matrix.fit(model.fit_predict(X=X_))
        
        sim_matrix = clt_sim_matrix.similarity
        self.cluster_matrix = sim_matrix/sim_matrix.diagonal()

        if self.distances:
            self.cluster_matrix = np.abs(np.log(self.cluster_matrix + 1e-8)) # Avoid log(0)

    def fit_predict(self, X):
        self.fit(X)
        y = self.aggregator.fit_predict(self.cluster_matrix)
        return y

def Ensemble_classifier_sklearn(X, number_clusters):
    '''
    Consensus Ensamble clustering classifier, takes matrix and number of clusters and returns labels predictions
    
    based on: 
    https://github.com/jaumpedro214/posts/blob/main/ensamble_clustering/ensamble_clustering.ipynb
    
    '''
    NUM_KMEANS = 6

    clustering_models = NUM_KMEANS * [
        # Note: Do not set a random_state, as the variability is crucial
        MiniBatchKMeans(n_clusters= number_clusters, batch_size=64, n_init=1, max_iter=20)
    ]
    aggregator_clt = SpectralClustering(n_clusters= number_clusters, affinity="precomputed")
    
    ens_clt = EnsembleCustering(clustering_models, aggregator_clt)
    y_ensemble = ens_clt.fit_predict(X)

    return y_ensemble


###############
# Subfunctions Consensus MF
###############

def Supervised_Consensus(MFiters, method, X_scaled, clusters):
    '''
    Takes standarized matrix X_scaled, and number of iterations of MF MFiters.
    '''
    
    adjusted_rs = []
    
    fowlkes = []

# Consensus MF
    for i in range(MFiters):

        if method == "PCA_sklearn":
            X, X_r = PCA_sklearn(X_scaled)

        if method == "NMF_sklearn":
            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = NMF_sklearn(X_scaled)

        if method == "NMF_sparse":
            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = NMF_sparse(X_scaled)

        if method == "MiniBatch_NMF":
            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = MiniBatch_NMF(X_scaled)

        if method == "Sparse_PCA":
            X, X_r = Sparse_PCA(X_scaled)

        if method == "Kernel_PCA":
            X, X_r = Kernel_PCA(X_scaled)

        if method == "Truncated_SVD":
            X, X_r = Truncated_SVD(X_scaled)

        if method == "Incremental_PCA":
            X, X_r = Incremental_PCA(X_scaled)

        if method == "IndependentCA":
            X, X_r = IndependentCA(X_scaled)

        if method == "MiniBatch_SparsePCA":
            X, X_r = MiniBatch_SparsePCA(X_scaled)
        
        labels = Ensemble_classifier_sklearn(X, 2)
    # accuracy evaluation of consensus clustering 

        acc = adjusted_rand_score(clusters, labels)
    
        adjusted_rs.append(acc)

        acc2 = fowlkes_mallows_score(clusters, labels)
        
        fowlkes.append(acc2)

    adjusted_rs = np.mean(adjusted_rs)
    fowlkes = np.mean(fowlkes)
    print(method + "evaluation metrics: ")
    print("Adjusted Rand Score...")
    print(adjusted_rs)
    print("Fowlkes Mallows Score...")
    print(fowlkes)

    

    return

#
def Unsupervised_Consensus(MFiters, method, X_scaled):
    '''
    Takes standarized matrix X_scaled, and number of iterations of NMF MFiters.
    '''

    silhouette = []

    calinski = []
# Consensus MF
    for i in range(MFiters):

        if method == "PCA_sklearn":
            X, X_r = PCA_sklearn(X_scaled)

        if method == "NMF_sklearn":
            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = NMF_sklearn(X_scaled)

        if method == "NMF_sparse":
            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = NMF_sparse(X_scaled)

        if method == "MiniBatch_NMF":
            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = MiniBatch_NMF(X_scaled)

        if method == "Sparse_PCA":
            X, X_r = Sparse_PCA(X_scaled)

        if method == "Kernel_PCA":
            X, X_r = Kernel_PCA(X_scaled)

        if method == "Truncated_SVD":
            X, X_r = Truncated_SVD(X_scaled)

        if method == "Incremental_PCA":
            X, X_r = Incremental_PCA(X_scaled)

        if method == "IndependentCA":
            X, X_r = IndependentCA(X_scaled)

        if method == "MiniBatch_SparsePCA":
            X, X_r = MiniBatch_SparsePCA(X_scaled)
        
        labels = Ensemble_classifier_sklearn(X, 2)

    # accuracy evaluation of consensus clustering 

        acc = silhouette_score(X, labels)
    
        silhouette.append(acc)
        
        acc2 = calinski_harabasz_score(X, labels)
    # error 
        calinski.append(acc2)


    silhouette = np.mean(silhouette)
    calinski = np.mean(calinski)
    
    print(method + "evaluation metrics: ")

    print("Silhouette Score...")
    print(silhouette)
    print("Calinski Harabasz Score...")
    print(calinski)
    
    return

###########
# Main pipeline 
###########
def Initial_Guess(X_scaled, method_list, clusters, Supervised):
    '''
    Takes standarized/normalizaed matrix X_scaled, number of iterations MFiters, and
    list of methods to scan for optimal initial guess based on inverse transformation reconstruction
    mean squared error.
    '''

    # Number of iterations for Matrix Factorization
    MFiters = 3

    for method in method_list:
        if Supervised:
            Supervised_Consensus(MFiters, method, X_scaled, clusters)
        else:
            Unsupervised_Consensus(MFiters, method, X_scaled)

    return


# Sparsity check

def sparsity_check(X):
    """
    Calculates sparsity, if sparsity > 65% returns True
    """
    sparsity = 1.0 - (np.count_nonzero(X) / float(X.size))

    if sparsity > .65:
        sparse = True
    else:
        sparse = False

    return sparse


def Model_auto_select(X, clusters, Supervised):
    """
    Selects MF method based on best initial guess (testing all methods with default parameters)
    by inputting original matrix.
    """
    # Sparsity check
    # if sparse use Sparsity methods, else use normal methods

    if sparsity_check(X) == True:

        method_list = ['NMF_sparse', 'Sparse_PCA', "MiniBatch_SparsePCA"]
        # Normalize instead of standarization
        X_scaled = normalize(X, norm='l1', axis=1)

        Initial_Guess(X_scaled, method_list, clusters, Supervised)
    else:

        method_list = ["PCA_sklearn", "NMF_sklearn", "Kernel_PCA",
                       "Truncated_SVD", "Incremental_PCA", "IndependentCA", "MiniBatch_NMF"]
        # method_list = ["PCA_sklearn", "NMF_sklearn"]
        X_scaled = preprocessing_standarization(X)

        Initial_Guess(X_scaled, method_list, clusters, Supervised)
    return


# Main test
def maintest():
    """
    working test
    """
    # clusterable dense matrix example
    X, clusters = make_blobs(n_samples=2000,
                             n_features=10,
                             centers=2,
                             cluster_std=0.4,
                             shuffle=True)

    Model_auto_select(X, clusters, Supervised = True)

    Model_auto_select(X, clusters, Supervised = False)

    return


maintest()
