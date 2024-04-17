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

def NMF_sklearn(X, **kwargs):

    model = NMF(init='random', **kwargs)

    W = model.fit_transform(X)
    
    X_r = model.inverse_transform(W)
    
    return W, X_r


# PCA

def PCA_sklearn(X, **kwargs):
    model = PCA(**kwargs)
    W = model.fit_transform(X)
    
    X_r = model.inverse_transform(W)

    return W, X_r

# Sparse PCA

def Sparse_PCA(X, **kwargs):
    #X_sparse = csr_matrix(X)

    model = SparsePCA(**kwargs)
    W = model.fit_transform(X)
    
    X_r = model.inverse_transform(W)
    
    return W, X_r

# Kernel PCA

def Kernel_PCA(X, **kwargs):

    model = KernelPCA(kernel='linear', fit_inverse_transform = True, **kwargs)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r
# Truncated SVD

def Truncated_SVD(X, **kwargs):
    model = TruncatedSVD(algorithm='randomized', **kwargs)
    W = model.fit_transform(X)
    
    X_r = model.inverse_transform(W)
    
    return W, X_r

# IncrementalPCA

def Incremental_PCA(X):
    # Used when dataset is too large to use PCA
    model = IncrementalPCA(n_components=2, batch_size = 10)
    W = model.fit_transform(X)
    
    X_r = model.inverse_transform(W)

    return W, X_r


# IndependentCA

def IndependentCA(X, **kwargs):
    model = FastICA(**kwargs)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r


def MiniBatch_SparsePCA(X, **kwargs):
    model = MiniBatchSparsePCA(**kwargs)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r

def MiniBatch_NMF(X, **kwargs):
    model = MiniBatchNMF(init = 'random', **kwargs)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r

################################
# Manifold and non-invertible methods
################################

from sklearn.decomposition import FactorAnalysis, DictionaryLearning, LatentDirichletAllocation
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding, MDS, SpectralEmbedding
def Factor_Analysis(X, **kwargs):
    transformer = FactorAnalysis(**kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat


def Dictionary(X, **kwargs):
    transformer = DictionaryLearning(max_iter=100, transform_algorithm = 'lasso_lars', transform_alpha = 0.1, **kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat

def LDA(X, **kwargs):
    transformer = LatentDirichletAllocation(**kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat

def t_SNE(X, **kwargs):
    transformer = TSNE(learning_rate = 'auto', init = 'random', perplexity = 3, **kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat


def Iso_map(X):
    transformer = Isomap(n_components = 2)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat

def Local_Linear_Embedding(X, **kwargs):
    transformer = LocallyLinearEmbedding(eigen_solver= 'auto', **kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat

def Multidimensional_scaling(X, **kwargs):
    transformer = MDS(normalized_stress='auto', **kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat

def Spectral_Embedding(X, **kwargs):
    transformer = SpectralEmbedding(**kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat

import umap
def U_MAP(X, **kwargs):
    reducer = umap.UMAP(**kwargs)
    X_hat = reducer.fit_transform(X)

    return X_hat, X_hat


from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs, make_sparse_spd_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import rand_score, adjusted_rand_score, mean_squared_error, silhouette_score, \
    fowlkes_mallows_score, calinski_harabasz_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, davies_bouldin_score
import warnings

warnings.filterwarnings("ignore")


from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering
from sklearn.datasets import make_moons, make_blobs
from scipy.sparse.csgraph import connected_components

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import pandas as pd
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
# Evaluation function
###############

def Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method):
    
    if mode == "Supervised":
        if evaluation_method == "adjusted_rand_score":
            score = adjusted_rand_score(clusters, labels)
        if evaluation_method == "fowlkes":
            score = fowlkes_mallows_score(clusters, labels)
        if evaluation_method == "mutual_info":
            score = adjusted_mutual_info_score(clusters, labels)
        if evaluation_method == "homogeneity":
            score, completeness, v_measure = homogeneity_completeness_v_measure(clusters, labels)
        if evaluation_method == "completeness":
            homogeneity, score, v_measure = homogeneity_completeness_v_measure(clusters, labels)
        if evaluation_method == "v_measure":
            homogeneity, completeness, score = homogeneity_completeness_v_measure(clusters, labels)

    if mode == "Unsupervised":
        if evaluation_method == "silhouette":
            score = silhouette_score(X, labels)
        if evaluation_method == "calinski":
            score = calinski_harabasz_score(X, labels)
        if evaluation_method == "davies":
            score = davies_bouldin_score(X, labels)

    if mode == "Reconstruction":
        if evaluation_method == "MSE":
            score = mean_squared_error(X_r, X_scaled)
    
    return score
###############
# Subfunctions Consensus MF
###############

def Supervised_Consensus(MFiters, method, X_scaled, clusters, mode, evaluation_method, **kwargs):
    '''
    Takes standarized matrix X_scaled, and number of iterations of MF MFiters.
    '''
    
    score_list = []

# Consensus MF
    for i in range(MFiters):

        if method == "PCA_sklearn":
            X, X_r = PCA_sklearn(X_scaled, **kwargs)

        if method == "NMF_sklearn":

            X, X_r = NMF_sklearn(X_scaled, **kwargs)

        if method == "MiniBatch_NMF":

            X, X_r = MiniBatch_NMF(X_scaled, **kwargs)

        if method == "Sparse_PCA":
            X, X_r = Sparse_PCA(X_scaled, **kwargs)

        if method == "Kernel_PCA":
            X, X_r = Kernel_PCA(X_scaled, **kwargs)

        if method == "Truncated_SVD":
            X, X_r = Truncated_SVD(X_scaled, **kwargs)

        if method == "Incremental_PCA":
            X, X_r = Incremental_PCA(X_scaled)

        if method == "IndependentCA":
            X, X_r = IndependentCA(X_scaled, **kwargs)

        if method == "MiniBatch_SparsePCA":
            X, X_r = MiniBatch_SparsePCA(X_scaled, **kwargs)
        
        if method == "Factor_Analysis":
            X, X_r = Factor_Analysis(X_scaled, **kwargs)

        if method == "Dictionary":
            X, X_r = Dictionary(X_scaled, **kwargs)

        if method == "LDA":
            X, X_r = LDA(X_scaled, **kwargs)

        if method == "t_SNE":
            X, X_r = t_SNE(X_scaled, **kwargs)

        if method == "Iso_map":
            X, X_r = Iso_map(X_scaled)

        if method == "Local_Linear_Embedding":
            X, X_r = Local_Linear_Embedding(X_scaled, **kwargs)

        if method == "Multidimensional_scaling":
            X, X_r = Multidimensional_scaling(X_scaled, **kwargs)

        if method == "Spectral_Embedding":
            X, X_r = Spectral_Embedding(X_scaled, **kwargs)

        if method == "U_MAP":
            X, X_r = U_MAP(X_scaled, **kwargs)

        labels = Ensemble_classifier_sklearn(X, 2)
    # accuracy evaluation of consensus clustering 

        score = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
    
        score_list.append(score)


    score_std = np.std(score_list)

    score_mean = np.mean(score_list)
    
    print(method + " evaluation metrics: ")
    print(evaluation_method + " score is ... " + str(score_mean) + " (" + str(score_std) + ")")

    return score_mean

###########
# Main pipeline 
###########

def Initial_Guess(X_scaled, method_list, clusters, mode, evaluation_method, **kwargs):
    '''
    Takes standarized/normalizaed matrix X_scaled, number of iterations MFiters, and
    list of methods to scan for optimal initial guess based on inverse transformation reconstruction
    mean squared error.
    '''

    # Number of iterations for Matrix Factorization
    MFiters = 3
    
    global_method_list = []
    global_score_list =  [] 
    for method in method_list:
        score = Supervised_Consensus(MFiters, method, X_scaled, clusters, mode, evaluation_method, **kwargs) 
        global_method_list.append(method)
        global_score_list.append(score)
    
    df = pd.DataFrame()
    df['method'] = global_method_list
    df['score'] = global_score_list
     
    most_optimal_model = df[df.score == df.score.max()]
    
    if mode == "Reconstruction":
         most_optimal_model = df[df.score == df.score.min()]

    print("Best initial guess is...")
    print(most_optimal_model)
    
    return


def Model_auto_select(X, clusters, mode, evaluation_method, **kwargs):
    """
    Selects MF method based on best initial guess (testing all methods with default parameters)
    by inputting original matrix.
    """
    # If negative values are present omit NMF methods 

    if (X >= 0).all():
        
        method_list = ["PCA_sklearn", "NMF_sklearn", "Kernel_PCA",
                "Truncated_SVD", "Incremental_PCA", "IndependentCA", "MiniBatch_NMF",
                "Sparse_PCA", "MiniBatch_SparsePCA", "Factor_Analysis",
                "Dictionary", "LDA", "t_SNE", "Iso_map", "Local_Linear_Embedding",
                "Multidimensional_scaling", "Spectral_Embedding", "U_MAP"]
        if mode == "Reconstruction":
            method_list = ["PCA_sklearn", "NMF_sklearn", "Kernel_PCA",
                "Truncated_SVD", "Incremental_PCA", "IndependentCA", "MiniBatch_NMF",
                "Sparse_PCA", "MiniBatch_SparsePCA"]

        Initial_Guess(X, method_list, clusters, mode, evaluation_method,  **kwargs)
    
    else:
        method_list = ["PCA_sklearn", "Kernel_PCA",
                "Truncated_SVD", "Incremental_PCA", "IndependentCA",
                "Sparse_PCA", "MiniBatch_SparsePCA", "Factor_Analysis",
                "Dictionary",  "t_SNE", "Iso_map", "Local_Linear_Embedding",
                "Multidimensional_scaling", "Spectral_Embedding", "U_MAP"]
        if mode == "Reconstruction":
            method_list = ["PCA_sklearn", "Kernel_PCA",
                "Truncated_SVD", "Incremental_PCA", "IndependentCA",
                "Sparse_PCA", "MiniBatch_SparsePCA"]


        Initial_Guess(X, method_list, clusters, mode, evaluation_method, **kwargs)

    return

# Main test
def maintest():
    """
    working test
    """
    # clusterable dense matrix example
    X, clusters = make_blobs(n_samples=200,
                             n_features=10,
                             centers=2,
                             cluster_std=0.4,
                             shuffle=True)
    
    X = preprocessing_standarization(X)
    
    Model_auto_select(X, clusters, mode = "Supervised", evaluation_method = "adjusted_rand_score",
            n_components = 2, random_state = None)
     
    Model_auto_select(X, clusters, mode = "Unsupervised", evaluation_method = "silhouette",
            n_components = 2, random_state = None)
      
    Model_auto_select(X, clusters, mode = "Reconstruction", evaluation_method = "MSE",
            n_components = 2, random_state = None)
    
    
    return


#maintest()
from sklearn import datasets

def Simulateddata_maintest():
    """
    AutoMF test with simulated/real data sets
    """
    # Iris dataset

    X = datasets.load_iris().data
    clusters = [0] # placeholder for non supervised modes

    Model_auto_select(X, clusters, mode = "Reconstruction", evaluation_method = "MSE",
            n_components = 2, random_state = None)

    
    # NMF synthetic simulated data
    df = pd.read_csv("NMFsim.tsv", sep = '\t')

    X = df.values

    Model_auto_select(X, clusters, mode = "Reconstruction", evaluation_method = "MSE",
            n_components = 2, random_state = None)
    
    # Single Cell Splatter simulated data 2 groups
    df = pd.read_csv("singlecell.tsv", sep = '\t')

    X = df.values

    Model_auto_select(X, clusters, mode = "Unsupervised", evaluation_method = "silhouette",
            n_components = 2, random_state = None)
    
    # Microarray simulations 2 groups

    df = pd.read_csv("microarraysim2.tsv", sep = '\t')
    X = df.values

    Model_auto_select(X, clusters, mode = "Unsupervised", evaluation_method = "silhouette",
            n_components = 2, random_state = None)
    return

Simulateddata_maintest()
