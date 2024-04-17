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

    model = NMF(**kwargs)
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

    model = SparsePCA(**kwargs)
    W = model.fit_transform(X)
    X_r = model.inverse_transform(W)
    
    return W, X_r

# Kernel PCA
def Kernel_PCA(X, **kwargs):

    model = KernelPCA(fit_inverse_transform = True, **kwargs)
    W = model.fit_transform(X)
    X_r = model.inverse_transform(W)

    return W, X_r


# Truncated SVD
def Truncated_SVD(X, **kwargs):

    model = TruncatedSVD(**kwargs)
    W = model.fit_transform(X)
    X_r = model.inverse_transform(W)
    
    return W, X_r


# IncrementalPCA
def Incremental_PCA(X,**kwargs):

    model = IncrementalPCA(**kwargs)
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

    model = MiniBatchNMF(**kwargs)
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

    transformer = DictionaryLearning(**kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat


def LDA(X, **kwargs):

    transformer = LatentDirichletAllocation(**kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat


def t_SNE(X, **kwargs):

    transformer = TSNE(**kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat


def Iso_map(X, **kwargs):

    transformer = Isomap(**kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat


def Local_Linear_Embedding(X, **kwargs):

    transformer = LocallyLinearEmbedding(**kwargs)
    X_hat = transformer.fit_transform(X)

    return X_hat, X_hat


def Multidimensional_scaling(X, **kwargs):

    transformer = MDS(**kwargs)
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

# Plotting functions visualizations with seaborn

import seaborn as sns
import matplotlib.pyplot as plt


def plot_score_comparison(df):

    sns.set_style("white")
    plot = sns.barplot(data=df, x="method", y="score")

    for item in plot.get_xticklabels():
        item.set_rotation(90)

    plt.tight_layout()

    fig = plot.get_figure()
    fig.savefig("Score_vs_method.png")

    return


def plot_clustering(X, labels):


    PC1 = X[:,0]
    PC2 = X[:,1]
    zipped = list(zip(PC1,
                  PC2,
                  labels))

    pc_df = pd.DataFrame(zipped,
                     columns=['PC1',
                              'PC2',
                              'Label'])

    plt.figure(figsize=(12,7))

    plot2 = sns.scatterplot(data=pc_df,
                x="PC1",
                y="PC2",
                hue="Label")

    plt.title("Figure 2",
          fontsize=16)
    plt.xlabel('First Principal Component',
           fontsize=16)
    plt.ylabel('Second Principal Component',
           fontsize=16)

    fig2 = plot2.get_figure()
    fig2.savefig("clustering.png")

    return


#################
# Consensus Ensamble clustering classifier K-means + spectral 
#################

from scipy.spatial.distance import cdist
import numpy as np

class ClusterSimilarityMatrix():
    """
    ClusterSimilarityMatrix is a class that computes the similarity matrix between clusters
    of different clustering algorithms. It is used to build an ensemble clustering algorithm
    based on the consensus of different clustering algorithms.
    """
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
    """
    EnsembleCustering is a class that implements an ensemble clustering algorithm based on
    the consensus of different clustering algorithms.
    """
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
        MiniBatchKMeans(n_clusters= number_clusters, batch_size=64, n_init=1, max_iter=20, random_state = 42)
    ]
    aggregator_clt = SpectralClustering(n_clusters= number_clusters, affinity="precomputed", random_state = 42)
    
    ens_clt = EnsembleCustering(clustering_models, aggregator_clt)
    y_ensemble = ens_clt.fit_predict(X)

    return y_ensemble


###############
# Evaluation function
###############

def Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method):
    """
    Evaluation function for clustering and reconstruction methods.
    X is the transformed matrix 
    X_r is the inverse transformed matrix for Reconstruction method only
    X_scaled is the original data matrix
    labels are the clustering labels
    clusters are the true labels for Supervised method only
    mode - Supervised, Unsupervised, Reconstruction
    evaluation method - selects evaluation score metrics 
    """
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
        score = 1 - np.abs(score)
        
    if mode == "Unsupervised":
        if evaluation_method == "silhouette":
            score = silhouette_score(X, labels)
        if evaluation_method == "calinski":
            score = calinski_harabasz_score(X, labels)
        if evaluation_method == "davies":
            score = davies_bouldin_score(X, labels)
        score = 1 - np.abs(score)
        
    if mode == "Reconstruction":
        if evaluation_method == "MSE":
            score = mean_squared_error(X_r, X_scaled)
    
    return score


###############
# Subfunctions Consensus MF
###############

def Supervised_Consensus(MFiters, method, X_scaled, clusters, number_clusters, mode, evaluation_method, visualizations, **kwargs):
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
            X, X_r = Incremental_PCA(X_scaled, **kwargs)

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
            X, X_r = Iso_map(X_scaled, **kwargs)

        if method == "Local_Linear_Embedding":
            X, X_r = Local_Linear_Embedding(X_scaled, **kwargs)

        if method == "Multidimensional_scaling":
            X, X_r = Multidimensional_scaling(X_scaled, **kwargs)

        if method == "Spectral_Embedding":
            X, X_r = Spectral_Embedding(X_scaled, **kwargs)

        if method == "U_MAP":
            X, X_r = U_MAP(X_scaled, **kwargs)

        labels = Ensemble_classifier_sklearn(X, number_clusters)
        # accuracy evaluation of consensus clustering 

        score = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
    
        score_list.append(score)


    score_std = np.std(score_list)
    score_mean = np.mean(score_list)

    return score_mean


###########
# Main pipeline 
###########

import ast

def Select_Best_Incument(X_scaled, method_list, clusters, number_clusters, mode, evaluation_method, visualizations,  **kwargs):
    '''
    Takes standarized/normalizaed matrix X_scaled, number of iterations MFiters, and
    list of methods to scan for optimal best incument by running Bayesian Optimization in all methods in method list.
    '''    

    # Number of iterations for Matrix Factorization
    MFiters = 3
    
    global_method_list = []
    global_score_list =  [] 
    dict_incumbent_list = []

    for method in method_list:
        dict_incumbent, score, oX, o_labels = Bayesian_Optimization(X_scaled, clusters, number_clusters, method, mode, evaluation_method, visualizations = False)
        global_method_list.append(method)
        global_score_list.append(score)
        dict_incumbent_list.append(str(dict_incumbent))
    
    df = pd.DataFrame()
    df['optimized_method'] = global_method_list
    df['score'] = global_score_list
    df['incumbent'] = dict_incumbent_list

    df['incumbent'] = df['incumbent'].apply(ast.literal_eval)
    
    if visualizations:
        plot_score_comparison(df)

    most_optimal_model = df[df.score == df.score.min()]

    print("Best optimal model found by Bayesian Optimization is...")
    print(most_optimal_model)
    
    print("Now performing Consensus Ensamble Clustering with best optimal model...")
    mean_score = Supervised_Consensus(MFiters, most_optimal_model['optimized_method'][0], X_scaled, clusters, number_clusters, mode, evaluation_method, visualizations, **most_optimal_model['incumbent'][0])
    print("Mean score from Consensus Clustering is...")
    print(mean_score)
    return most_optimal_model, mean_score


def Model_auto_select(X_scaled, clusters, number_clusters, mode, evaluation_method, visualizations, **kwargs):
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

        optimal_model = Select_Best_Incument(X_scaled, method_list, clusters, number_clusters, mode, evaluation_method, visualizations,  **kwargs)
    
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


        optimal_model = Select_Best_Incument(X_scaled, method_list, clusters, number_clusters, mode, evaluation_method, visualizations, **kwargs)

    return optimal_model


from sklearn import datasets
from sklearn.datasets import make_classification

def is_invertible(a):
    """
    Checks if matrix is invertible
    """
    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]


from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer, Categorical

from sklearn.decomposition import PCA
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario
from ConfigSpace.conditions import InCondition
from smac import HyperparameterOptimizationFacade, Scenario

##################################################################################
# SMAC Bayesian Optmiziation classes
##################################################################################

class PCA_BO:
    """
    PCA class for Bayesian Optimization with SMAC3
    configspace: ConfigurationSpace object, hyperparameter gridsearch for Bayesian Optimization
    train: function that takes a Configuration object and returns a float, task to be optimized using Evaluation_metrics function
    """
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        svd_solver = Categorical("svd_solver", ['auto', 'full', 'randomized'], default="auto")
        n_components = Integer("n_components", (2, 5), default=3)

        power_iteration_normalizer = Categorical("power_iteration_normalizer",['auto', 'QR','LU'], default = 'auto')
        use_power = InCondition(child=power_iteration_normalizer, parent=svd_solver, values=["randomized"])
        
        n_oversamples = Integer("n_oversamples", (3,11), default=10)
        use_overs = InCondition(child = n_oversamples, parent = svd_solver, values=["randomized"])
        
        cs.add_hyperparameters([n_components, svd_solver, power_iteration_normalizer, n_oversamples])
        cs.add_conditions([use_power,use_overs])
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        
        config_dict = config.get_dictionary()
        
        X, X_r = PCA_sklearn(X_scaled, **config_dict)
        
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation


class NMF_BO:
    """
    NMF class for Bayesian Optimization with SMAC3
    configspace: ConfigurationSpace object, hyperparameter gridsearch for Bayesian Optimization
    train: function that takes a Configuration object and returns a float, task to be optimized using Evaluation_metrics function
    """
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        init = Categorical("init", ['random', 'nndsvd', 'nndsvda', 'nndsvdar'], default='random')
        solver = Categorical("solver", ['cd', 'mu'], default='cd')
        beta_loss = Categorical("beta_loss", ['frobenius', 'kullback-leibler','itakura-saito'], default='frobenius')
        use_beta = InCondition(child=beta_loss, parent=solver, values=["mu"])
        n_components = Integer("n_components", (2, 5), default=3)
        verbose = Integer("verbose", (0, 1), default=0)
        l1_ratio = Float("l1_ratio", (0.0, 1.0), default=0.0)
        alpha_W = Float("alpha_W", (0.0, 1.0), default=0.0)

        cs.add_hyperparameters([n_components, init, solver, beta_loss, verbose, l1_ratio, alpha_W])
        cs.add_conditions([use_beta])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
    
        config_dict = config.get_dictionary()
        
        X, X_r = NMF_sklearn(X_scaled, **config_dict)
        
        
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation


class MiniBatchNMF_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        init = Categorical("init", ['random', 'nndsvd', 'nndsvda', 'nndsvdar'], default='random')
        n_components = Integer("n_components", (2, 5), default=3)
        batch_size = Integer("batch_size", (100,2000), default=1024)
        beta_loss = Categorical("beta_loss", ['frobenius', 'kullback-leibler','itakura-saito'], default='frobenius')
        l1_ratio = Float("l1_ratio", (0.0, 1.0), default=0.0)
        alpha_W = Float("alpha_W", (0.0, 1.0), default=0.0)

        cs.add_hyperparameters([n_components, init, batch_size, beta_loss, l1_ratio, alpha_W])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        
        config_dict = config.get_dictionary()
        
        X, X_r = MiniBatch_NMF(X_scaled, **config_dict)
        
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation
 

class KernelPCA_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        kernel = Categorical("kernel", ['linear', 'poly', 'rbf', 'sigmoid', 'cosine'], default='linear')
        n_components = Integer("n_components", (2, 5), default=3)
        degree = Integer("degree", (2, 5), default=3)
        eigen_solver = Categorical("eigen_solver", ['auto', 'dense', 'arpack', 'randomiZed'], default='auto')
        alpha = Float("alpha", (0.1, 1.0), default=1.0)
        cs.add_hyperparameters([n_components, kernel, degree, eigen_solver, alpha])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        
        config_dict = config.get_dictionary()
        
        X, X_r = Kernel_PCA(X_scaled, **config_dict)
        
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation


class TruncatedSVD_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        n_components = Integer("n_components", (2, 5), default=3)
        n_iter = Integer("n_iter", (3, 7), default=5)
        n_oversamples = Integer("n_oversamples", (3,11), default=10)
        power_iteration_normalizer = Categorical("power_iteration_normalizer",['auto', 'QR','LU'], default = 'auto')

        cs.add_hyperparameters([n_components, power_iteration_normalizer, n_iter, n_oversamples])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Truncated_SVD(X_scaled, **config_dict)    
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation       



class IncrementalPCA_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        n_cols = X_scaled.shape[1]

        batch_size = Integer("batch_size", (5 * n_cols, 5 *n_cols + 30), default=5 * n_cols) # check if default None if it works, None = 5 * n_features we can use this instead (5*n, 5*n + 20)
        n_components = Integer("n_components", (2, 5), default=3)

        
        cs.add_hyperparameters([n_components, batch_size])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Incremental_PCA(X_scaled, **config_dict)    
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation 



class FastICA_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        fun = Categorical("fun", ['logcosh', 'exp', 'cube'], default='logcosh')
        n_components = Integer("n_components", (2, 5), default=3)
        algorithm = Categorical("algorithm", ['parallel', 'deflation'], default='parallel')
        whiten_solver = Categorical("whiten_solver", ['eigh', 'svd'], default='svd')
        
        cs.add_hyperparameters([n_components, fun, algorithm, whiten_solver])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = IndependentCA(X_scaled, **config_dict)  
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation 
        

class SparsePCA_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        method = Categorical("method", ['lars', 'cd'], default='lars')
        n_components = Integer("n_components", (2, 5), default=3)
        alpha = Float("alpha", (0.1, 1.0), default=1.0)
        ridge_alpha = Float("ridge_alpha", (0.01, 0.10), default=0.01)

        
        cs.add_hyperparameters([n_components, method, alpha, ridge_alpha])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Sparse_PCA(X_scaled, **config_dict)  
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation 
        
        
class MiniBatchSparsePCA_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        method = Categorical("method", ['lars', 'cd'], default='lars')
        n_components = Integer("n_components", (2, 5), default=3)
        alpha = Float("alpha", (0.1, 1.0), default=1.0)
        ridge_alpha = Float("ridge_alpha", (0.01, 0.10), default=0.01)
        batch_size = Integer("batch_size", (2,5), default=3)

        cs.add_hyperparameters([n_components, method, alpha, ridge_alpha, batch_size])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = MiniBatch_SparsePCA(X_scaled, **config_dict)    
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation 
 

class FactorAnalysis_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        svd_method = Categorical("svd_method", ['lapack', 'randomized'], default='randomized')
        n_components = Integer("n_components", (2, 5), default=3)
        iterated_power = Integer("iterated_power", (2, 5), default=3)
        rotation = Categorical("rotation", ['varimax', 'quartimax'], default='varimax')
        
        cs.add_hyperparameters([n_components, svd_method, iterated_power, rotation])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Factor_Analysis(X_scaled, **config_dict)  
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation            



class DictionaryLearning_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        transform_algorithm = Categorical("transform_algorithm", ['lasso_cd', 'lasso_lars', 'omp', 'threshold', 'lars'], default='omp')
        n_components = Integer("n_components", (2, 5), default=3)
        alpha = Float("alpha", (0.1, 1.0), default=1.0)
        fit_algorithm = Categorical("fit_algorithm", ['lars', 'cd'], default='lars')
        #Include fit algorithm parameter
        
        cs.add_hyperparameters([n_components, transform_algorithm, alpha, fit_algorithm])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Dictionary(X_scaled, **config_dict)  
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation 


class LatentDirichletAllocation_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        learning_method = Categorical("learning_method", ['batch', 'online'], default='batch')
        n_components = Integer("n_components", (2, 15), default=10)
        learning_decay = Float("learning_decay", (0.5, 1.0), default=0.7)
        learning_offset = Float("learning_offset", (5, 15), default=10)
        max_iter = Integer("max_iter", (5, 15), default=10)


        
        cs.add_hyperparameters([n_components, learning_method, learning_decay, learning_offset, max_iter])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = LDA(X_scaled, **config_dict)      
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation


class TSNE_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        method = Categorical("method", ['barnes_hut', 'exact'], default='barnes_hut')
        n_components = Integer("n_components", (2, 3), default=2)
        perplexity = Float("perplexity", (5, 50), default=30)
        early_exaggeration = Float("early_exaggeration", (5, 50), default=12)
        learning_rate = Float("learning_rate", (10, 1000), default=200)
        n_iter = Integer("n_iter", (250, 1000), default=1000)
        n_iter_without_progress = Integer("n_iter_without_progress", (50, 500), default=300)
        init = Categorical("init", ['random', 'pca'], default='pca')
        angle = Float("angle", (0.1, 1.0), default=0.5)
        cs.add_hyperparameters([n_components, method, perplexity, early_exaggeration, learning_rate, n_iter, n_iter_without_progress, init, angle])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = t_SNE(X_scaled, **config_dict)     
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation

                
                
class Isomap_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        n_neighbors = Integer("n_neighbors", (5,15), default=5)
        eigen_solver = Categorical("eigen_solver", ['auto', 'arpack', 'dense'], default='auto')
        n_components = Integer("n_components", (2, 5), default=2)
        path_method = Categorical("path_method", ['auto', 'FW', 'D'], default='auto')
        neighbors_algorithm = Categorical("neighbors_algorithm", ['auto', 'brute', 'kd_tree', 'ball_tree'], default='auto')
        
        cs.add_hyperparameters([eigen_solver, n_neighbors, n_components, path_method, neighbors_algorithm])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Iso_map(X_scaled, **config_dict)      
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation               
                

class LocallyLinearEmbedding_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        method = Categorical("method", ['standard', 'modified', 'ltsa'], default='standard') # Hessian method need to set condition with number of neighbors functin of number of components
        n_components = Integer("n_components", (2, 5), default=3)
        eigen_solver = Categorical("eigen_solver", ['auto', 'arpack', 'dense'], default='auto')
        n_neighbors = Integer("n_neighbors", (5,15), default=5)
        neighbors_algorithm = Categorical("neighbors_algorithm", ['auto', 'brute', 'kd_tree', 'ball_tree'], default='auto')
        # add n_neigh = 25
        cs.add_hyperparameters([n_components, method, eigen_solver, n_neighbors, neighbors_algorithm])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Local_Linear_Embedding(X_scaled, **config_dict)      
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation  



class MDS_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        dissimilarity = Categorical("dissimilarity", ['euclidean'], default='euclidean')
        n_components = Integer("n_components", (2, 5), default=3)
        n_init = Integer("n_init", (2, 5), default=4)

        # add n_neigh = 25
        cs.add_hyperparameters([n_components, dissimilarity, n_init])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Multidimensional_scaling(X_scaled, **config_dict)      
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation  



class SpectralEmbedding_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        eigen_solver = Categorical("eigen_solver", ['lobpcg', 'amg'], default='lobpcg')
        n_components = Integer("n_components", (2, 5), default=3)
        affinity = Categorical("affinity", ['nearest_neighbors', 'rbf'], default='nearest_neighbors')

        cs.add_hyperparameters([n_components, eigen_solver, affinity])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = Spectral_Embedding(X_scaled, **config_dict)      
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation    


class UMAP_BO:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        n_neighbors = Integer("n_neighbors", (5,100), default=10)
        n_components = Integer("n_components", (2, 3), default=2)
        min_dist = Float("min_dist", (0.0, 0.99), default=0.1)
        metric = Categorical("metric", ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 
                                        'canberra', 'braycurtis', 'haversine',
                                        'mahalanobis', 'wminkowski', 'seuclidean',
                                        'cosine', 'correlation',
                                        'hamming', 'jaccard', 'dice', 'russellrao',
                                        'kulsinski', 'rogerstanimoto', 'sokalmichener',
                                        'sokalsneath', 'yule'
                                        ], default='euclidean')

        cs.add_hyperparameters([n_components, n_neighbors, min_dist, metric])
 
        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
        config_dict = config.get_dictionary()
        
        X, X_r = U_MAP(X_scaled, **config_dict)      
        if mode == "Supervised" or mode == "Unsupervised":
            labels = Ensemble_classifier_sklearn(X, number_clusters)
        else:
            labels = [0]
                    
        evaluation = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
        
        return evaluation



def Bayesian_Optimization(X_scaled, clusters, number_clusters, method, mode, evaluation_method, visualizations, **kwargs):

    mode = mode
    evaluation_method = evaluation_method
    
    if method == "PCA_sklearn":
    
        model = PCA_BO()

    if method == "NMF_sklearn":
        
        model = NMF_BO()
        
    if method == "MiniBatch_NMF":

        model = MiniBatchNMF_BO()
        
    if method == "Sparse_PCA":
        
        model = SparsePCA_BO()
        
    if method == "Kernel_PCA":
        
        model = KernelPCA_BO()
        
    if method == "Truncated_SVD":
        
        model = TruncatedSVD_BO()
        
    if method == "Incremental_PCA":
        
        model = IncrementalPCA_BO()
        
    if method == "IndependentCA":

        model = FastICA_BO()
        
    if method == "MiniBatch_SparsePCA":
        
        model = MiniBatchSparsePCA_BO()
        
    if method == "Factor_Analysis":
        
        model = FactorAnalysis_BO()
        
    if method == "Dictionary":
        
        model = DictionaryLearning_BO()
        
    if method == "LDA":
        
        model = LatentDirichletAllocation_BO()
        
    if method == "t_SNE":
        
        model = TSNE_BO()

    if method == "Iso_map":

        model = Isomap_BO()

    if method == "Local_Linear_Embedding":
        
        model = LocallyLinearEmbedding_BO()

    if method == "Multidimensional_scaling":
        
        model = MDS_BO()

    if method == "Spectral_Embedding":
        
        model = SpectralEmbedding_BO()
        
    if method == "U_MAP":
        
        model = UMAP_BO()
        
    scenario = Scenario(model.configspace, n_trials=10)
    
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

    smac = HyperparameterOptimizationFacade(
        scenario,
        model.train,
        initial_design=initial_design,
        overwrite=True,  
    )

    incumbent = smac.optimize()

    print(incumbent)
    dict_incumbent = incumbent.get_dictionary()
    
    if method == "PCA_sklearn":
    
        oX, oX_r = PCA_sklearn(X_scaled,**dict_incumbent)

    if method == "NMF_sklearn":
        
        oX, oX_r = NMF_sklearn(X_scaled,**dict_incumbent)
        
    if method == "MiniBatch_NMF":

        oX, oX_r = MiniBatch_NMF(X_scaled,**dict_incumbent)
        
    if method == "Sparse_PCA":
        
        oX, oX_r = Sparse_PCA(X_scaled,**dict_incumbent)
        
    if method == "Kernel_PCA":
        
        oX, oX_r = Kernel_PCA(X_scaled,**dict_incumbent)
        
    if method == "Truncated_SVD":
        
        oX, oX_r = Truncated_SVD(X_scaled,**dict_incumbent)
        
    if method == "Incremental_PCA":
        
        oX, oX_r = Incremental_PCA(X_scaled,**dict_incumbent)
        
    if method == "IndependentCA":

        oX, oX_r = IndependentCA(X_scaled,**dict_incumbent)
        
    if method == "MiniBatch_SparsePCA":
        
        oX, oX_r = MiniBatch_SparsePCA(X_scaled,**dict_incumbent)
        
    if method == "Factor_Analysis":
        
        oX, oX_r = Factor_Analysis(X_scaled,**dict_incumbent)
        
    if method == "Dictionary":
        
        oX, oX_r = Dictionary(X_scaled,**dict_incumbent)
        
    if method == "LDA":
        
        oX, oX_r = LDA(X_scaled,**dict_incumbent)
        
    if method == "t_SNE":
        
        oX, oX_r = t_SNE(X_scaled,**dict_incumbent)

    if method == "Iso_map":

        oX, oX_r = Iso_map(X_scaled,**dict_incumbent)

    if method == "Local_Linear_Embedding":
        
        oX, oX_r = Local_Linear_Embedding(X_scaled,**dict_incumbent)

    if method == "Multidimensional_scaling":
        
        oX, oX_r = Multidimensional_scaling(X_scaled,**dict_incumbent)

    if method == "Spectral_Embedding":
        
        oX, oX_r = Spectral_Embedding(X_scaled,**dict_incumbent)
        
    if method == "U_MAP":
        
        oX, oX_r = U_MAP(X_scaled,**dict_incumbent)

    if mode == "Supervised" or mode == "Unsupervised":
        o_labels = Ensemble_classifier_sklearn(oX, number_clusters)
    else:
        o_labels = [0]

    score = Evaluation_metrics(oX, oX_r, X_scaled, o_labels, clusters, mode, evaluation_method)
    print("Incumbent score ...")
    print(score)

    ################### visualizations for clustering ##################################
    if visualizations:
        plot_clustering(oX, o_labels)

    return dict_incumbent, score, oX, o_labels


BayesianOptimization_toyspacetest = True

if BayesianOptimization_toyspacetest:
    """
    Testing Bayesian Hyperparameter search module only test search space
    """
        # Define globally *** for now since we are still using functions in the future use class self
    mode = "Supervised"
    evaluation_method = "adjusted_rand_score" 
    number_clusters = 2
    
        # clusterable dense matrix example
    X, clusters = make_blobs(n_samples=200,
                             n_features=10,
                             centers=number_clusters,
                             cluster_std=0.4,
                             shuffle=True)

    X_scaled = preprocessing_standarization(X)
    

    method_list = ["PCA_sklearn", "Kernel_PCA",
                "Truncated_SVD", "Incremental_PCA", "IndependentCA",
                "Sparse_PCA", "MiniBatch_SparsePCA", "Factor_Analysis",
                "Dictionary", "t_SNE", "Iso_map", "Local_Linear_Embedding",
                "Multidimensional_scaling", "Spectral_Embedding", "U_MAP"]
    
    for optimal_model in method_list:
    #Model_optimization(X, clusters, optimal_model, "Reconstruction", "MSE")
        print(optimal_model)    
        Bayesian_Optimization(X_scaled, clusters, 2,  optimal_model, mode, evaluation_method, visualizations = False)
    #Model_optimization(X, clusters, optimal_model, "Unsupervised", "silhouette")
    
    X_scaled = np.absolute(X_scaled)
    
    Bayesian_Optimization(np.absolute(X_scaled), clusters, 2, "NMF_sklearn", mode, evaluation_method, visualizations = False)
    Bayesian_Optimization(np.absolute(X_scaled), clusters, 2, "MiniBatch_NMF", mode, evaluation_method, visualizations = False)
    Bayesian_Optimization(np.absolute(X_scaled), clusters, 2, "LDA", mode, evaluation_method, visualizations = False)

Bayesian_pipeline_test = False


if Bayesian_pipeline_test:
    """
    Testing Bayesian Optimization pipeline
    """
        # Define globally *** for now since we are still using functions in the future use class self
    mode = "Supervised"
    evaluation_method = "adjusted_rand_score" 
    number_clusters = 2
    
        # clusterable dense matrix example
    X, clusters = make_blobs(n_samples=200,
                             n_features=10,
                             centers=number_clusters,
                             cluster_std=0.4,
                             shuffle=True)

    X_scaled = preprocessing_standarization(X)
    Model_auto_select(X_scaled, clusters, number_clusters, mode, evaluation_method, visualizations = False, random_state = 0)

