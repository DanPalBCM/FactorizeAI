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

        labels = Ensemble_classifier_sklearn(X, number_clusters)
    # accuracy evaluation of consensus clustering 

        score = Evaluation_metrics(X, X_r, X_scaled, labels, clusters, mode, evaluation_method)
    
        score_list.append(score)


    score_std = np.std(score_list)

    score_mean = np.mean(score_list)
        
    #print(method + " evaluation metrics: ")
    #print(evaluation_method + " score is ... " + str(score_mean) + " (" + str(score_std) + ")")

    return score_mean

###########
# Main pipeline 
###########

def Initial_Guess(X_scaled, method_list, clusters, number_clusters, mode, evaluation_method, visualizations,  **kwargs):
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
        score = Supervised_Consensus(MFiters, method, X_scaled, clusters, number_clusters, mode, evaluation_method, visualizations, **kwargs) 
        global_method_list.append(method)
        global_score_list.append(score)
    
    df = pd.DataFrame()
    df['method'] = global_method_list
    df['score'] = global_score_list
    
    if visualizations:
        plot_score_comparison(df)

    most_optimal_model = df[df.score == df.score.max()]
    
    if mode == "Reconstruction":
         most_optimal_model = df[df.score == df.score.min()]

    print("Best initial guess is...")
    print(most_optimal_model)
    
    return most_optimal_model


def Model_auto_select(X, clusters, number_clusters, mode, evaluation_method, visualizations, **kwargs):
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

        optimal_model = Initial_Guess(X, method_list, clusters, number_clusters, mode, evaluation_method, visualizations,  **kwargs)
    
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


        optimal_model = Initial_Guess(X, method_list, clusters, number_clusters, mode, evaluation_method, visualizations, **kwargs)

    return optimal_model




# For supervised classification gridsearch 
def Model_optimization(X, clusters, number_clusters,  optimal_model, mode, evaluation_method, visualizations):
    print(optimal_model) 
    if optimal_model == "PCA_sklearn":
        n_components_list = [2,3,4,5]
        svd_solver_list = ['auto', 'full', 'arpack', 'randomized']
        cmse = np.zeros((len(n_components_list), len(svd_solver_list)))
        
        for i in range(len(n_components_list)):
            for j in range(len(svd_solver_list)):

                model = PCA(n_components = n_components_list[i], svd_solver = svd_solver_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score
        
        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = PCA(n_components = n_components_list[min_index[0]], svd_solver = svd_solver_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)
        
        oX_r = o_model.inverse_transform(oW)
        
        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]
        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)
        
        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")
        
        print(score)

    if optimal_model == "NMF_sklearn":
        
        n_components_list = [2,3,4,5]
        init_list = ['random', 'nndsvd', 'nndsvda', 'nndsvdar', None]
        cmse = np.zeros((len(n_components_list), len(init_list)))

        for i in range(len(n_components_list)):
            for j in range(len(init_list)):

                model = NMF(n_components = n_components_list[i], init = init_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = NMF(n_components = n_components_list[min_index[0]], init = init_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)

        
        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)


        print("Best parameters found ...")

        print(score)

    if optimal_model == "MiniBatch_NMF":
        
        n_components_list = [2,3,4,5]
        init_list = ['random', 'nndsvd', 'nndsvda', 'nndsvdar', None]
        cmse = np.zeros((len(n_components_list), len(init_list)))

        for i in range(len(n_components_list)):
            for j in range(len(init_list)):

                model = MiniBatchNMF(n_components = n_components_list[i], init = init_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = MiniBatchNMF(n_components = n_components_list[min_index[0]], init = init_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)


        print("Best parameters found ...")

        print(score)
    
    if optimal_model == "Kernel_PCA":
        n_components_list = [2,3,4,5]
        # No precomputed kernel for inversetransorm
        kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
        cmse = np.zeros((len(n_components_list), len(kernel_list)))

        for i in range(len(n_components_list)):
            for j in range(len(kernel_list)):

                model = KernelPCA(fit_inverse_transform = True, n_components = n_components_list[i], kernel = kernel_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = KernelPCA(fit_inverse_transform = True, n_components = n_components_list[min_index[0]], kernel = kernel_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)


        print("Best parameters found ...")

        print(score)


    if optimal_model == "Truncated_SVD":
        param1_list = [2,3,4,5]
        param2_list = ['OR', 'LU', 'none']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = TruncatedSVD(n_components = param1_list[i], power_iteration_normalizer = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = TruncatedSVD(n_components = param1_list[min_index[0]], power_iteration_normalizer = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)


    if optimal_model == "Incremental_PCA":
        param1_list = [2,3,4,5]
        param2_list = [10, 20, None]
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = IncrementalPCA(n_components = param1_list[i], batch_size = param2_list[j])
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = IncrementalPCA(n_components = param1_list[min_index[0]], batch_size = param2_list[min_index[1]])

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)


        print("Best parameters found ...")

        print(score)


    if optimal_model == "IndependentCA":
        param1_list = [2,3,4,5]
        param2_list = ['logcosh', 'exp', 'cube']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = FastICA(n_components = param1_list[i], fun = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = FastICA(n_components = param1_list[min_index[0]], fun = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)


    if optimal_model == "Sparse_PCA":
        param1_list = [2,3,4,5]
        param2_list = ['lars', 'cd']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = SparsePCA(n_components = param1_list[i], method = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = SparsePCA(n_components = param1_list[min_index[0]], method = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)



    
    if optimal_model == "MiniBatch_SparsePCA":
        param1_list = [2,3,4,5]
        param2_list = ['lars', 'cd']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = MiniBatchSparsePCA(n_components = param1_list[i], method = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = MiniBatchSparsePCA(n_components = param1_list[min_index[0]], method = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)



    if optimal_model == "Factor_Analysis":
        param1_list = [2,3,4,5]
        param2_list = ['lapack', 'randomized']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = FactorAnalysis(n_components = param1_list[i], svd_method = param2_list[j], random_state = 42)
                W = model.fit_transform(X)
                
                #noninvertable
                X_r = W

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = FactorAnalysis(n_components = param1_list[min_index[0]], svd_method = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)


        print("Best parameters found ...")

        print(score)
    

    if optimal_model == "Dictionary":
        param1_list = [2,3,4,5]
        param2_list = ['lasso_cd', 'lasso_lars', 'omp', 'threshold', 'lars']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = DictionaryLearning(max_iter = 10, n_components = param1_list[i], transform_algorithm = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = W

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = DictionaryLearning(max_iter = 10, n_components = param1_list[min_index[0]], transform_algorithm = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)


        print("Best parameters found ...")

        print(score)



    if optimal_model == "LDA":
        param1_list = [2,3,4,5]
        param2_list = ['batch', 'online']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = LatentDirichletAllocation(n_components = param1_list[i], learning_method = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = W

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = LatentDirichletAllocation(n_components = param1_list[min_index[0]], learning_method = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)


    if optimal_model == "t_SNE":
        # no more than 4 compnents for barnes_hut algorithm
        param1_list = [2,3]
        param2_list = ['barnes_hut', 'exact']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = TSNE(n_components = param1_list[i], method = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = W

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = TSNE(n_components = param1_list[min_index[0]], method = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)


    if optimal_model == "Iso_map":
        param1_list = [2,3,4,5]
        param2_list = [5,10,15]
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = Isomap(n_components = param1_list[i], n_neighbors = param2_list[j])
                W = model.fit_transform(X)

                X_r = W

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = Isomap(n_components = param1_list[min_index[0]], n_neighbors = param2_list[min_index[1]])

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)



    if optimal_model == "Local_Linear_Embedding":
        param1_list = [2,3,4,5]
        param2_list = ['standard', 'hessian', 'modified', 'ltsa']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = LocallyLinearEmbedding(n_neighbors = 25, n_components = param1_list[i], method = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = W

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = LocallyLinearEmbedding(n_neighbors = 25, n_components = param1_list[min_index[0]], method = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)

    if optimal_model == "Multidimensional_scaling":
        param1_list = [2,3,4,5]
        # precomputed needs square input
        param2_list = ['euclidean']
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = MDS(n_components = param1_list[i], dissimilarity = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = W 

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = MDS(n_components = param1_list[min_index[0]], dissimilarity = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)

    
    if optimal_model == "Spectral_Embedding":
        param1_list = [2,3,4,5]
        # AMG requires pyamg installation
        param2_list = ['arpack', 'lobpcg', 'amg']

        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = SpectralEmbedding(n_components = param1_list[i], eigen_solver = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = W

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = SpectralEmbedding(n_components = param1_list[min_index[0]], eigen_solver = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)


    if optimal_model == "U_MAP":
        param1_list = [2,3,4,5]
        # AMG requires pyamg installation
        param2_list = [5, 10, 100]
        # UMAP wants n_neighbors to be higher than n_components * (n_components + 3) / 2
        cmse = np.zeros((len(param1_list), len(param2_list)))

        for i in range(len(param1_list)):
            for j in range(len(param2_list)):

                model = umap.UMAP(n_components = param1_list[i], n_neighbors = param2_list[j], random_state = 42)
                W = model.fit_transform(X)

                X_r = W

                if mode == "Supervised" or mode == "Unsupervised":
                    labels = Ensemble_classifier_sklearn(W, number_clusters)

                else:
                    labels = [0]

                score = Evaluation_metrics(W, X_r, X, labels, clusters, mode, evaluation_method)
                print(score)
                cmse[i][j] = score

        if mode == "Reconstruction":
            min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        else:
            # We want max score
            min_index = np.unravel_index(np.argmax(cmse, axis=None), cmse.shape)

        o_model = umap.UMAP(n_components = param1_list[min_index[0]], n_neighbors = param2_list[min_index[1]], random_state = 42)

        oW = o_model.fit_transform(X)

        oX_r = oW

        if mode == "Supervised" or mode == "Unsupervised":
            o_labels = Ensemble_classifier_sklearn(oW, number_clusters)
        else:
            o_labels = [0]

        score = Evaluation_metrics(oW, oX_r, X, o_labels, clusters, mode, evaluation_method)

        ################### visualizations for clustering ##################################
        if visualizations:
            plot_clustering(oW, o_labels)

        print("Best parameters found ...")

        print(score)
    
    # RETURNS optimal algorithm o_model, transformed matrix oW, and resulting clustering labels

    return o_model, oW, o_labels 

# MAIN CONNECTING WRAPPER 

def AutoMF_main(X, clusters, number_clusters, mode, evaluation_method, visualizations, **kwargs):

    optimal_model = Model_auto_select(X, clusters, number_clusters, mode, evaluation_method, visualizations, **kwargs)

    optimal_finetuned_model, X_transformed, clustering_labels = Model_optimization(X, clusters, number_clusters, str(optimal_model["method"].tolist()[0]), mode, evaluation_method, visualizations)
    
    print(optimal_finetuned_model)
    
    return

from sklearn import datasets
from sklearn.datasets import make_classification
def is_invertible(a):

    return a.shape[0] == a.shape[1] and np.linalg.matrix_rank(a) == a.shape[0]

def Main_Test():
    """
    Working test

    Test all 3 modes in a diverse family of toy and simulated datasets. 
    """
## Supervised ##

    # clusterable dense matrix example
    n_clusters = 2
    X, clusters = make_blobs(n_samples=200,
                             n_features=10,
                             centers=n_clusters,
                             cluster_std=0.4,
                             shuffle=True)
    
    print("Make blobs test with " + str(n_clusters) + " clusters: ")
    AutoMF_main(X, clusters, n_clusters, mode = "Supervised", evaluation_method = "adjusted_rand_score",
            visualizations = False, n_components = 2, random_state = 42)
    
    # classification
    X, clusters = make_classification(n_samples=200,
                             n_features=10,
                             n_classes=2)
    
    print("Classification test with " + str(n_clusters) + " clusters: ")
    AutoMF_main(X, clusters, n_clusters, mode = "Supervised", evaluation_method = "adjusted_rand_score",
            visualizations = False, n_components = 2, random_state = 42)

## Unsupervised ##

    # Single Cell Splatter simulated data 2 groups
    df = pd.read_csv("singlecell.tsv", sep = '\t')

    X = df.values
    print("Single cell splatter simulated data test with " + str(2) + " clusters: ")
    AutoMF_main(X, clusters, 2, mode = "Unsupervised", evaluation_method = "silhouette",
            visualizations = False, n_components = 2, random_state = 42)
    
    # Microarray simulations 2 groups

    df = pd.read_csv("microarraysim2.tsv", sep = '\t')
    X = df.values
    print("Microarray simulated data test with " + str(2) + " clusters: ")
    AutoMF_main(X, clusters, 2, mode = "Unsupervised", evaluation_method = "silhouette",
           visualizations = False, n_components = 2, random_state = 42)

    # Iris dataset

    X = datasets.load_iris().data
    clusters = [0] # placeholder for non supervised modes
    print("Iris data test with 3 clusters: ")
    AutoMF_main(X, clusters, 3, mode = "Unsupervised", evaluation_method = "silhouette",
            visualizations = False, n_components = 2, random_state = 42)

## Reconstruction ##

    # NMF synthetic simulated data
    df = pd.read_csv("NMFsim.tsv", sep = '\t')

    X = df.values
    
    if is_invertible(X):
        print("NMF reconstruction simulated data test: ")
        AutoMF_main(X, clusters, mode = "Reconstruction", evaluation_method = "MSE",
            visualizations = False, n_components = 2, random_state = 42)
    


#Main_Test()




def Hyper_test():
    """
    Testing Hyperparameter search module only
    """
        # clusterable dense matrix example
    X, clusters = make_blobs(n_samples=200,
                             n_features=10,
                             centers=2,
                             cluster_std=0.4,
                             shuffle=True)

    X = preprocessing_standarization(X)
    
    method_list = ["PCA_sklearn", "Kernel_PCA",
                "Truncated_SVD", "Incremental_PCA", "IndependentCA",
                "Sparse_PCA", "MiniBatch_SparsePCA", "Factor_Analysis",
                "Dictionary", "t_SNE", "Iso_map", "Local_Linear_Embedding",
                "Multidimensional_scaling", "Spectral_Embedding", "U_MAP"]

    for optimal_model in method_list:
    #Model_optimization(X, clusters, optimal_model, "Reconstruction", "MSE")
        Model_optimization(X, clusters, 2,  optimal_model, "Supervised", "adjusted_rand_score", visualizations = False)
    #Model_optimization(X, clusters, optimal_model, "Unsupervised", "silhouette")
    
    Model_optimization(np.absolute(X), clusters, 2, "NMF_sklearn", "Supervised", "adjusted_rand_score", visualizations = False)
    Model_optimization(np.absolute(X), clusters, 2, "MiniBatch_NMF", "Supervised", "adjusted_rand_score", visualizations = False)
    Model_optimization(np.absolute(X), clusters, 2, "LDA", "Supervised", "adjusted_rand_score", visualizations = False)

Hyper_test()
