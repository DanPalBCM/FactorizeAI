import numpy as np
from matplotlib import pyplot as plt



from ConfigSpace import Configuration, ConfigurationSpace, Float, Integer

from sklearn.decomposition import PCA
from smac import HyperparameterOptimizationFacade as HPOFacade
from smac import RunHistory, Scenario

from smac import HyperparameterOptimizationFacade, Scenario


from sklearn.datasets import make_blobs, make_sparse_spd_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import rand_score, adjusted_rand_score, mean_squared_error, silhouette_score, \
    fowlkes_mallows_score, calinski_harabasz_score, adjusted_mutual_info_score, homogeneity_completeness_v_measure, davies_bouldin_score

from smac import HyperparameterOptimizationFacade, Scenario

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

#X = test matrix
#Y = output matrix after MF

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
    
def PCA_sklearn(X, **kwargs):
    model = PCA(**kwargs)
    W = model.fit_transform(X)
    
    X_r = model.inverse_transform(W)

    return W, X_r
    
    
class PCA_MF:
    @property
    def configspace(self) -> ConfigurationSpace:
        cs = ConfigurationSpace(seed=0)
        n_components = Integer("n_components", (2, 5), default=4)
        cs.add_hyperparameters([n_components])

        return cs

    def train(self, config: Configuration, seed: int = 0) -> float:
    
        X, clusters = make_blobs(n_samples=200,
                             n_features=10,
                             centers=2,
                             cluster_std=0.4,
                             shuffle=True,
                             random_state = 42)
                             
        W, X_r = PCA_sklearn(X, random_state = 42)
        labels = [0]
        clusters = [0]
        evaluation = Evaluation_metrics(X,X_r,X, labels, clusters, "Reconstruction", "MSE")
        
        return evaluation

#evaluation will be minimized 


                             
if __name__ == "__main__":
    model = PCA_MF()

    # Scenario object specifying the optimization "environment"
    scenario = Scenario(model.configspace, n_trials=10)
    
    initial_design = HyperparameterOptimizationFacade.get_initial_design(scenario, n_configs=5)

    # Now we use SMAC to find the best hyperparameters
    smac = HyperparameterOptimizationFacade(
        scenario,
        model.train,
        initial_design=initial_design,
        overwrite=True,  # If the run exists, we overwrite it; alternatively, we can continue from last state
    )

    incumbent = smac.optimize()

    print(incumbent)
