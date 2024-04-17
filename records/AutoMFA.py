from sklearn.decomposition import NMF, PCA, SparsePCA, KernelPCA, TruncatedSVD, IncrementalPCA, FastICA, MiniBatchSparsePCA, MiniBatchNMF
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
    #X_sparse = csr_matrix(X)

    model = SparsePCA(n_components=2)
    W = model.fit_transform(X)
    
    X_r = model.inverse_transform(W)
    
    return W, X_r

# Kernel PCA

def Kernel_PCA(X):

    model = KernelPCA(n_components=2, kernel='linear', fit_inverse_transform = True)
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
    model = IncrementalPCA(n_components=2, batch_size = 10)
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
    model = MiniBatchSparsePCA(n_components = 2)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r

def MiniBatch_NMF(X):
    model = MiniBatchNMF(n_components = 2)
    W = model.fit_transform(X)

    X_r = model.inverse_transform(W)

    return W, X_r

from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_blobs, make_sparse_spd_matrix
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.metrics import rand_score, adjusted_rand_score, mean_squared_error, silhouette_score, fowlkes_mallows_score, calinski_harabasz_score
import warnings
warnings.filterwarnings("ignore")
import pandas as pd

###############
# Main Wrappers
###############

def Initial_Guess(X_scaled, method_list):
    '''
    Takes standarized/normalizaed matrix X_scaled, number of iterations MFiters, and
    list of methods to scan for optimal initial guess based on inverse transformation reconstruction
    mean squared error.
    '''
    global_method_list = []
    global_error_list = []

    for method in method_list:
        
        if method == "PCA_sklearn":

            X, X_r = PCA_sklearn(X_scaled)
        
            mse = mean_squared_error(X_scaled, X_r)


        if method == "NMF_sklearn":
            
            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = NMF_sklearn(X_scaled)

            mse = mean_squared_error(X_scaled, X_r)

        if method == "NMF_sparse":

            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = NMF_sparse(X_scaled)

            mse = mean_squared_error(X_scaled, X_r)
    
        if method == "MiniBatch_NMF":
            
            X_scaled = X_scaled + np.absolute(X_scaled.min())

            X, X_r = MiniBatch_NMF(X_scaled)

            mse = mean_squared_error(X_scaled, X_r)
         
        if method == "Sparse_PCA":

            X, X_r = Sparse_PCA(X_scaled)

            mse = mean_squared_error(X_scaled, X_r)
 
        if method == "Kernel_PCA":

            X, X_r = Kernel_PCA(X_scaled)
 
            mse = mean_squared_error(X_scaled, X_r)
        
        if method == "Truncated_SVD":

            X, X_r =  Truncated_SVD(X_scaled)

            mse = mean_squared_error(X_scaled, X_r)
 
        if method == "Incremental_PCA":

            X, X_r = Incremental_PCA(X_scaled)

            mse = mean_squared_error(X_scaled, X_r)
 
        if method == "IndependentCA":

            X, X_r = IndependentCA(X_scaled)
 
            mse = mean_squared_error(X_scaled, X_r)
        
        if method == "MiniBatch_SparsePCA":

            X, X_r = MiniBatch_SparsePCA(X_scaled)

            mse = mean_squared_error(X_scaled, X_r)
 
        global_method_list.append(method)
        global_error_list.append(mse)
   
        print(method + " evaluation metrics: ")
    
        print("Mean Squared error...")
        print(mse)
    
    df = pd.DataFrame()
    df['method'] = global_method_list
    df['error'] = global_error_list
  
    most_optimal_model = df[df.error == df.error.min()]
    
    print("Best initial guess is...")
    print(most_optimal_model)

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


def Model_auto_select(X):
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
        
        Initial_Guess(X_scaled, method_list)
    else: 

        method_list = ["PCA_sklearn", "NMF_sklearn", "Kernel_PCA",
                "Truncated_SVD", "Incremental_PCA", "IndependentCA", "MiniBatch_NMF"]
        #method_list = ["PCA_sklearn", "NMF_sklearn"]
        X_scaled = preprocessing_standarization(X)

        Initial_Guess(X_scaled, method_list)
    return


# For supervised classification gridsearch 
def Model_optimization(optimal_model,X):
    
    if optimal_model == "PCA":
        n_components_list = [2,3,4,5]
        svd_solver_list = ['auto', 'full', 'arpack', 'randomized']
        cmse = np.zeros((len(n_components_list), len(svd_solver_list)))
        
        for i in range(len(n_components_list)):
            for j in range(len(svd_solver_list)):

                model = PCA(n_components = n_components_list[i], svd_solver = svd_solver_list[j])
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                mse = mean_squared_error(X, X_r)
                
                cmse[i][j] = mse
        
        min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        
        o_model = PCA(n_components = n_components_list[min_index[0]], svd_solver = svd_solver_list[min_index[1]])

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)
        
        mse = mean_squared_error(X, oX_r)

        print("Best parameters found ...")
        
        print(mse)

    if optimal_model == "NMF":
        
        X = X + np.absolute(X.min())

        n_components_list = [2,3,4,5]
        init_list = ['random', 'nndsvd', 'nndsvda', 'nndsvdar', None]

        cmse = np.zeros((len(n_components_list), len(init_list)))
        
        for i in range(len(n_components_list)):
            for j in range(len(init_list)):

                model = NMF(n_components = n_components_list[i], init = init_list[j])
                W = model.fit_transform(X)

                X_r = model.inverse_transform(W)

                mse = mean_squared_error(X, X_r)
                
                cmse[i][j] = mse
        
        min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        
        o_model = NMF(n_components = n_components_list[min_index[0]], init = init_list[min_index[1]])

        oW = o_model.fit_transform(X)

        oX_r = o_model.inverse_transform(oW)
        
        mse = mean_squared_error(X, oX_r)

        print("Best parameters found ...")
        
        print(mse)


    if optimal_model == "NMF_Sparse":
        X = X + np.absolute(X.min())
        X_sparse = csr_matrix(X)
        n_components_list = [2,3,4,5]
        init_list = ['random', 'nndsvd', 'nndsvda', 'nndsvdar', None]

        cmse = np.zeros((len(n_components_list), len(init_list)))
        
        for i in range(len(n_components_list)):
            for j in range(len(init_list)):

                model = NMF(n_components = n_components_list[i], init = init_list[j])
                W = model.fit_transform(X_sparse)

                X_r = model.inverse_transform(W)

                mse = mean_squared_error(X, X_r)
                
                cmse[i][j] = mse
        
        min_index = np.unravel_index(np.argmin(cmse, axis=None), cmse.shape)
        
        o_model = NMF(n_components = n_components_list[min_index[0]], init = init_list[min_index[1]])

        oW = o_model.fit_transform(X_sparse)

        oX_r = o_model.inverse_transform(oW)
        
        mse = mean_squared_error(X, oX_r)

        print("Best parameters found ...")
        
        print(mse)


    return


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



# Main test
def maintest():
    """
    working test
    """
# clusterable dense matrix example 
    X, clusters = make_blobs(n_samples = 2000,
                  n_features = 10,
                  centers = 2,
                  cluster_std = 0.4,
                  shuffle = True)

    Model_auto_select(X)
    
    # Assume best model was PCA
    print("Best model hyperparameter tuning starting ...")
    X_scaled = preprocessing_standarization(X)
    Model_optimization("PCA", X_scaled)

    # Assume best model was NMF
    print("Best model hyperparameter tuning starting ...")
    Model_optimization("NMF", X_scaled)
# Sparse toy matrix to test sparsity check
    X = make_sparse_spd_matrix(100, alpha = .98, smallest_coef = .4, largest_coef=.8, random_state = 42)


    Model_auto_select(X)
    
    # Assume best model was NMF Sparse
    print("Best sparse model hyperparameter tuning starting ...")
    X_norm = normalize(X, norm='l1', axis=1)
    
    Model_optimization("NMF_Sparse", X_norm)
    return

from sklearn import datasets
from sklearn.feature_extraction.text import TfidfVectorizer
def Simulateddata_maintest():
    """
    AutoMF test for PCA simulated data, should find optimal model PCA
    """
    # PCA simulated naive reconstruction
    X = PCA_simulated()

    method_list = ["PCA_sklearn", "NMF_sklearn", "Kernel_PCA",
                "Truncated_SVD", "Incremental_PCA", "IndependentCA", "MiniBatch_NMF"]

    Initial_Guess(X, method_list)
    
    # Iris data set
    X = datasets.load_iris().data

    Initial_Guess(X, method_list)
    
    # Microarray data https://cran.r-project.org/web/packages/madsim/madsim.pdf
    # According to : Constructing Endophenotypes of Complex Diseases Using Non-Negative Matrix Factorization and Adjusted Rand Index
    # NMF has higher purity
    df = pd.read_csv("microarraysim.tsv", sep="\t")   # read dummy .tsv file into memory

    X = df.values  # access the numpy array containing values
    
    Initial_Guess(X, method_list)

    df = pd.read_csv("microarraysim2.tsv", sep = '\t')

    X = df.values

    Initial_Guess(X, method_list)


    df = pd.read_csv("NMFsim.tsv", sep = '\t')

    X = df.values

    Initial_Guess(X, method_list)
    # Assume best model was PCA
    #print("Best model hyperparameter tuning starting ...")
    #X_scaled = preprocessing_standarization(X)
    #Model_optimization("PCA", X_scaled)

def Simulateddata_maintest2():
    """
    Fix sparsity first
    """
    documents = pd.read_csv('news-data.csv')

    vect = TfidfVectorizer(min_df=500, stop_words='english')

# Fit and transform
    X = vect.fit_transform(documents.headline_text)

    print(X.shape)
    method_list = ['NMF_sparse', 'Sparse_PCA', "MiniBatch_SparsePCA"]

    Initial_Guess(X.toarray(), method_list)
 

Simulateddata_maintest()
#maintest()
