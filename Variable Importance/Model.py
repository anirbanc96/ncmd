import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

np.random.seed(42)

def generate_fully_modular_data(n, theta, d1, d2, a, sigma=0.2):

    X1 = np.random.uniform(-a, a, (n, d1))
    X2 = np.random.uniform(-a, a, (n, d2))
    X3 = np.random.uniform(-a, a, (n, 1))
    epsilon = np.random.normal(0, sigma, (n, 1))

    sum_X1 = np.sum(X1, axis=1, keepdims=True)
    sum_X2 = np.sum(X2, axis=1, keepdims=True)

    # Existing bilinear interaction
    interaction_2way = sum_X1 * X3

    # NEW: Three-way interaction term
    interaction_3way = sum_X1 * sum_X2 * X3

    # Updated model equation
    Y = theta * sum_X1 + theta * sum_X2 + theta * X3 + (2-theta) * interaction_2way + (2-theta) * interaction_3way + epsilon
    return X1, X2, X3, Y.flatten()

def compute_fully_modular_population(theta, d1, d2, a, sigma=0.2):


    v_X = (a**2) / 3.0

    v_f1 = (theta**2) * (d1 * v_X)
    v_f2 = (theta**2) * (d2 * v_X)
    v_f3 = (theta**2) * v_X  
    v_epsilon = sigma**2

    v_f13 = ((2 - theta)**2) * (d1 * v_X) * v_X
    v_f123 = ((2 - theta)**2) * (d1 * v_X) * (d2 * v_X) * v_X 

    v_Y = v_f1 + v_f2 + v_f3 + v_f13 + v_f123 + v_epsilon

    eta1 = v_f1 / v_Y
    eta2 = v_f2 / v_Y
    eta2_13 = v_f13 / v_Y  
    
    return eta1, eta2, eta2_13

def nn_expectation_term(X, Y, K=5):
    n = len(Y)
    nbrs = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree').fit(X)
    _, indices = nbrs.kneighbors(X)
    nn_indices = indices[:, 1:]
    return np.mean(Y * np.sum(Y[nn_indices], axis=1)) / K

def estimate_all_sobol_indices(X1, X2, X3, Y, K=15):
    n = len(Y)
    denominator = np.mean(Y**2) - ((np.sum(Y)**2 - np.sum(Y**2)) / (n * (n - 1)))
    u_v_term = (np.sum(Y)**2 - np.sum(Y**2)) / (n * (n - 1))

    def get_eta(X):
        return (nn_expectation_term(X, Y, K) - u_v_term) / denominator

    hat_eta1 = get_eta(X1)
    hat_eta2 = get_eta(X2)
    hat_eta3 = get_eta(X3)

    X13 = np.hstack((X1, X3))
    hat_eta_joint_13 = get_eta(X13)

    hat_eta2_13 = hat_eta_joint_13 - hat_eta1 - hat_eta3
    return hat_eta1, hat_eta2, hat_eta2_13