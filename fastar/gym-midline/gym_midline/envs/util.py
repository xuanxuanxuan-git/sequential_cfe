import numpy as np
import random
import math
import pandas as pd
from sklearn.preprocessing import minmax_scale
from scipy import stats


def calculate_mean_noise(noise_points):
    new_mean = np.average(noise_points, axis=0)
    return new_mean

def sample_plausible_noise(center, sigma, n_samples, kde):
    np.random.seed(10)
    random.seed(20)
    ## sample from normal distribution
    dim = len(center)
    normal_sampled_points = np.random.multivariate_normal(mean=center, cov=np.eye(dim)*np.square(sigma), size=5*n_samples)
    random.shuffle(normal_sampled_points)

    ## resample noise
    re_distribution = kde.pdf(normal_sampled_points.T)
    # re_distribution = minmax_scale(re_distribution)
    re_distribution /= np.sum(re_distribution)
    # sanity check
    if np.isnan(re_distribution).any():
        small_value = 1e-10
        re_distribution = [small_value if math.isnan(x) else x for x in re_distribution]
        re_distribution /= np.sum(re_distribution)
    re_sampled_points = normal_sampled_points[np.random.choice(len(normal_sampled_points), size=n_samples, p=re_distribution)]

    ## compute new center of the resampled noise
    # new_center = calculate_mean_noise(re_sampled_points)
    
    return re_sampled_points

def calculate_mean_fast(center, sigma, n_samples, kde):
    normal_gaussian = stats.multivariate_normal(mean=center, cov=np.eye(len(center))*np.square(sigma))
    points_to_evaluate = np.random.uniform(low=-sigma*3, high=sigma*3, size=(n_samples, len(center))) + center
    kde_values = kde.pdf(points_to_evaluate.T)

    product_values = kde_values * normal_gaussian.pdf(points_to_evaluate)
    new_center = np.average(points_to_evaluate, weights=product_values, axis=0)

    return new_center


def calculate_ir(noise, model):
    noise_df = pd.DataFrame(noise)
    y_check_prime = model.predict(noise_df)
    # success_indices = (y_check_prime<0.5).nonzero()
    success_time = (y_check_prime<0.5).sum().item()
    inval_rate = success_time/len(noise_df)

    return inval_rate