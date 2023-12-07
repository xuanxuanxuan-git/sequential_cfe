import numpy as np
import random
import math
import pandas as pd
from sklearn.preprocessing import minmax_scale

def calculate_mean_noise(noise_points, probabilities):
    # if np.sum(probabilities) != 1:
    #     probabilities /= np.sum(probabilities)
    probabilities = np.array([1/len(noise_points)]*len(noise_points))
    noise_points = np.array(noise_points)
    weighted_means = np.sum(noise_points * probabilities[:, np.newaxis], axis=0)

    return weighted_means


def sample_plausible_noise(center, sigma, n_samples, kde):
    np.random.seed(10)
    random.seed(20)
    ## sample from normal distribution
    dim = len(center)
    normal_sampled_points = np.random.multivariate_normal(mean=center, cov=np.eye(dim)*np.square(sigma), size=5*n_samples)
    random.shuffle(normal_sampled_points)

    ## resample noise
    re_distribution = kde.logpdf(normal_sampled_points.T)
    re_distribution = minmax_scale(re_distribution)
    re_distribution /= np.sum(re_distribution)
    # sanity check
    if np.isnan(re_distribution).any():
        small_value = 1e-10
        re_distribution = [small_value if math.isnan(x) else x for x in re_distribution]
        re_distribution /= np.sum(re_distribution)
    re_sampled_points = normal_sampled_points[np.random.choice(len(normal_sampled_points), size=n_samples, p=re_distribution)]

    ## compute new center of the resampled noise
    new_center = calculate_mean_noise(re_sampled_points, kde.pdf(re_sampled_points.T))
    
    return re_sampled_points, new_center

def calculate_ir(noise, model):
    noise_df = pd.DataFrame(noise)
    y_check_prime = model.predict(noise_df)
    # success_indices = (y_check_prime<0.5).nonzero()
    success_time = (y_check_prime<0.5).sum().item()
    inval_rate = success_time/len(noise_df)

    return inval_rate