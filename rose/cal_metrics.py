import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler
import sys, os, copy
from scipy import stats
import importlib
from util import *

def find_air(cfe, sigma, n_samples, kde, model):
    
    if not isinstance(cfe, np.ndarray):
        cfe = cfe.iloc[0].tolist()
    sampled_noise_pl = sample_plausible_noise(cfe, sigma, n_samples, kde)
    sampled_noise_ga = sample_gaussian_noise(cfe, sigma, n_samples)

    pl_air = calculate_ir(sampled_noise_pl, model)
    ga_air = calculate_ir(sampled_noise_ga, model)

    return pl_air, ga_air


def noisy_implementation(factual_point, action_seq, kde, model, sigma, freq):
    current_state = list(copy.deepcopy(factual_point.iloc[0]))
    action_seq = action_seq[0]
    for (feature, amount) in action_seq:
        current_state[feature] = current_state[feature] + amount
        current_state = sample_plausible_noise(current_state, sigma, 1, kde)[0]
    if model.predict(np.array(current_state).reshape(1,-1)) < 0.5:
        return False
    else:
        return True


def repeat_noisy_implementation(factual_point, action_seq, kde, model, sigma, freq, times = 1000):
    ir = 0
    for i in range(times):
        result = noisy_implementation(factual_point, action_seq, kde, model, sigma, freq)
        if not result:
            ir += 1   
    air = ir/times
    return air

def calculate_dist(dataset, factual_point, action_seq):

    current_state = list(copy.deepcopy(factual_point.iloc[0]))
    action_seq = action_seq[0]
    for (feature, amount) in action_seq:
        current_state[feature] = current_state[feature] + amount
    
    cf_point = np.array(current_state).reshape(1, -1)
    diff = (cf_point[0] - factual_point.iloc[0].values)
    l0_dist = np.count_nonzero(diff)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(dataset)
    scaled_factual = scaler.transform(factual_point)
    scaled_cf = scaler.transform(cf_point)

    l1_dist = np.linalg.norm((scaled_cf-scaled_factual), ord=1, axis=1)
    return l1_dist, l0_dist


def calculate_metrics(method, final_cfs, cfs_found, find_cfs_points, model, dataset, scaler, setting, time_taken, num_episodes, train_time, action_seq_all, density_dataset, save=False):
    
    # noise_type = "non"
    # noise_type = "pla"
    noise_type = "pth"

    # scaled_dataset is test dataset if adult, or train dataset if heloc
    if ("adult" in method):
        scaled_dataset = density_dataset
    else:
        scaled_dataset = scaler.transform(dataset)

    avg_sparsity = []
    computed_cfs = []
    avg_pl_air = []
    avg_ga_air =[]
    avg_dist = []
    avg_noise_rate = []
    count = 0
    step_sigma = 0.001
    print("step sigma: ", step_sigma)

    for seq, dt in enumerate(cfs_found):
        if dt:  # find the metrics only if a cfe was found for a datapoint it was requested for. 
            cfe = final_cfs[seq:seq+1]
            computed_cfs.append(cfe.to_numpy())
            cfe_prediction = model.predict(cfe)[0]
            original_datapoint = find_cfs_points[seq: seq+1]
            original_prediction = model.predict(original_datapoint)[0]
            try:
                assert cfe_prediction != original_prediction
            except:
                pass
        
            pl_air, ga_air = find_air(cfe, sigma=np.sqrt(0.01), n_samples=1000, kde=stats.gaussian_kde(scaled_dataset.T), model=model)
            action_seq = action_seq_all[count:count+1]
            # 0.001, 0.0005, 0.00025 different u
            # 0.0006, (0.001), 0.0014, 0.002
            noise_rate = repeat_noisy_implementation(original_datapoint, action_seq, kde=stats.gaussian_kde
            (scaled_dataset.T), model=model, sigma=np.sqrt(step_sigma), freq=1, times=100)
            avg_noise_rate.append(noise_rate)
            count += 1

            l1_dist, sparsity = calculate_dist(scaled_dataset, original_datapoint, action_seq)
            avg_dist.append(l1_dist)
            
            avg_sparsity.append(sparsity)
            avg_pl_air.append(pl_air)
            avg_ga_air.append(ga_air)
    
    print("Gaussian AIR: {}, std: {}".format(round(np.mean(avg_ga_air), 2), np.std(avg_ga_air)))
    print("Plausible AIR: {}, std: {}".format(round(np.mean(avg_pl_air), 2), np.std(avg_pl_air)))
    print("Path variation noise rate: {}, std: {}".format(round(np.mean(avg_noise_rate), 2), np.std(avg_noise_rate)))
    print("Normalised l1 dist: {}, std: {}".format(round(np.mean(avg_dist), 2), np.std(avg_dist)))
    print("Sparsity: {}, std: {}".format(round(np.mean(avg_sparsity), 2), np.std(avg_sparsity)))
    
    validity = sum(cfs_found) * 100.0 / find_cfs_points.shape[0]
    
    # Header: setting,efficacy,sparsity,l1,time,gaussian,plausible,path
    file = f"output/results/all_metrics_{method}.csv"
    if not os.path.exists(file):
        with open(file, "w") as f:
            print("setting,num_episodes,noise,train_time,efficacy,sparsity,l1,time,gaussian,plausible,path", file=f)
    with open(file, "a") as f:
        print(setting, num_episodes, noise_type, round(train_time, 2), round(validity, 2), 
              f"{round(np.mean(avg_sparsity), 2)}({round(np.std(avg_sparsity), 2)})", 
              f"{round(np.mean(avg_dist), 2)}({round(np.std(avg_dist), 2)})", 
              round(time_taken, 2), 
              f"{round(np.mean(avg_ga_air), 2)}({round(np.std(avg_ga_air), 2)})", 
              f"{round(np.mean(avg_pl_air), 2)}({round(np.std(avg_pl_air), 2)})", 
              f"{round(np.mean(avg_noise_rate), 2)}({round(np.std(avg_noise_rate), 2)})", sep=',', file=f)
