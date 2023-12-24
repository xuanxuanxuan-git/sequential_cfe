import numpy as np
import torch
# import matplotlib.pyplot as plt
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs
import sys, time, os
import pandas as pd
debug = False
debug2 = False
import cal_metrics
from util import *


def return_counterfactual(obs_original, obs, eval_recurrent_hidden_states, eval_masks, actor_critic, eval_envs, device, episode, env_name, args):
    done = False
    steps = 0
    path = [obs_original[0].copy()]
    max_steps = 200
    if args.eval:
        max_steps = 50
    knn_distances = 0
    env_ = eval_envs.venv.venv.envs[0].env
    if debug2:
        print("Starting: ", obs_original.shape, list(obs_original))

    while (steps < max_steps) and (not done):
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True)

        # Obser reward and next obs
        obs, reward, done, infos, obs_original = eval_envs.step(action)
        if ("german" in env_name) or ("adult" in env_name) or ("default" in env_name) or ("compas" in env_name) or ("heloc" in env_name):
            this_knn_dist = env_.distance_to_closest_k_points(obs_original)
            knn_distances += this_knn_dist
            # print(this_knn_dist, len(path), "KNN dist")
        # print(obs)
        # original = obs * np.sqrt(ob_rms.var + args.eps) + ob_rms.mean

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        if debug2:
            print(obs_original.shape, list(obs_original), "hello", steps, action, reward)
        steps += 1

        if "step" in env_name:
            raise NotImplementedError
            if obs_original[0][0] >= 5.0:
                done = True
            path.append(obs_original[0])

        if done or steps == max_steps:
            if debug:
                lambda_ = env_name.split("v")[-1]
                # with open(f"german_num_steps{lambda_}.txt", "a") as f:
                #     print(f"{steps}", file=f)
                print(f"Episode: {episode}, Steps taken:{steps}")
            # eval_episode_rewards.append(reward)     # append only the last reward. 
            # if not done:
            #     reward = 0
            # break

        path.append(obs_original[0].copy())
    # print("actual actions", env_.action_seq)
    # rollback_action_seq(path[0], env_.action_seq, path[-1])
    # this returns only the last reward, and return the avg knn_distance not sum
    return path, reward, done, knn_distances / len(path), env_.action_seq

def evaluate(actor_critic, ob_rms, env_name, seed, num_processes, eval_log_dir,
             device, args, num_episodes, train_time, env=None):

    global debug
    eval_envs = make_vec_envs(env_name, seed + num_processes, num_processes,
                              None, eval_log_dir, device, True)

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    env_ = eval_envs.venv.venv.envs[0].env
    eval_episode_rewards = []
    episodes = 3
    
    if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name) or ("syndata" in env_name) or ("compas" in env_name) or ("heloc" in env_name):
        episodes = len(env_.undesirable_x)
    episodes = 100
    find_cfs_points = env_.scaler.transform(env_.undesirable_x[:episodes])
    
    trajectories = []
    knn_dist = 0
    correct = 0
    cfs_found = []
    final_cfs = []
    action_seq_all = []

    num_datapoints = episodes
    st = time.time()
    for episode in range(num_datapoints):
        if args.eval:
            os.environ['SEQ'] = f"{episode}"
        obs, obs_original = eval_envs.reset()

        eval_recurrent_hidden_states = torch.zeros(
            num_processes, actor_critic.recurrent_hidden_state_size, device=device)
        eval_masks = torch.zeros(num_processes, 1, device=device)
        path, reward, done, knn_distances, action_seq = return_counterfactual(obs_original, obs, eval_recurrent_hidden_states, eval_masks, actor_critic, eval_envs, device, episode, env_name, args)
        if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name) or ("syndata" in env_name) or ("compas" in env_name) or ("heloc" in env_name):
            # Found a counterfactual successfully. 
            if done:
                try:
                    assert env_.classifier.predict(path[-1].reshape(1, -1))[0] == 1     # a valid CFE
                except:
                    print("CFE is not valid")
                correct += 1
                trajectories.append(len(path))
                knn_dist += knn_distances
                cfs_found.append(True)
                action_seq_all.append(action_seq)
            else:
                cfs_found.append(False)
            final_cfs.append(path[-1])      # there will be problem if we append boolean here
        else:
            trajectories.append(path)
        
        if episode % 1000 == 0:
            print(episode)

        if debug:
            print(reward, knn_distances, episode)
        
        eval_episode_rewards.append(reward)
    
    eval_envs.close()

    if args.eval and ("german" in env_name) or ("adult" in env_name) or ("default" in env_name) or ("syndata" in env_name) or ("compas" in env_name) or ("heloc" in env_name):
        method = "ROSE"
        time_taken = time.time() - st
        print(f"Time: {time.time() - st}")
        lambda_ = env_name.split("v")[-1]
        var = str(lambda_) + "_" + str(args.gamma) + "_" + str(args.num_steps) + "_" + str(args.lr) + "_" + str(args.clip_param) 

        print(f"Setting:{var}, Correct: {correct}")
        if len(trajectories) > 0:
            print(f"Avg. KNN Distance: {knn_dist/correct:.2f}")
            print(f"Avg. path length: {np.mean(np.array(trajectories)):.2f}")

        dataset = env_.scaler.transform(env_.dataset)
        dataset = pd.DataFrame(dataset, columns=env_.dataset.columns.tolist())
        
        if "adult" in env_name:
            name_dataset = "adult"
        elif "german" in env_name:
            name_dataset = "german"
        elif "default" in env_name:
            name_dataset = "default"
        elif "syndata" in env_name:
            name_dataset = "syndata"
        elif "compas" in env_name:
            name_dataset = "compas"
        elif "heloc" in env_name:
            name_dataset = "heloc"

        final_cfs = pd.DataFrame(final_cfs, columns=env_.dataset.columns.tolist())
        find_cfs_points = pd.DataFrame(find_cfs_points, columns=env_.dataset.columns.tolist())

        cal_metrics.calculate_metrics(method + name_dataset, final_cfs, cfs_found, find_cfs_points, env_.classifier, env_.dataset, env_.scaler, var, time_taken, num_episodes, train_time, action_seq_all, env_.test_dataset, save=False)

    print("Evaluation using {} episodes: mean reward {:.5f}\n".format(
            episodes, torch.stack(eval_episode_rewards).mean().item()))
