import gym, torch
import numpy as np
import random, copy, os, sys
from sklearn.neighbors import NearestNeighbors
from scipy import stats

sys.path.append("../../../../")
sys.path.append("../")
sys.path.append("../../")
import classifier_dataset as classifier
from util import *

class GermanCredit(gym.Env):
    metadata = {"render.modes": ["human"]}
    """ A custom OpenAI gym for the reduced version of German Credit dataset """

    def __init__(self, dist_lambda):
        super(GermanCredit, self).__init__()
        file1 = (
            f"{os.path.dirname(os.path.realpath(__file__))}/german_redone.csv"
        )
        clf, dataset, scaler, X_test, X_train = classifier.train_model_german(
            file=file1, parameter=1
        )
        # Discrete action space
        self.action_space = gym.spaces.Discrete(2 * len(dataset.columns))

        low = np.ones(shape=len(dataset.columns)) * -1.0
        high = np.ones(shape=len(dataset.columns))
        self.observation_space = gym.spaces.Box(
            low=low, high=high, dtype=np.float64
        )
        self.state = None
        self.dist_lambda = dist_lambda
        self.immutable_features = [
            "Personal-status",
            "Number-of-people-being-lible",
            "Foreign-worker",
            "Purpose",
        ]
        self.dataset = dataset
        self.train_dataset = scaler.transform(X_train)
        self.state_count = 0
        self.scaler = scaler
        self.classifier = clf
        self.states = {}
        self.states_reverse = {}
        self.no_neighbours = 1
        self.knn_lambda = dist_lambda
        self.knn = NearestNeighbors(n_neighbors=5, p=1)
        self.knn.fit(scaler.transform(self.dataset))
        self.numerical_features = [1, 4, 7, 10, 12, 15, 17]
        self.seq = -1
        os.environ["SEQ"] = "-1"
        self.undesirable_x = []
        try:
            self.undesirable_x = np.load(
                f"{os.path.dirname(os.path.realpath(__file__))}/../../../datapoints_to_generate_cfes/undesirable_x_german.npy"
            )
            print("Found")
        except:
            undesirable_x = []
            for no, i in enumerate(self.dataset.to_numpy()):
                if (
                    self.classifier.predict(
                        self.scaler.transform(i.reshape(1, -1))
                    )
                    == 0
                ):
                    undesirable_x.append(tuple(i))
            self.undesirable_x = np.array(undesirable_x)
            # np.save(f"{os.path.dirname(os.path.realpath(__file__))}/../../../datapoints_to_generate_cfes/undesirable_x_german.npy", undesirable_x)

        print(
            len(self.undesirable_x), "Total datapoints to run the approach on"
        )
        self.kde = stats.gaussian_kde(scaler.transform(dataset).T)
        self.action_seq = []
        self.reset()

    # def model(self):
    #     # The probability of belonging to class 1 (the desired class)
    #     probability_class1 = self.classifier.predict_proba(
    #         self.state.reshape(1, -1)
    #     )[0][1]
    #     # print("resulting state:", self.state, probability_class1)
    #     # If the probability of belonging to the desired class is greater than 0.5, then it is a valid CFE.
    #     if probability_class1 >= 0.5:
    #         return 100, True
        
    #     return probability_class1, False
 
    def model(self):
        # The probability of belonging to class 1 (the desired class)
        probability_class1 = self.classifier.predict_proba(
            self.state.reshape(1, -1)
        )[0][1]
        
        threshold = 75
        if probability_class1 >= 0.5:
            next_state_noise = sample_plausible_noise_train(self.state, sigma=0.1, n_samples=400, kde=self.kde)
            # next_state_noise = sample_gaussian_noise(self.state, sigma=0.1, n_samples=100)
            noise_air = calculate_ir(next_state_noise, self.classifier)

            reward = 100 - noise_air*100
            if reward >= threshold:
                return reward, True
            else: 
                return probability_class1, False
        
        return probability_class1, False
    
    def shift_mean(self, center):

        # It will shift the center every one/two/four steps
        # if self.noise_fre % 2 == 0:
        #     return np.array(center)
        # 0.0005 (0.022), 0.00025 (0.02)
        # 0.0006 (0.025), 0.001 (0.03), 0.0014 (0.037), 0.002 (0.045)
        next_state_center = calculate_mean_fast(center, sigma=0.03, n_samples=50, kde=self.kde)
        return next_state_center
    
    
    def step(self, action):

        # comment off if using CPU
        # if not isinstance(action, int) and len(action) == 1:
        #     action = action[0]

        if isinstance(action, torch.Tensor):
            action = action.numpy()[0][0]
            assert isinstance(action, (int, np.int64))
            type_ = 1

        elif isinstance(action, np.ndarray):
            type_ = 2

        elif isinstance(action, (int, np.int64)):
            type_ = 1

        else:
            raise NotImplementedError

        info = {}

        if type_ == 1:
            feature_changing = (
                action // 2
            )  # this is the feature that is changing
            decrease = bool(action % 2)
            if decrease:
                amount = -0.05
            else:
                amount = 0.05

        elif type_ == 2:
            decrease = False
            amount = np.clip(
                action[0], self.action_space.low[0], self.action_space.high[0]
            )
            if amount < 0:
                decrease = True
            feature = np.clip(
                action[1], self.action_space.low[1], self.action_space.high[1]
            )
            feature += 1  # casts in 0 to 2 range
            feature_changing = int(
                feature * (len(self.dataset.columns) // 2)
            )  # we need int not round

        else:
            assert False

        reward = -10
        done = False

        for imf in self.immutable_features:
            if imf in self.dataset.iloc[:, feature_changing].name:
                return self.state, reward, done, info

        # age can't decrease
        if self.dataset.iloc[:, feature_changing].name == "age" and decrease:
            return self.state, reward, done, info

        # Job can't decrease
        elif self.dataset.iloc[:, feature_changing].name == "Job" and decrease:
            return self.state, reward, done, info

        action_ = amount
        next_state = list(copy.deepcopy(self.state))
        next_state[feature_changing] = self.state[feature_changing] + action_
        knn_dist_loss = self.knn_lambda * self.distance_to_closest_k_points(
            next_state
        )
        assert knn_dist_loss >= 0
        knn_dist_loss = 0
        constant = 0  # constant loss for each action

        if decrease:
            if (
                next_state[feature_changing] > -1.0
            ):  # lowest value for a feature is -1.0
                self.state = np.array(next_state)
                reward, done = self.model()
                if not done:
                    self.state = self.shift_mean(next_state)
                reward = (
                    reward - constant - knn_dist_loss
                )  # constant cost for each action
                self.action_seq.append((feature_changing, action_))
            else:
                reward = (
                    -10
                )  # This is the reward in the case of an incorrect action.
                done = False
        else:
            if next_state[feature_changing] < 1.0:  # highest value possible
                self.state = np.array(
                    next_state
                )  # change self.state only if next_state is valid
                reward, done = self.model()
                if not done:
                    self.state = self.shift_mean(next_state)
                reward = reward - constant - knn_dist_loss
                self.action_seq.append((feature_changing, action_))
            else:
                reward = -10
                done = False

        # print(self.action_seq)
        return self.state, reward, done, info

    def distance_to_closest_k_points(self, state):
        state = np.array([state]).reshape(1, -1)
        nearest_dist, nearest_points = self.knn.kneighbors(
            state, self.no_neighbours, return_distance=True
        )
        return np.mean(nearest_dist)

    def reset(self):
        seq = int(os.environ["SEQ"])
        self.action_seq = []
        if len(self.undesirable_x) == 0:
            return
        # This is used during training of the agent.
        if seq == -1:
            idx = random.randrange(self.train_dataset.shape[0])
            self.state = self.train_dataset[idx]
        # This is used during evaluation of a trained agent
        else:
            self.state = self.scaler.transform(
                np.array(self.undesirable_x[seq]).reshape(1, -1)
            )[0]
        return self.state

    def render(self, mode="human", close=False):
        print(f"State: {self.state}")


class GermanCredit01nr(GermanCredit):
    def __init__(self, enable_render=True):
        super(GermanCredit01nr, self).__init__(dist_lambda=0.1)


if __name__ == "__main__":
    x = GermanCredit01nr()
    print(x.step(1))
    print(x.step(5))
