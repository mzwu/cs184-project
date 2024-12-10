import numpy as np
import utils
import pg
import matplotlib.pyplot as plt

def sample(theta, N):
    """ samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 100 steps
    """
    total_rewards = []
    total_grads = []

    # get initial observation
    state = utils.starting_position()
    col_len = state.shape[0]
    state.reshape((col_len, 1))
    num_actions = 80

    # loop through each trajectory
    for i in range(N):
        curr_rewards = []
        curr_grads = []

        for j in range(100):
            # sample action
            phis = pg.extract_features(state, num_actions)
            act_dist = pg.compute_action_distribution(theta, phis)
            action = np.random.choice([*range(80)], 1, p=act_dist[0])

            # get state and reward from taking action
            state, cleared, done = utils.transition(state, action)
            state.reshape((col_len, 1))
            curr_rewards.append((-1 * j - 1)) # should be reward

            # calculate and append gradient
            gradient = pg.compute_log_softmax_grad(theta, phis, action)
            curr_grads.append(gradient)

            # go to next trajectory iteration
            if done:
                state = utils.starting_position()
                state.reshape((col_len, 1))
                break 

        total_rewards.append(curr_rewards)
        total_grads.append(curr_grads)

    return total_grads, total_rewards


def train(N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(4,1)

    episode_rewards = []

    # loop through iterations
    for i in range(T):
        # sample gradients and rewards
        gradients, rewards = sample(theta, N)

        # calculate average reward
        avg_reward = sum([sum(traj) for traj in rewards]) / N

        # print current metrics
        print("iteration:", i)
        print(avg_reward)

        # do natural policy gradient
        episode_rewards.append(avg_reward)
        fisher = pg.compute_fisher_matrix(gradients, lamb)
        v_grad = pg.compute_value_gradient(gradients, rewards)
        eta = pg.compute_eta(delta, fisher, v_grad)
        theta = theta + eta * np.linalg.inv(fisher) @ v_grad

    return theta, episode_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()