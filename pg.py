import numpy as np
from sklearn.kernel_approximation import RBFSampler

rbf_feature = RBFSampler(gamma=1, random_state=12345)

def extract_features(state, num_actions):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """

    # subtract max from vector
    logits_max = np.max(logits)
    logits_stable = logits - logits_max

    # apply softmax formula
    result = np.exp(logits_stable) / np.sum(np.exp(logits_stable), axis=1-axis)

    return result



def compute_action_distribution(theta, phis):
    """ compute probability distrubtion over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: softmax probability distribution over actions (shape 1 x |A|)
    """

    # feature mapping
    feature_map = theta.T @ phis 

    # compute softmax
    return compute_softmax(feature_map, axis=0)


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """

    # compute action distribution
    action_dist = compute_action_distribution(theta, phis) 

    # get expectation
    expectation = phis @ action_dist.T

    # subtract expectation from phi at action
    result = phis[:, [action_idx]] - expectation

    return result


def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the fisher information matrix using the sampled trajectories gradients

    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :param lamb: lambda value used for regularization 

    :return: fisher information matrix (shape d x d)
    
    

    Note: don't forget to take into account that trajectories might have different lengths
    """

    # initialize F_hat, d, N
    d = len(grads[0][0])
    N = len(grads)
    F_hat = np.zeros((d, d))
    
    # loop through trajectories
    for grad_trajectory in grads:
        curr_sum = np.zeros((d, d))
        H = len(grad_trajectory)

        # loop through timesteps in trajectory
        for grad_h in grad_trajectory:
            curr_sum += grad_h @ grad_h.T
        curr_sum = curr_sum / H 
        F_hat = F_hat + curr_sum 
    
    # compute final F_hat
    F_hat = F_hat / N + lamb * np.identity(d)
    return F_hat

def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards

    :param grads: ist of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)
    """

    d = len(grads[0][0])
    N = len(grads)
    PG_est = np.zeros((d, 1))

    b = sum([sum(i) for i in rewards]) / N

    for t, grad_trajectory in enumerate(grads):
        curr_sum = np.zeros((d, 1))
        H = len(grad_trajectory)

        # loop through timesteps in trajectory
        for h, grad_h in enumerate(grad_trajectory):
            reward_remain = sum(rewards[t][h:])
            curr_sum += grad_h * (reward_remain - b)
        curr_sum = curr_sum / H 
        PG_est = PG_est + curr_sum 

    # compute final PG estimate
    PG_est = PG_est / N
    return PG_est

def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent

    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """

    epsilon = 10e-6

    # calculate denominator
    denom = v_grad.T @ np.linalg.inv(fisher) @ v_grad + epsilon

    # calculate and return eta
    eta = np.sqrt(delta / denom)
    return eta

