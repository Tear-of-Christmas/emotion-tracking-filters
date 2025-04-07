import numpy as np

# Kalman Filter
def kalman_filter(observed, Q=2.0, R=4.0):
    n = len(observed)
    x = np.zeros(n)
    P = 1.0
    x[0] = observed[0]
    for t in range(1, n):
        x_pred = x[t - 1]
        P_pred = P + Q
        K = P_pred / (P_pred + R)
        x[t] = x_pred + K * (observed[t] - x_pred)
        P = (1 - K) * P_pred
    return x

# Bayesian Updating
def bayesian_update(observed, prior_mean=50.0, prior_var=10.0, obs_var=9.0):
    means = []
    post_mean = prior_mean
    post_var = prior_var
    for z in observed:
        weight = post_var / (post_var + obs_var)
        post_mean = post_mean + weight * (z - post_mean)
        post_var = (post_var * obs_var) / (post_var + obs_var)
        means.append(post_mean)
    return np.array(means)

# Moving Average
def moving_average(observed, window=3):
    ma = []
    for i in range(len(observed)):
        if i < window:
            ma.append(np.mean(observed[:i+1]))
        else:
            ma.append(np.mean(observed[i-window+1:i+1]))
    return np.array(ma)
