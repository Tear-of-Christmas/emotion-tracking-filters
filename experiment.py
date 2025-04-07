import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from models import kalman_filter, bayesian_update, moving_average

# Generate test data
def generate_emotion_data(seed=42):
    np.random.seed(seed)
    low_range = np.random.uniform(25, 35, size=3)
    normal_range = np.random.uniform(50, 70, size=4)
    true_emotion = np.concatenate([low_range, normal_range])
    observed_emotion = true_emotion + np.random.normal(0, 3, len(true_emotion))
    return true_emotion, observed_emotion

# Run experiment
def run_experiment():
    true_emotion, observed_emotion = generate_emotion_data()

    kalman = kalman_filter(observed_emotion)
    bayesian = bayesian_update(observed_emotion)
    moving_avg = moving_average(observed_emotion, window=3)

    mse_kalman = mean_squared_error(true_emotion, kalman)
    mse_bayesian = mean_squared_error(true_emotion, bayesian)
    mse_moving_avg = mean_squared_error(true_emotion, moving_avg)

    r_kalman, _ = pearsonr(true_emotion, kalman)
    r_bayesian, _ = pearsonr(true_emotion, bayesian)
    r_moving_avg, _ = pearsonr(true_emotion, moving_avg)

    plt.figure(figsize=(10, 5))
    plt.plot(true_emotion, label='True Emotion', linestyle='--', color='black', linewidth=2)
    plt.plot(observed_emotion, label='Observed Emotion', alpha=0.3, linestyle=':', color='gray')
    plt.plot(kalman, label='Kalman Filter', linewidth=2, color='orange')
    plt.plot(bayesian, label='Bayesian Update', linewidth=2, color='purple')
    plt.plot(moving_avg, label='Moving Average', linewidth=2, color='green')
    plt.title("7-Day Emotion Tracking: Kalman vs Bayesian vs Moving Avg")
    plt.xlabel("Time Step")
    plt.ylabel("Emotion Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("figures/emotion-tracking-example.png")
    plt.show()

    print("\nMSE and Pearson Correlation Results")
    print(f"Kalman Filter     - MSE: {mse_kalman:.2f}, r: {r_kalman:.3f}")
    print(f"Bayesian Update   - MSE: {mse_bayesian:.2f}, r: {r_bayesian:.3f}")
    print(f"Moving Average    - MSE: {mse_moving_avg:.2f}, r: {r_moving_avg:.3f}")

if __name__ == "__main__":
    run_experiment()
