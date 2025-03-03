import os
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size=50):
    """Compute the moving average with explicit padding at the edges."""
    pad_size = window_size // 2  # Half window size for symmetric padding
    padded_data = np.pad(data, pad_size, mode='edge')  # Extend edge values
    return np.convolve(padded_data, np.ones(window_size)/window_size, mode='valid')

class RLAnalyzer:
    def __init__(self, rl_algorithms, save_path="plots"):
        """
        Initializes the analyzer with multiple RL algorithm instances.
        
        Args:
            rl_algorithms (dict): Dictionary of {"algorithm_name": RLClass_instance}
            save_path (str): Directory to save plots
        """
        self.rl_algorithms = rl_algorithms
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def plot_individual_performance(self, window_size=50):
        """
        Plots individual performances for each RL algorithm in a single figure with two subplots.
        Uses moving average for smoothing.
        """
        for rl_instance in self.rl_algorithms:
            name = rl_instance.algo_type
            fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # 3 rows, 1 column

            # Apply moving average
            smoothed_steps = moving_average(rl_instance.episode_steps, window_size)
            smoothed_rewards = moving_average(rl_instance.episode_rewards, window_size)
            smoothed_average_rewards = moving_average(rl_instance.episode_average_rewards, window_size)
            episodes = np.arange(len(smoothed_steps))  # Adjusted episode indices

            # Steps per episode
            axes[0].plot(episodes, smoothed_steps, label=f"Steps (Window={window_size})", color="blue")
            axes[0].set_xlabel("Episode")
            axes[0].set_ylabel("Steps")
            axes[0].set_title(f"{name} - Steps per Episode (Smoothed)")
            axes[0].legend()
            axes[0].grid()

            # Total reward per episode
            axes[1].plot(episodes, smoothed_rewards, label=f"Reward (Window={window_size})", color="red")
            axes[1].set_xlabel("Episode")
            axes[1].set_ylabel("Total Reward")
            axes[1].set_title(f"{name} - Total Reward per Episode (Smoothed)")
            axes[1].legend()
            axes[1].grid()

            # Average reward per episode
            axes[2].plot(episodes, smoothed_average_rewards, label=f"Reward (Window={window_size})", color="red")
            axes[2].set_xlabel("Episode")
            axes[2].set_ylabel("Average Reward")
            axes[2].set_title(f"{name} - Average Reward per Episode (Smoothed)")
            axes[2].legend()
            axes[2].grid()

            plt.tight_layout()
            plt.savefig(f"{self.save_path}/{name}_performance.png")
            plt.close()

    def plot_comparison(self, window_size=50):
        """
        Plots comparison of all RL algorithms in the same figure with two subplots.
        Uses moving average for smoothing.
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))  # 3 rows, 1 column

        # Steps per episode
        for rl_instance in self.rl_algorithms:
            name = rl_instance.algo_type
            smoothed_steps = moving_average(rl_instance.episode_steps, window_size)
            episodes = np.arange(len(smoothed_steps))
            axes[0].plot(episodes, smoothed_steps, label=f"{name} Steps (Window={window_size})")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Steps per Episode")
        axes[0].set_title("Comparison of Steps per Episode (Smoothed)")
        axes[0].legend()
        axes[0].grid()

        # Total reward per episode
        for rl_instance in self.rl_algorithms:
            name = rl_instance.algo_type
            smoothed_total_rewards = moving_average(rl_instance.episode_rewards, window_size)
            episodes = np.arange(len(smoothed_total_rewards))
            axes[1].plot(episodes, smoothed_total_rewards, label=f"{name} Reward (Window={window_size})")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Average Reward per Episode")
        axes[1].set_title("Comparison of Total Rewards per Episode (Smoothed)")
        axes[1].legend()
        axes[1].grid()

        # Average reward per episode
        for rl_instance in self.rl_algorithms:
            name = rl_instance.algo_type
            smoothed_average_rewards = moving_average(rl_instance.episode_average_rewards, window_size)
            episodes = np.arange(len(smoothed_average_rewards))
            axes[2].plot(episodes, smoothed_average_rewards, label=f"{name} Reward (Window={window_size})")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("Average Reward per Episode")
        axes[2].set_title("Comparison of Average Rewards per Episode (Smoothed)")
        axes[2].legend()
        axes[2].grid()

        plt.tight_layout()
        plt.savefig(f"{self.save_path}/comparison.png")
        plt.close()

    
    def run_analysis(self, window_size=50):
        """
        Runs the analysis by plotting individual and comparative performance.
        """
        self.plot_individual_performance(window_size)
        self.plot_comparison(window_size)
        print(f"Plots saved in {self.save_path}")
