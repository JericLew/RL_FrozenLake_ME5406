import numpy as np
from rl_analyser import RLAnalyzer
from RL import FirstVisitMonteCarlo, SARSA, QLearning
from frozen_lake_env import FrozenLakeEnv

def main():
    map_path = '4x4.txt'
    optimal_threshold = 6 # BASED ON THE MAP (MANUALLY CALCULATED)
    num_episode = 2000
    max_step = np.inf
    gamma = 0.99
    epsilon = 0.1
    lr = 0.1

    # First-visit Monte Carlo
    monte_carlo_4x4 = FirstVisitMonteCarlo(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
    monte_carlo_4x4.run()

    # SARSA
    sarsa_4x4 = SARSA(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
    sarsa_4x4.run()

    # Q-learning
    q_learning_4x4 = QLearning(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
    q_learning_4x4.run()

    # Analyze 4x4
    rl_algos_4x4 = [monte_carlo_4x4, sarsa_4x4, q_learning_4x4]
    analyzers_4x4 = RLAnalyzer(rl_algos_4x4, save_path='4x4_plots')
    analyzers_4x4.run_analysis(window_size=50)
    for rl_algo in rl_algos_4x4:
        print(f"4x4 {rl_algo.algo_type}\
                success: {rl_algo.success_count},\
                failure: {rl_algo.failure_count},\
                first successful policy: {rl_algo.first_successful_policy},\
                first optimal policy: {rl_algo.first_optimal_policy}")

    map_path = '10x10.txt'
    optimal_threshold = 18 # BASED ON THE MAP (MANUALLY CALCULATED)
    num_episode = 10000
    max_step = np.inf
    gamma = 0.99
    epsilon = 0.1
    lr = 0.1

    # First-visit Monte Carlo
    monte_carlo_10x10 = FirstVisitMonteCarlo(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
    monte_carlo_10x10.run()

    # SARSA
    sarsa_10x10 = SARSA(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
    sarsa_10x10.run()

    # Q-learning
    q_learning_10x10 = QLearning(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
    q_learning_10x10.run()

    # Analyze 4x4
    rl_algos_10x10 = [monte_carlo_10x10, sarsa_10x10, q_learning_10x10]
    analyzers_10x10 = RLAnalyzer(rl_algos_10x10, save_path='10x10_plots')
    analyzers_10x10.run_analysis(window_size=250)
    for rl_algo in rl_algos_10x10:
        print(f"10x10 {rl_algo.algo_type}\
                success: {rl_algo.success_count},\
                failure: {rl_algo.failure_count},\
                first successful policy: {rl_algo.first_successful_policy},\
                first optimal policy: {rl_algo.first_optimal_policy}")
    

if __name__ == "__main__":
    main()