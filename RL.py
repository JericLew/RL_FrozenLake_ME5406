import numpy as np
from frozen_lake_env import FrozenLakeEnv


class RLBase:
    """
    Base Class for RL
    """
    def __init__(self, map_path, optimal_threshold, num_episode=2000, max_step=np.inf, gamma=0.95, epsilon=0.1, lr=0.1):
        self.env = FrozenLakeEnv(map_path)
        self.optimal_threshold = optimal_threshold # threshold for optimal policy

        self.num_action = len(self.env.actions)
        self.action_list = [i for i in range(self.num_action)] # left, right, up, down
        self.map_x, self.map_y = self.env.map_x, self.env.map_y
        self.num_episode = num_episode
        self.max_step = max_step # max step per episode,
        self.gamma = gamma  # discount rate
        self.epsilon = epsilon # epsilon greedy
        self.lr = lr # learning rate for SARSA and Q-Learning

        self.algo_type = None
        self.episode_steps = [] # length of each episode
        self.episode_average_rewards = [] # (total reward / step) for each episode
        self.success_count = 0
        self.failure_count = 0
        self.first_successful_policy = None
        self.first_optimal_policy = None

    def init_policy_table(self):
        """
        Init policy table

        returns:
            policy_table (dict)
                keys: state tuples
                values: list of action probs (index 0: left, 1: right, 2: up, 3: down)
            Note: each action is init with equal prob
        """
        policy_table = {}
        for y in range(self.map_y):
            for x in range(self.map_x):
                state = x, y
                policy_table[state] = [1 / self.num_action] * self.num_action
        return policy_table

    def init_Q_table(self):
        """
        Init Q table

        returns:
            Q_table (dict)
                keys: state tuples
                values: list of action values (index 0: left, 1: right, 2: up, 3: down)
            Note: each action is init with 0 action value
        """
        Q_table = {}
        for y in range(self.map_y):
            for x in range(self.map_x):
                state = x, y
                Q_table[state] = [0] * self.num_action
        return Q_table
    
    def update_policy(self, policy_list, Q_list):
        """
        Update given given state's policy list based on Q list

        args:
            policy_list (list): list of action probs from policy_table[state]
            Q_list (list): list of action values from Q_table[state]

        Note:
            - Updates the referenced list witht epsilon greedy policy
            - If two actions have the same Q, pick one randomly
        """
        max_Q_action_id = [] # list of action id that result in max Q
        for id, q in enumerate(Q_list):
            if q == max(Q_list):
                max_Q_action_id.append(id)
        best_action = np.random.choice(max_Q_action_id) # random best action

        for a in range(self.num_action):
            if a == best_action: # exploit
                policy_list[a] = 1 - self.epsilon + self.epsilon / self.num_action
            else:
                policy_list[a] = self.epsilon / self.num_action # exploration

    def eval_policy(self, policy_table):
        """
        Checks if policy is succesful and optimal

        args:
            policy_table (dict): policy table, used to choose action

        returns:
            policy_successful (bool): True if greedy policy results in reaching goal
            policy_optimal (bool): True if greedy policy is optimal
        """
        policy_successful = False
        policy_optimal = False

        state = self.env.reset()
        done = False
        state_list = []
        while not done:
            action = int(np.argmax(policy_table[state])) # greedy policy
            dx, dy = self.env.actions[action]
            state_list.append(state)
            new_x, new_y = state[0] + dx, state[1] + dy
            if not (0 <= new_x < self.map_x and 0 <= new_y < self.map_y):
                done = True # not optimal: tried to go out of bounds
            elif (new_x, new_y) in state_list:
                done = True # not optimal: loop
            elif self.env.map[new_y, new_x] == '.':
                state = new_x, new_y
            elif self.env.map[new_y, new_x] == 'H':
                done = True # not optimal: fall into hole
            elif self.env.map[new_y, new_x] == 'G':
                policy_successful = True
                if len(state_list) <= self.optimal_threshold:
                    policy_optimal = True
                done = True # near optimal: reach goal
        return policy_successful, policy_optimal

class FirstVisitMonteCarlo(RLBase):
    """
    First-visit Monte Carlo without exploring starts
    """    

    def __init__(self, map_path, optimal_threshold, num_episode=2000, max_step=np.inf, gamma=0.95, epsilon=0.1, lr=0.1):
        super().__init__(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
        self.algo_type = 'Monte Carlo'

    def init_return_table(self):
        """
        Init return table

        returns:
            return_table (dict)
                keys: state tuples
                values: list of list of returns
                    - first index is action_id 0: left, 1: right, 2: up, 3: down
                    - second index is episode
            Note: init with empty list of list
        """

        return_table = {}
        for y in range(self.map_y):
            for x in range(self.map_x):
                state = x, y
                return_table[state] = [[] for i in range(self.num_action)]
        return return_table

    def run_episode(self, policy_table):
        """
        Run a episode given current policy_table

        Args:
            policy_table (dict): policy table, used to choose action.

        Returns:
            state_list (list): list of state tuples visited with len = num steps
            action_list (list): list of actions taken with len = num steps
            return_list (list): list of first visit returns with len = num steps
        """

        step = 0
        done = False
        state = self.env.reset()
        state_list, action_list, reward_list, return_list = [], [], [], []
        while not done:
            step += 1
            action = np.random.choice(self.action_list, p=policy_table[state])
            state_list.append(state)
            action_list.append(action)
            state, reward, done = self.env.step(action)
            reward_list.append(reward)
            if step > self.max_step:
                break
        G = 0
        for i in range(len(state_list)-1, -1, -1): # trace back the episode
            G = self.gamma * G + reward_list[i]
            return_list.append(G)
        return_list.reverse() # reverse the traced back list to positive order
        return state_list, action_list, return_list, reward_list

    def run(self):
        """
        Run FirstVisitMonteCarlo

        Returns:
            policy_table (dict)
        """

        policy_table = self.init_policy_table()
        Q_table = self.init_Q_table()
        return_table = self.init_return_table()

        for episode in range(self.num_episode):
            state_list, action_list, return_list, reward_list = self.run_episode(policy_table)
            temp_return_table = self.init_return_table()
            for i in range(len(state_list)): # for step in episode
                state, action = state_list[i], action_list[i]
                if not temp_return_table[state][action]: # if state action pair not visited before
                    temp_return_table[state][action].append(return_list[i]) # set visited
                    return_table[state][action].append(return_list[i]) # add return to return table
                Q = np.mean(return_table[state][action]) # average return
                Q_table[state][action] = Q # update Q table
                self.update_policy(policy_table[state], Q_table[state])
            # collect data for analysis
            self.episode_steps.append(len(state_list))
            self.episode_average_rewards.append(sum(reward_list)/len(state_list))
            if reward_list[-1] == 1:
                self.success_count += 1
            else:
                self.failure_count += 1
            success, optimal = self.eval_policy(policy_table)
            if success and self.first_successful_policy == None:
                self.first_successful_policy = episode
            if optimal and self.first_optimal_policy == None:
                self.first_optimal_policy = episode
            # Plot heatmap
            if episode % 100 == 0 or episode == self.num_episode - 1:
                self.env.visualize_q_table(episode, Q_table, self.algo_type)
        # self.env.generate_gif(self.algo_type)
        self.env.visualize_deterministic_policy(policy_table, self.algo_type)
        return policy_table

class SARSA(RLBase):
    """
    SARSA
    """    
    def __init__(self, map_path, optimal_threshold, num_episode=2000, max_step=np.inf, gamma=0.95, epsilon=0.1, lr=0.1):
        super().__init__(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
        self.algo_type = 'SARSA'

    def run(self):
        """
        Run SARSA

        Returns:
            policy_table (dict)
        """

        policy_table = self.init_policy_table()
        Q_table = self.init_Q_table()
        for episode in range(self.num_episode):
            step   = 0
            done   = False
            state  = self.env.reset()
            action = np.random.choice(self.action_list, p=policy_table[state])
            reward_list = []
            while not done:
                step += 1
                new_state, reward, done = self.env.step(action) # step env, get r and s'
                reward_list.append(reward)
                new_action = np.random.choice(self.action_list, p=policy_table[new_state]) # get a'
                new_Q = Q_table[new_state][new_action] # find Q(s', a')
                Q_table[state][action] += self.lr * (reward + self.gamma * new_Q - Q_table[state][action]) # TD update rule
                self.update_policy(policy_table[state], Q_table[state])
                state = new_state
                action = new_action
                if step > self.max_step:
                    break
            # collect data for analysis
            self.episode_steps.append(step)
            self.episode_average_rewards.append(sum(reward_list)/step)
            if reward_list[-1] == 1:
                self.success_count += 1
            else:
                self.failure_count += 1
            success, optimal = self.eval_policy(policy_table)
            if success and self.first_successful_policy == None:
                self.first_successful_policy = episode
            if optimal and self.first_optimal_policy == None:
                self.first_optimal_policy = episode
            # Plot heatmap
            if episode % 100 == 0 or episode == self.num_episode - 1:
                self.env.visualize_q_table(episode, Q_table, self.algo_type)
        # self.env.generate_gif(self.algo_type)
        self.env.visualize_deterministic_policy(policy_table, self.algo_type)
        return Q_table

class QLearning(RLBase):
    """
    Q-learning
    """    

    def __init__(self, map_path, optimal_threshold, num_episode=2000, max_step=np.inf, gamma=0.95, epsilon=0.1, lr=0.1):
        super().__init__(map_path, optimal_threshold, num_episode, max_step, gamma, epsilon, lr)
        self.algo_type = 'Q-learning'

    def run(self):
        """
        Run Q-learning

        Returns:
            policy_table (dict)
        """

        policy_table = self.init_policy_table()
        Q_table = self.init_Q_table()
        for episode in range(self.num_episode):
            step   = 0
            done   = False
            state  = self.env.reset()
            reward_list = []
            while not done:
                step += 1
                action = np.random.choice(self.action_list, p=policy_table[state])
                new_state, reward, done = self.env.step(action) # step env, get r and s'
                reward_list.append(reward)
                new_Q = max(Q_table[new_state]) # max of Q(s', a'), greedy
                Q_table[state][action] += self.lr * (reward + self.gamma * new_Q - Q_table[state][action])
                self.update_policy(policy_table[state], Q_table[state])
                state = new_state
                if step > self.max_step:
                    break
            # collect data for analysis
            self.episode_steps.append(step)
            self.episode_average_rewards.append(sum(reward_list)/step)
            if reward_list[-1] == 1:
                self.success_count += 1
            else:
                self.failure_count += 1
            success, optimal = self.eval_policy(policy_table)
            if success and self.first_successful_policy == None:
                self.first_successful_policy = episode
            if optimal and self.first_optimal_policy == None:
                self.first_optimal_policy = episode
            if episode % 100 == 0 or episode == self.num_episode - 1:
                self.env.visualize_q_table(episode, Q_table, self.algo_type)
        # self.env.generate_gif(self.algo_type)
        self.env.visualize_deterministic_policy(policy_table, self.algo_type)
        return Q_table
    
if __name__ == '__main__':
    from rl_analyser import RLAnalyzer

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