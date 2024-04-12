"""
  Run this file at first, in order to see what is it printng. Instead of the print() use the respective log level
"""
############################### LOGGER
from abc import ABC, abstractmethod
from logs import *
import numpy as np
import matplotlib.pyplot as plt
import csv
logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)



class Bandit(ABC):
    ##==== DO NOT REMOVE ANYTHING FROM THIS CLASS ====##

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # log average reward (use f strings to make it informative)
        # log average regret (use f strings to make it informative)
        pass

#--------------------------------------#



import numpy as np
import matplotlib.pyplot as plt

class Visualization:

    def plot1(self, rewards, algorithm_names):
        """
        Visualize the performance of each bandit over time in both linear and logarithmic scale.

        :param rewards: List of arrays with the rewards of each algorithm
        :param algorithm_names: Names of the algorithms corresponding to the rewards
        """
        plt.figure(figsize=(12, 6))

        # Linear scale plot
        plt.subplot(1, 2, 1)
        for i, reward in enumerate(rewards):
            cumulative_rewards = np.cumsum(reward)
            plt.plot(cumulative_rewards, label=algorithm_names[i])
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards Over Time (Linear Scale)')
        plt.legend()

        # Log scale plot
        plt.subplot(1, 2, 2)
        for i, reward in enumerate(rewards):
            cumulative_rewards = np.cumsum(reward)
            plt.plot(cumulative_rewards, label=algorithm_names[i])
        plt.xscale('log')
        plt.xlabel('Trial (Log Scale)')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards Over Time (Log Scale)')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def plot2(self, e_greedy_rewards, thompson_rewards):
        """
        Compare E-greedy and Thompson Sampling cumulative rewards and regrets.

        :param e_greedy_rewards: Array of rewards from the E-Greedy algorithm
        :param thompson_rewards: Array of rewards from the Thompson Sampling algorithm
        """
        plt.figure(figsize=(12, 6))

        # Plot cumulative rewards
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(e_greedy_rewards), label='Epsilon-Greedy')
        plt.plot(np.cumsum(thompson_rewards), label='Thompson Sampling')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Reward')
        plt.title('Cumulative Rewards Comparison')
        plt.legend()

        # Plot cumulative regrets
        plt.subplot(1, 2, 2)
        optimal_rewards = np.max([e_greedy_rewards, thompson_rewards], axis=0)
        e_greedy_regrets = optimal_rewards - e_greedy_rewards
        thompson_regrets = optimal_rewards - thompson_rewards
        plt.plot(np.cumsum(e_greedy_regrets), label='Epsilon-Greedy Regrets')
        plt.plot(np.cumsum(thompson_regrets), label='Thompson Sampling Regrets')
        plt.xlabel('Trial')
        plt.ylabel('Cumulative Regret')
        plt.title('Cumulative Regrets Comparison')
        plt.legend()

        plt.tight_layout()
        plt.show()




#--------------------------------------#

class EpsilonGreedy:

    """
    An implementation of the Epsilon-Greedy algorithm for the multi-armed bandit problem.
    
    Attributes:
        p (list of float): Probabilities of success for each arm.
        epsilon (float): The probability of choosing a random arm; exploration rate.
        n_arms (int): Number of arms.
        counts (numpy array): Array to keep track of the number of times each arm was chosen.
        values (numpy array): Array to keep track of the average reward received from each arm (deprecated).
        estimates (numpy array): Array to keep track of the estimated values of each arm.
        rewards (list of float): List to store rewards received during the experiment.
        total_reward (float): Total accumulated reward.
    """

    def __init__(self, probabilities, epsilon):

        """
        Initializes the EpsilonGreedy algorithm with specified probabilities and epsilon.
        
        Args:
            probabilities (list of float): Probabilities of success for each arm.
            epsilon (float): The probability of choosing a random arm.
        """

        self.p = probabilities
        self.epsilon = epsilon
        self.n_arms = len(probabilities)
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.estimates = np.zeros(self.n_arms)
        self.rewards = []  # Initialize the rewards attribute
        self.total_reward = 0 
        self.optimal_rewards = []


    def __repr__(self):
        """
        Returns a formal representation of the EpsilonGreedy object.
        """
        return f"EpsilonGreedy(epsilon={self.epsilon}, probabilities={self.p})"


    def pull(self):
        """
        Conducts one pull of the bandit arm according to the epsilon-greedy strategy.
        
        Returns:
            int: The index of the chosen arm.
        """
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.p))
        else:
            return np.argmax(self.estimates)

    def update(self, chosen_arm, reward):
        """
        Updates the estimated values for the chosen arm based on the received reward.
        
        Args:
            chosen_arm (int): Index of the arm that was chosen.
            reward (float): Reward received from choosing the arm.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.estimates[chosen_arm]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.estimates[chosen_arm] = new_value

    def experiment(self, N):
        """
        Runs the bandit experiment for a specified number of trials.
        
        Args:
            N (int): Number of trials to run.
        
        Returns:
            numpy array: An array of rewards received during the experiment.
        """
        rewards = np.zeros(N)
        for i in range(N):
            chosen_arm = self.pull()
            reward = np.random.rand() < self.p[chosen_arm]
            self.update(chosen_arm, reward)
            rewards[i] = reward
            self.rewards.append(reward)  # Store each reward in the instance attribute
        self.total_reward = np.sum(rewards)
        return rewards

    def report(self, file_path='epsilon_greedy_report.csv'):
        """
        Generates a report of the experiment, logging average reward and regret, and saves it to a CSV file.
        
        Args:
            file_path (str): Path to the CSV file where the report will be saved.
        """
        if len(self.rewards) == 0:
            logger.warning("No rewards to report. Run experiment first.")
            return

        average_reward = np.mean(self.rewards)
        optimal_reward = max(self.p)
        regrets = [optimal_reward - reward for reward in self.rewards]
        average_regret = np.mean(regrets)

        # Log the average reward and regret
        logger.info(f'Average Reward for Epsilon-Greedy: {average_reward:.4f}')
        logger.info(f'Average Regret for Epsilon-Greedy: {average_regret:.4f}')

        # Write the results to a CSV file
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Bandit', 'Reward', 'Algorithm'])  # Updated header
            for i, reward in enumerate(self.rewards):
                writer.writerow([i+1, reward, 'Epsilon-Greedy'])





#--------------------------------------#
class ThompsonSampling:
    """
    Implements the Thompson Sampling algorithm for solving the multi-armed bandit problem.

    Attributes:
        p (list of float): A list containing the true probability of reward for each arm.
        n_arms (int): The number of arms in the bandit problem.
        alpha (numpy array): The array of alpha parameters for the beta distribution of each arm.
        beta (numpy array): The array of beta parameters for the beta distribution of each arm.
        rewards (list of float): The list of rewards obtained after each trial.
        optimal_rewards (list of float): The list of the maximum possible rewards at each trial for regret calculation.

    Methods:
        __init__(self, probabilities): Constructor for the ThompsonSampling class.
        __repr__(self): Returns an 'official' string representation of the object.
        pull(self): Simulates pulling an arm of the multi-armed bandit.
        update(self, chosen_arm, reward): Updates the parameters and rewards after pulling an arm.
        experiment(self, N): Runs the bandit experiment for N trials.
        report(self, file_path): Generates a report of the experiment and writes it to a CSV file.
    """
    def __init__(self, probabilities):
        """
        Initializes a new instance of the ThompsonSampling class.

        Args:
            probabilities (list of float): The true probability of reward for each arm.
        """
        self.p = probabilities
        self.n_arms = len(probabilities)
        self.alpha = np.ones(self.n_arms)
        self.beta = np.ones(self.n_arms)
        self.rewards = []  # Tracking rewards
        self.optimal_rewards = []  # Tracking optimal rewards (for regret calculation)

    def __repr__(self):
        """
        Returns an 'official' string representation of the ThompsonSampling object, which can be used to recreate
        the object if fed back into Python.

        Returns:
            str: A string representation of the ThompsonSampling instance.
        """
        return f"ThompsonSampling(probabilities={self.p})"

    def pull(self):
        """
        Simulates the action of pulling an arm based on Thompson Sampling strategy. This method uses the beta 
        distribution parameters (alpha and beta) to draw samples and then selects the arm with the highest sample.

        Returns:
            int: The index of the chosen arm.
        """
        samples = [np.random.beta(a, b) for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)

    def update(self, chosen_arm, reward):
        """
        Updates the parameters of the beta distribution (alpha and beta) for the chosen arm based on the reward
        received from pulling the arm. It also updates the rewards and optimal_rewards lists.

        Args:
            chosen_arm (int): The index of the arm that was pulled.
            reward (float): The reward received from pulling the chosen arm.
        """
        self.alpha[chosen_arm] += reward
        self.beta[chosen_arm] += 1 - reward
        self.rewards.append(reward)
        self.optimal_rewards.append(max(self.p))  # Assuming the maximum probability is the optimal

    def experiment(self, N):
        """
        Runs the bandit experiment for a specified number of trials (N). It pulls an arm and updates the parameters
        and rewards in each trial.

        Args:
            N (int): The number of trials to run the experiment.

        Returns:
            list of float: The list of rewards received during the experiment.
        """
        for i in range(N):
            chosen_arm = self.pull()
            reward = np.random.rand() < self.p[chosen_arm]
            self.update(chosen_arm, reward)
        return self.rewards

    def report(self, file_path='thompson_sampling_report.csv'):
        """
        Generates a report that includes the average reward and average regret from the experiment. It logs these
        statistics and writes the details of the rewards into a CSV file specified by file_path.

        Args:
            file_path (str, optional): The file path where the report CSV will be saved. Defaults to 
            'thompson_sampling_report.csv'.
        """
        average_reward = np.mean(self.rewards)
        average_regret = np.mean([opt - act for opt, act in zip(self.optimal_rewards, self.rewards)])

        logger.info(f'Average Reward for Thompson Sampling: {average_reward}')
        logger.info(f'Average Regret for Thompson Sampling: {average_regret}')

        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Bandit', 'Reward', 'Algorithm'])  # Updated header
            for i, reward in enumerate(self.rewards):
                writer.writerow([i+1, reward, 'Thompson Sampling'])




def comparison(thompson_rewards, epsilon_rewards):
    """
    This function compares the performances of the Thompson Sampling and Epsilon-Greedy algorithms
    by plotting their cumulative rewards over time.

    Args:
    - thompson_rewards (list of float): The list of rewards received from the Thompson Sampling algorithm.
    - epsilon_rewards (list of float): The list of rewards received from the Epsilon-Greedy algorithm.
    """
    thompson_cumulative = np.cumsum(thompson_rewards)
    epsilon_cumulative = np.cumsum(epsilon_rewards)

    plt.figure(figsize=(10, 6))
    plt.plot(thompson_cumulative, label='Thompson Sampling', color='blue')
    plt.plot(epsilon_cumulative, label='Epsilon-Greedy', color='red')

    plt.title('Comparison of Thompson Sampling and Epsilon-Greedy Algorithms')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Reward')

    plt.legend()

    plt.show()

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")