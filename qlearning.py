# Importing necessary libraries
import gym  # Library for the OpenAI Gym environments
import numpy as np  # Library for numerical computations
import random  # Library for random number generation
from collections import defaultdict  # Data structures from the collections module
import matplotlib.pyplot as plt  # Library for plotting

# Create the CartPole environment
env = gym.make('CartPole-v1')

# Q-learning training function
def train_q_learning(env, num_episodes=1000, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
    # Function to discretize continuous state into bins
    def discretize_state(state, bins):
        return tuple(np.digitize(s, bins=b) for s, b in zip(state, bins))

    # Initialize the Q-table with zeros
    q_table = defaultdict(lambda: np.zeros(env.action_space.n))

    # Define the bins for discretizing state space
    state_bins = [
        np.linspace(-4.8, 4.8, 10),  # Bins for the cart position
        np.linspace(-4, 4, 10),  # Bins for the cart velocity
        np.linspace(-0.418, 0.418, 10),  # Bins for the pole angle
        np.linspace(-4, 4, 10)  # Bins for the pole velocity at tip
    ]

    rewards = []  # List to store rewards per episode

    # Training loop for each episode
    for episode in range(num_episodes):
        state = env.reset()  # Reset environment to initial state
        state = discretize_state(state, state_bins)  # Discretize the state
        total_reward = 0
        done = False

        # Loop for each time step in the episode
        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Choose random action
            else:
                action = np.argmax(q_table[state])  # Choose best action based on Q-table

            next_state, reward, done, _ = env.step(action)  # Take action and observe the result
            next_state = discretize_state(next_state, state_bins)  # Discretize the next state
            total_reward += reward

            # Update Q-table using the Bellman equation
            best_next_action = np.argmax(q_table[next_state])
            td_target = reward + gamma * q_table[next_state][best_next_action]
            q_table[state][action] += alpha * (td_target - q_table[state][action])

            state = next_state  # Move to the next state

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon
        rewards.append(total_reward)  # Store the total reward for the episode
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    return q_table, rewards  # Return the Q-table and rewards

# Train Q-learning and DQN
q_table, q_learning_rewards = train_q_learning(env)  # Train Q-learning agent

# print(max(q_learning_rewards)

# Plotting the rewards
plt.figure(figsize=(10,6))  # Set the figure size
plt.plot(q_learning_rewards, label='Q-learning Train')  # Plot Q-learning training rewards
plt.xlabel('Episode')  # Label x-axis
plt.ylabel('Total Reward')  # Label y-axis
plt.title('Q-Learning (Episode vs Rewards)')
plt.legend()  # Display legend
plt.show()  # Show the plot