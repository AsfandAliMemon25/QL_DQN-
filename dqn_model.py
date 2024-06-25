# Importing necessary libraries
import gym  # Library for the OpenAI Gym environments
import numpy as np  # Library for numerical computations
import random  # Library for random number generation
from collections import defaultdict, deque  # Data structures from the collections module
import tensorflow as tf  # Library for deep learning
from tensorflow.keras import layers, models, optimizers, losses  # Keras modules for building and training neural networks
import matplotlib.pyplot as plt  # Library for plotting


# Create the CartPole environment
env = gym.make('CartPole-v1')

# Define the neural network for DQN
class DQN(models.Model):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        # Define the layers of the neural network
        self.fc1 = layers.Dense(24, activation='relu')
        self.fc2 = layers.Dense(24, activation='relu')
        self.fc3 = layers.Dense(action_dim)

    def call(self, x):
        # Define the forward pass
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)

# DQN training function
def train_dqn(env, num_episodes=200, learning_rate=0.001, batch_size=64, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01, target_update=10):
    state_dim = env.observation_space.shape[0]  # Dimension of state space
    action_dim = env.action_space.n  # Dimension of action space

    policy_net = DQN(state_dim, action_dim)  # Initialize policy network
    target_net = DQN(state_dim, action_dim)  # Initialize target network
    target_net.set_weights(policy_net.get_weights())  # Synchronize target network with policy network

    optimizer = optimizers.Adam(learning_rate)  # Adam optimizer for training
    memory = deque(maxlen=2000)  # Replay memory

    # Function to select action using epsilon-greedy policy
    def select_action(state):
        if random.random() < epsilon:
            return random.choice(range(action_dim))  # Random action
        else:
            q_values = policy_net(tf.convert_to_tensor(state, dtype=tf.float32))  # Q-values from policy network
            return tf.argmax(q_values[0]).numpy()  # Best action based on Q-values

    # Function to optimize the model
    def optimize_model():
        if len(memory) < batch_size:
            return  # Return if not enough samples in memory
        batch = random.sample(memory, batch_size)  # Randomly sample a batch from memory
        states, actions, rewards, next_states, dones = zip(*batch)

        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.int64)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)

        # Calculate the loss using the Bellman equation
        with tf.GradientTape() as tape:
            current_q_values = tf.reduce_sum(policy_net(states) * tf.one_hot(actions, action_dim), axis=1)
            max_next_q_values = tf.reduce_max(target_net(next_states), axis=1)
            expected_q_values = rewards + gamma * max_next_q_values * (1 - dones)
            loss = losses.MSE(expected_q_values, current_q_values)

        grads = tape.gradient(loss, policy_net.trainable_variables)  # Compute gradients
        optimizer.apply_gradients(zip(grads, policy_net.trainable_variables))  # Apply gradients

    rewards = []  # List to store rewards per episode

    # Training loop for each episode
    for episode in range(num_episodes):
        state = env.reset()  # Reset environment to initial state
        total_reward = 0
        done = False

        # Loop for each time step in the episode
        while not done:
            state_tensor = tf.convert_to_tensor(state, dtype=tf.float32)[tf.newaxis, ...]
            action = select_action(state_tensor)  # Select action
            next_state, reward, done, _ = env.step(action)  # Take action and observe the result
            total_reward += reward

            memory.append((state, action, reward, next_state, done))  # Store experience in replay memory
            state = next_state  # Move to the next state

            optimize_model()  # Optimize the model

        if episode % target_update == 0:
            target_net.set_weights(policy_net.get_weights())  # Update target network

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Decay epsilon
        rewards.append(total_reward)  # Store the total reward for the episode
        if episode % 100 == 0:
            print(f"Episode: {episode}, Total Reward: {total_reward}")

    return policy_net, rewards  # Return the policy network and rewards


policy_net, dqn_rewards = train_dqn(env)  # Train DQN agent

# print(max(dqn_rewards))

# Plotting the rewards
plt.figure(figsize=(10,6))  # Set the figure size
plt.plot(dqn_rewards, label='DQN Train')  # Plot DQN training rewards
plt.xlabel('Episode')  # Label x-axis
plt.ylabel('Total Reward')  # Label y-axis
plt.title('DQN (Episode vs Rewards)')
plt.legend()  # Display legend
plt.show()  # Show the plot

