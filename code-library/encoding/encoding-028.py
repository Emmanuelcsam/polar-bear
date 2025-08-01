#!/usr/bin/env python3
"""
Comprehensive Reinforcement Learning Tutorial Implementation
Based on the Machine Learning with Phil tutorials
"""

import sys
import subprocess
import importlib
import os
from datetime import datetime
import time

# Color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def log(message, level="INFO"):
    """Log messages with timestamp and color coding"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if level == "INFO":
        color = Colors.OKBLUE
    elif level == "SUCCESS":
        color = Colors.OKGREEN
    elif level == "WARNING":
        color = Colors.WARNING
    elif level == "ERROR":
        color = Colors.FAIL
    elif level == "HEADER":
        color = Colors.HEADER + Colors.BOLD
    else:
        color = ""
    
    print(f"{color}[{timestamp}] {level}: {message}{Colors.ENDC}")

def check_and_install_package(package_name, import_name=None):
    """Check if a package is installed, if not, install it"""
    if import_name is None:
        import_name = package_name
    
    log(f"Checking for {package_name}...", "INFO")
    
    try:
        importlib.import_module(import_name)
        log(f"{package_name} is already installed", "SUCCESS")
        return True
    except ImportError:
        log(f"{package_name} not found. Installing...", "WARNING")
        
        try:
            # Upgrade pip first
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
            
            # Install the package
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", package_name])
            
            # Verify installation
            importlib.import_module(import_name)
            log(f"{package_name} installed successfully", "SUCCESS")
            return True
            
        except subprocess.CalledProcessError as e:
            log(f"Failed to install {package_name}: {e}", "ERROR")
            return False
        except ImportError:
            log(f"Failed to import {package_name} after installation", "ERROR")
            return False

def check_and_install_dependencies():
    """Check and install all required dependencies"""
    log("CHECKING AND INSTALLING DEPENDENCIES", "HEADER")
    
    dependencies = [
        ("numpy", "numpy"),
        ("tensorflow", "tensorflow"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("gym", "gym"),
        ("gym[atari]", "gym"),
        ("gym[box2d]", "gym"),
        ("matplotlib", "matplotlib"),
        ("opencv-python", "cv2"),
        ("Pillow", "PIL"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn")
    ]
    
    all_installed = True
    
    for package, import_name in dependencies:
        if not check_and_install_package(package, import_name):
            all_installed = False
    
    if all_installed:
        log("All dependencies installed successfully!", "SUCCESS")
    else:
        log("Some dependencies failed to install. Please check the errors above.", "ERROR")
        sys.exit(1)
    
    return all_installed

# Import required libraries after installation
def import_libraries():
    """Import all required libraries"""
    log("Importing libraries...", "INFO")
    
    global np, tf, torch, nn, optim, gym, plt, cv2
    
    import numpy as np
    import tensorflow as tf
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import gym
    import matplotlib.pyplot as plt
    import cv2
    
    log("All libraries imported successfully", "SUCCESS")

class DeepQNetwork:
    """Deep Q Network implementation in TensorFlow"""
    def __init__(self, lr, n_actions, name, input_dims, fc1_dims, chkpt_dir):
        log(f"Initializing Deep Q Network: {name}", "INFO")
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.input_dims = input_dims
        self.chkpt_dir = chkpt_dir
        self.sess = tf.Session()
        
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()
        
        log(f"Deep Q Network {name} initialized successfully", "SUCCESS")
    
    def build_network(self):
        log(f"Building network architecture for {self.name}", "INFO")
        
        with tf.variable_scope(self.name):
            self.input = tf.placeholder(tf.float32, shape=[None, *self.input_dims], name='inputs')
            self.actions = tf.placeholder(tf.float32, shape=[None, self.n_actions], name='action_taken')
            self.q_target = tf.placeholder(tf.float32, shape=[None], name='q_value')
            
            # Convolutional layers
            conv1 = tf.layers.conv2d(inputs=self.input, filters=32, kernel_size=(8,8),
                                    strides=4, name='conv1',
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv1_activated = tf.nn.relu(conv1)
            
            conv2 = tf.layers.conv2d(inputs=conv1_activated, filters=64, kernel_size=(4,4),
                                    strides=2, name='conv2',
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv2_activated = tf.nn.relu(conv2)
            
            conv3 = tf.layers.conv2d(inputs=conv2_activated, filters=128, kernel_size=(3,3),
                                    strides=1, name='conv3',
                                    kernel_initializer=tf.variance_scaling_initializer(scale=2))
            conv3_activated = tf.nn.relu(conv3)
            
            # Flatten and fully connected layers
            flat = tf.layers.flatten(conv3_activated)
            dense1 = tf.layers.dense(flat, units=self.fc1_dims, activation=tf.nn.relu,
                                   kernel_initializer=tf.variance_scaling_initializer(scale=2))
            
            self.Q_values = tf.layers.dense(dense1, units=self.n_actions,
                                          kernel_initializer=tf.variance_scaling_initializer(scale=2))
            
            # Loss function
            self.q = tf.reduce_sum(tf.multiply(self.Q_values, self.actions), axis=1)
            self.loss = tf.reduce_mean(tf.square(self.q - self.q_target))
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            
        log(f"Network architecture built successfully for {self.name}", "SUCCESS")

class DQNAgent:
    """Deep Q Learning Agent"""
    def __init__(self, alpha, gamma, mem_size, n_actions, epsilon, batch_size,
                 replace_target=10000, input_dims=(210, 160, 4),
                 q_next_dir='tmp/q_next', q_eval_dir='tmp/q_eval'):
        log("Initializing DQN Agent", "INFO")
        
        self.action_space = list(range(n_actions))
        self.n_actions = n_actions
        self.gamma = gamma
        self.mem_size = mem_size
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.replace_target = replace_target
        self.mem_cntr = 0
        self.learn_step = 0
        
        # Initialize memory
        self.memory = []
        
        # Create networks
        self.q_eval = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                  name='q_eval', chkpt_dir=q_eval_dir, fc1_dims=512)
        self.q_next = DeepQNetwork(alpha, n_actions, input_dims=input_dims,
                                  name='q_next', chkpt_dir=q_next_dir, fc1_dims=512)
        
        log("DQN Agent initialized successfully", "SUCCESS")
    
    def store_transition(self, state, action, reward, state_, done):
        """Store transition in memory"""
        if self.mem_cntr < self.mem_size:
            self.memory.append([state, action, reward, state_, done])
        else:
            self.memory[self.mem_cntr % self.mem_size] = [state, action, reward, state_, done]
        self.mem_cntr += 1
    
    def choose_action(self, state):
        """Choose action using epsilon-greedy strategy"""
        state = state[np.newaxis, :]
        rand = np.random.random()
        
        if rand < 1 - self.epsilon:
            actions = self.q_eval.sess.run(self.q_eval.Q_values,
                                          feed_dict={self.q_eval.input: state})
            action = np.argmax(actions)
        else:
            action = np.random.choice(self.action_space)
        
        return action
    
    def learn(self):
        """Learn from batch of experiences"""
        if self.mem_cntr > self.batch_size:
            log("Agent learning from experience batch", "INFO")
            
            # Replace target network
            if self.learn_step % self.replace_target == 0 and self.replace_target is not None:
                self.update_graph()
            
            # Sample batch
            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size else self.mem_size
            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            
            # Extract batch data
            state_batch = np.array([self.memory[b][0] for b in batch])
            action_batch = np.array([self.memory[b][1] for b in batch])
            reward_batch = np.array([self.memory[b][2] for b in batch])
            state_batch_ = np.array([self.memory[b][3] for b in batch])
            terminal_batch = np.array([self.memory[b][4] for b in batch])
            
            # Calculate target values
            q_eval = self.q_eval.sess.run(self.q_eval.Q_values,
                                         feed_dict={self.q_eval.input: state_batch})
            q_next = self.q_next.sess.run(self.q_next.Q_values,
                                         feed_dict={self.q_next.input: state_batch_})
            
            max_actions = np.argmax(q_next, axis=1)
            
            q_target = q_eval.copy()
            batch_idx = np.arange(self.batch_size)
            
            # Update Q values
            q_target[batch_idx, action_batch] = reward_batch + \
                self.gamma * q_next[batch_idx, max_actions] * (1 - terminal_batch)
            
            # Create action array
            actions = np.zeros((self.batch_size, self.n_actions))
            actions[batch_idx, action_batch] = 1.0
            
            # Train network
            _ = self.q_eval.sess.run(self.q_eval.train_op,
                                    feed_dict={self.q_eval.input: state_batch,
                                             self.q_eval.actions: actions,
                                             self.q_eval.q_target: q_target[batch_idx, action_batch]})
            
            # Update epsilon
            if self.epsilon > 0.01:
                self.epsilon *= 0.9999
            else:
                self.epsilon = 0.01
            
            self.learn_step += 1
    
    def update_graph(self):
        """Copy weights from eval to target network"""
        log("Updating target network", "INFO")
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_eval')
        
        for t, e in zip(t_params, e_params):
            self.q_eval.sess.run(tf.assign(t, e))

class PolicyGradientAgent:
    """Policy Gradient Agent for Lunar Lander"""
    def __init__(self, lr, gamma, n_actions, layer1_size, layer2_size, input_dims, chkpt_dir='tmp/'):
        log("Initializing Policy Gradient Agent", "INFO")
        
        self.lr = lr
        self.gamma = gamma
        self.n_actions = n_actions
        self.layer1_size = layer1_size
        self.layer2_size = layer2_size
        self.input_dims = input_dims
        self.chkpt_dir = chkpt_dir
        self.action_space = list(range(n_actions))
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
        self.sess = tf.Session()
        self.build_net()
        self.sess.run(tf.global_variables_initializer())
        
        log("Policy Gradient Agent initialized successfully", "SUCCESS")
    
    def build_net(self):
        log("Building Policy Gradient network", "INFO")
        
        with tf.variable_scope('PolicyNetwork'):
            self.input = tf.placeholder(tf.float32, shape=[None, self.input_dims], name='inputs')
            self.label = tf.placeholder(tf.int32, shape=[None], name='labels')
            self.G = tf.placeholder(tf.float32, shape=[None], name='G')
            
            l1 = tf.layers.dense(inputs=self.input, units=self.layer1_size,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            l2 = tf.layers.dense(inputs=l1, units=self.layer2_size,
                               activation=tf.nn.relu,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            l3 = tf.layers.dense(inputs=l2, units=self.n_actions,
                               kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            self.actions = tf.nn.softmax(l3, name='probabilities')
            
            # Loss function
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=l3, labels=self.label)
            
            loss = neg_log_prob * self.G
            
            self.cost = tf.reduce_mean(loss)
            self.train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.cost)
            
        log("Policy Gradient network built successfully", "SUCCESS")
    
    def choose_action(self, observation):
        """Choose action based on policy"""
        observation = observation.reshape(1, -1)
        probabilities = self.sess.run(self.actions,
                                     feed_dict={self.input: observation})[0]
        action = np.random.choice(self.action_space, p=probabilities)
        return action
    
    def store_transition(self, observation, action, reward):
        """Store transition in memory"""
        self.state_memory.append(observation)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
    
    def learn(self):
        """Learn from episode"""
        log("Policy Gradient Agent learning from episode", "INFO")
        
        state_memory = np.array(self.state_memory)
        action_memory = np.array(self.action_memory)
        reward_memory = np.array(self.reward_memory)
        
        # Calculate discounted rewards
        G = np.zeros_like(reward_memory)
        for t in range(len(reward_memory)):
            G_sum = 0
            discount = 1
            for k in range(t, len(reward_memory)):
                G_sum += reward_memory[k] * discount
                discount *= self.gamma
            G[t] = G_sum
        
        # Normalize rewards
        mean = np.mean(G)
        std = np.std(G) if np.std(G) > 0 else 1
        G = (G - mean) / std
        
        # Train network
        _ = self.sess.run(self.train_op,
                         feed_dict={self.input: state_memory,
                                   self.label: action_memory,
                                   self.G: G})
        
        # Clear memory
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []

class GridWorld:
    """Custom Grid World Environment"""
    def __init__(self, m, n, magicSquares):
        log(f"Initializing GridWorld environment ({m}x{n})", "INFO")
        
        self.grid = np.zeros((m, n))
        self.m = m
        self.n = n
        self.stateSpace = list(range(self.m * self.n))
        self.stateSpace.remove((m*n) - 1)  # Remove terminal state
        self.stateSpacePlus = list(range(self.m * self.n))
        self.actionSpace = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1}
        self.possibleActions = ['U', 'D', 'L', 'R']
        
        self.addMagicSquares(magicSquares)
        self.agentPosition = 0
        
        log("GridWorld environment initialized successfully", "SUCCESS")
    
    def addMagicSquares(self, magicSquares):
        """Add magic teleportation squares"""
        self.magicSquares = magicSquares
        i = 2
        for square in self.magicSquares:
            x = square // self.n
            y = square % self.n
            self.grid[x][y] = i
            i += 1
            x = self.magicSquares[square] // self.n
            y = self.magicSquares[square] % self.n
            self.grid[x][y] = i
            i += 1
    
    def isTerminalState(self, state):
        """Check if state is terminal"""
        return state in self.stateSpacePlus and state not in self.stateSpace
    
    def getAgentRowAndColumn(self):
        """Get agent's current position"""
        x = self.agentPosition // self.n
        y = self.agentPosition % self.n
        return x, y
    
    def setState(self, state):
        """Set agent's state"""
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 0
        self.agentPosition = state
        x, y = self.getAgentRowAndColumn()
        self.grid[x][y] = 1
    
    def offGridMove(self, newState, oldState):
        """Check if move is off grid"""
        if newState not in self.stateSpacePlus:
            return True
        elif oldState % self.m == 0 and newState % self.m == self.m - 1:
            return True
        elif oldState % self.m == self.m - 1 and newState % self.m == 0:
            return True
        else:
            return False
    
    def step(self, action):
        """Take a step in the environment"""
        x, y = self.getAgentRowAndColumn()
        resultingState = self.agentPosition + self.actionSpace[action]
        
        if resultingState in self.magicSquares:
            resultingState = self.magicSquares[resultingState]
        
        reward = -1 if not self.isTerminalState(resultingState) else 0
        
        if not self.offGridMove(resultingState, self.agentPosition):
            self.setState(resultingState)
            return resultingState, reward, self.isTerminalState(resultingState), None
        else:
            return self.agentPosition, reward, self.isTerminalState(self.agentPosition), None
    
    def reset(self):
        """Reset environment"""
        self.agentPosition = 0
        self.grid = np.zeros((self.m, self.n))
        self.addMagicSquares(self.magicSquares)
        return self.agentPosition
    
    def render(self):
        """Render the grid"""
        print('------------------')
        for row in self.grid:
            for col in row:
                if col == 0:
                    print('-', end='\t')
                elif col == 1:
                    print('X', end='\t')
                elif col == 2:
                    print('Ain', end='\t')
                elif col == 3:
                    print('Aout', end='\t')
                elif col == 4:
                    print('Bin', end='\t')
                elif col == 5:
                    print('Bout', end='\t')
            print('\n')
        print('------------------')

def preprocess_frame(frame):
    """Preprocess game frames"""
    # Convert to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 84))
    return resized / 255.0

def stack_frames(stacked_frames, frame, buffer_size=4):
    """Stack frames for temporal information"""
    if stacked_frames is None:
        stacked_frames = np.zeros((buffer_size, *frame.shape))
        for idx in range(buffer_size):
            stacked_frames[idx, :] = frame
    else:
        stacked_frames[0:buffer_size-1, :] = stacked_frames[1:, :]
        stacked_frames[buffer_size-1, :] = frame
    
    return stacked_frames

def run_dqn_breakout():
    """Run Deep Q Learning on Breakout"""
    log("STARTING DEEP Q LEARNING ON BREAKOUT", "HEADER")
    
    env = gym.make('BreakoutDeterministic-v4')
    
    agent = DQNAgent(gamma=0.99, epsilon=1.0, alpha=0.0001,
                     input_dims=(180, 160, 4), n_actions=env.action_space.n,
                     mem_size=25000, batch_size=32)
    
    scores = []
    eps_history = []
    numGames = 50
    stack_size = 4
    
    log(f"Training DQN on Breakout for {numGames} games", "INFO")
    
    # Fill memory with random gameplay
    log("Filling memory with random gameplay...", "INFO")
    while agent.mem_cntr < agent.mem_size:
        done = False
        observation = env.reset()
        observation = observation[30:210, :160]
        observation = np.mean(observation, axis=2)
        
        stacked_frames = None
        stacked_frames = stack_frames(stacked_frames, observation, stack_size)
        
        while not done:
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action)
            
            observation_ = observation_[30:210, :160]
            observation_ = np.mean(observation_, axis=2)
            stacked_frames_ = stack_frames(stacked_frames, observation_, stack_size)
            
            action_onehot = np.zeros(env.action_space.n)
            action_onehot[action] = 1
            
            agent.store_transition(stacked_frames.reshape(180, 160, 4),
                                 action, reward,
                                 stacked_frames_.reshape(180, 160, 4), done)
            
            stacked_frames = stacked_frames_
    
    log("Memory filled. Starting training...", "SUCCESS")
    
    # Training loop
    for i in range(numGames):
        done = False
        observation = env.reset()
        observation = observation[30:210, :160]
        observation = np.mean(observation, axis=2)
        
        stacked_frames = None
        stacked_frames = stack_frames(stacked_frames, observation, stack_size)
        
        score = 0
        lastAction = 0
        
        while not done:
            if np.random.random() < 0.05:
                action = lastAction
            else:
                action = agent.choose_action(stacked_frames.reshape(180, 160, 4))
                lastAction = action
            
            observation_, reward, done, info = env.step(action)
            score += reward
            
            observation_ = observation_[30:210, :160]
            observation_ = np.mean(observation_, axis=2)
            stacked_frames_ = stack_frames(stacked_frames, observation_, stack_size)
            
            agent.store_transition(stacked_frames.reshape(180, 160, 4),
                                 action, reward,
                                 stacked_frames_.reshape(180, 160, 4), done)
            
            stacked_frames = stacked_frames_
            agent.learn()
        
        scores.append(score)
        eps_history.append(agent.epsilon)
        
        avg_score = np.mean(scores[max(0, i-10):(i+1)])
        log(f'Episode {i+1}, Score: {score:.1f}, Average: {avg_score:.1f}, Epsilon: {agent.epsilon:.4f}', "INFO")
        
        if (i+1) % 10 == 0:
            log(f"Checkpoint at episode {i+1}", "SUCCESS")
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    
    plt.subplot(1, 2, 2)
    plt.plot(eps_history)
    plt.title('Epsilon History')
    plt.xlabel('Episode')
    plt.ylabel('Epsilon')
    
    plt.tight_layout()
    plt.savefig('dqn_breakout_results.png')
    log("Results saved to dqn_breakout_results.png", "SUCCESS")

def run_policy_gradient_lunar_lander():
    """Run Policy Gradient on Lunar Lander"""
    log("STARTING POLICY GRADIENT ON LUNAR LANDER", "HEADER")
    
    env = gym.make('LunarLander-v2')
    
    agent = PolicyGradientAgent(lr=0.0005, gamma=0.99, n_actions=4,
                               layer1_size=64, layer2_size=64, input_dims=8)
    
    score_history = []
    num_episodes = 500
    
    log(f"Training Policy Gradient on Lunar Lander for {num_episodes} episodes", "INFO")
    
    for episode in range(num_episodes):
        done = False
        score = 0
        observation = env.reset()
        
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            agent.store_transition(observation, action, reward)
            observation = observation_
            score += reward
        
        score_history.append(score)
        agent.learn()
        
        avg_score = np.mean(score_history[max(0, episode-25):(episode+1)])
        
        if episode % 25 == 0:
            log(f'Episode {episode}, Score: {score:.1f}, Average: {avg_score:.1f}', "INFO")
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(score_history)
    plt.plot(np.convolve(score_history, np.ones(25)/25, mode='valid'))
    plt.title('Policy Gradient on Lunar Lander')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.legend(['Score', '25-Episode Average'])
    plt.savefig('policy_gradient_lunar_lander_results.png')
    log("Results saved to policy_gradient_lunar_lander_results.png", "SUCCESS")

def run_q_learning_grid_world():
    """Run Q-Learning on custom Grid World"""
    log("STARTING Q-LEARNING ON GRID WORLD", "HEADER")
    
    # Create Grid World
    magicSquares = {18: 54, 63: 14}
    env = GridWorld(9, 9, magicSquares)
    
    # Q-Learning parameters
    alpha = 0.1
    gamma = 1.0
    epsilon = 1.0
    
    # Initialize Q-table
    Q = {}
    for state in env.stateSpacePlus:
        for action in env.possibleActions:
            Q[(state, action)] = 0
    
    numGames = 50000
    totalRewards = np.zeros(numGames)
    
    log(f"Training Q-Learning on Grid World for {numGames} games", "INFO")
    
    for i in range(numGames):
        if i % 5000 == 0:
            log(f'Starting game {i}', "INFO")
        
        env.reset()
        done = False
        epRewards = 0
        observation = env.agentPosition
        
        while not done:
            rand = np.random.random()
            if rand < 1 - epsilon:
                action = max(env.possibleActions, 
                           key=lambda x: Q[(observation, x)])
            else:
                action = np.random.choice(env.possibleActions)
            
            observation_, reward, done, info = env.step(action)
            epRewards += reward
            
            action_ = max(env.possibleActions, 
                        key=lambda x: Q[(observation_, x)])
            
            Q[(observation, action)] = Q[(observation, action)] + \
                alpha * (reward + gamma * Q[(observation_, action_)] - Q[(observation, action)])
            
            observation = observation_
        
        if epsilon > 0.01:
            epsilon -= 2 / numGames
        
        totalRewards[i] = epRewards
    
    # Show final path
    log("Showing optimal path:", "INFO")
    env.reset()
    env.render()
    done = False
    
    while not done:
        action = max(env.possibleActions, 
                   key=lambda x: Q[(env.agentPosition, x)])
        _, _, done, _ = env.step(action)
        env.render()
        time.sleep(0.5)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(totalRewards)
    plt.title('Q-Learning on Grid World')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.savefig('q_learning_grid_world_results.png')
    log("Results saved to q_learning_grid_world_results.png", "SUCCESS")

def run_sarsa_cartpole():
    """Run SARSA on CartPole"""
    log("STARTING SARSA ON CARTPOLE", "HEADER")
    
    env = gym.make('CartPole-v1')
    
    # Discretize the state space
    n_bins = 10
    n_bins_angle = 10
    
    def get_state(observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        cart_pos_bin = np.digitize(cart_pos, np.linspace(-2.4, 2.4, n_bins))
        cart_vel_bin = np.digitize(cart_vel, np.linspace(-3, 3, n_bins))
        pole_angle_bin = np.digitize(pole_angle, np.linspace(-0.2, 0.2, n_bins_angle))
        pole_vel_bin = np.digitize(pole_vel, np.linspace(-2, 2, n_bins))
        
        return (cart_pos_bin, cart_vel_bin, pole_angle_bin, pole_vel_bin)
    
    # Initialize Q-table
    Q = {}
    for s1 in range(n_bins + 1):
        for s2 in range(n_bins + 1):
            for s3 in range(n_bins_angle + 1):
                for s4 in range(n_bins + 1):
                    for a in range(2):
                        Q[((s1, s2, s3, s4), a)] = 0
    
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    num_episodes = 5000
    rewards_history = []
    
    log(f"Training SARSA on CartPole for {num_episodes} episodes", "INFO")
    
    for episode in range(num_episodes):
        observation = env.reset()
        state = get_state(observation)
        
        # Choose initial action
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = max(range(2), key=lambda x: Q[(state, x)])
        
        done = False
        total_reward = 0
        
        while not done:
            observation_, reward, done, info = env.step(action)
            state_ = get_state(observation_)
            total_reward += reward
            
            # Choose next action
            if np.random.random() < epsilon:
                action_ = env.action_space.sample()
            else:
                action_ = max(range(2), key=lambda x: Q[(state_, x)])
            
            # SARSA update
            Q[(state, action)] = Q[(state, action)] + \
                alpha * (reward + gamma * Q[(state_, action_)] - Q[(state, action)])
            
            state = state_
            action = action_
        
        rewards_history.append(total_reward)
        
        # Decay epsilon
        if epsilon > 0.01:
            epsilon *= 0.995
        
        if episode % 100 == 0:
            avg_reward = np.mean(rewards_history[max(0, episode-100):episode+1])
            log(f'Episode {episode}, Average Reward: {avg_reward:.1f}, Epsilon: {epsilon:.4f}', "INFO")
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rewards_history)
    plt.plot(np.convolve(rewards_history, np.ones(100)/100, mode='valid'))
    plt.title('SARSA on CartPole')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend(['Reward', '100-Episode Average'])
    plt.savefig('sarsa_cartpole_results.png')
    log("Results saved to sarsa_cartpole_results.png", "SUCCESS")

def run_double_q_learning_cartpole():
    """Run Double Q-Learning on CartPole"""
    log("STARTING DOUBLE Q-LEARNING ON CARTPOLE", "HEADER")
    
    env = gym.make('CartPole-v1')
    
    # Discretize the state space
    n_bins = 10
    n_bins_angle = 10
    
    def get_state(observation):
        cart_pos, cart_vel, pole_angle, pole_vel = observation
        
        cart_pos_bin = np.digitize(cart_pos, np.linspace(-2.4, 2.4, n_bins))
        cart_vel_bin = np.digitize(cart_vel, np.linspace(-3, 3, n_bins))
        pole_angle_bin = np.digitize(pole_angle, np.linspace(-0.2, 0.2, n_bins_angle))
        pole_vel_bin = np.digitize(pole_vel, np.linspace(-2, 2, n_bins))
        
        return (cart_pos_bin, cart_vel_bin, pole_angle_bin, pole_vel_bin)
    
    # Initialize two Q-tables
    Q1 = {}
    Q2 = {}
    for s1 in range(n_bins + 1):
        for s2 in range(n_bins + 1):
            for s3 in range(n_bins_angle + 1):
                for s4 in range(n_bins + 1):
                    for a in range(2):
                        Q1[((s1, s2, s3, s4), a)] = 0
                        Q2[((s1, s2, s3, s4), a)] = 0
    
    alpha = 0.1
    gamma = 1.0
    epsilon = 1.0
    num_episodes = 10000
    rewards_history = []
    
    log(f"Training Double Q-Learning on CartPole for {num_episodes} episodes", "INFO")
    
    for episode in range(num_episodes):
        observation = env.reset()
        state = get_state(observation)
        
        done = False
        total_reward = 0
        
        while not done:
            # Choose action using sum of Q1 and Q2
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = max(range(2), key=lambda x: Q1[(state, x)] + Q2[(state, x)])
            
            observation_, reward, done, info = env.step(action)
            state_ = get_state(observation_)
            total_reward += reward
            
            # Update Q1 or Q2 with 50% probability
            if np.random.random() < 0.5:
                # Update Q1
                max_action = max(range(2), key=lambda x: Q1[(state_, x)])
                Q1[(state, action)] = Q1[(state, action)] + \
                    alpha * (reward + gamma * Q2[(state_, max_action)] - Q1[(state, action)])
            else:
                # Update Q2
                max_action = max(range(2), key=lambda x: Q2[(state_, x)])
                Q2[(state, action)] = Q2[(state, action)] + \
                    alpha * (reward + gamma * Q1[(state_, max_action)] - Q2[(state, action)])
            
            state = state_
        
        rewards_history.append(total_reward)
        
        # Decay epsilon
        if epsilon > 0.01:
            epsilon -= 2 / num_episodes
        
        if episode % 500 == 0:
            avg_reward = np.mean(rewards_history[max(0, episode-100):episode+1])
            log(f'Episode {episode}, Average Reward: {avg_reward:.1f}, Epsilon: {epsilon:.4f}', "INFO")
    
    env.close()
    
    # Plot results
    plt.figure(figsize=(10, 6))
    running_avg = np.convolve(rewards_history, np.ones(100)/100, mode='valid')
    plt.plot(running_avg)
    plt.title('Double Q-Learning on CartPole (Running Average)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward (100 episodes)')
    plt.savefig('double_q_learning_cartpole_results.png')
    log("Results saved to double_q_learning_cartpole_results.png", "SUCCESS")

def main_menu():
    """Display main menu and handle user selection"""
    while True:
        print("\n" + "="*60)
        print(Colors.HEADER + Colors.BOLD + "REINFORCEMENT LEARNING TUTORIAL MENU" + Colors.ENDC)
        print("="*60)
        print("\nSelect an algorithm to run:\n")
        print("1. Deep Q-Learning on Breakout")
        print("2. Policy Gradient on Lunar Lander")
        print("3. Q-Learning on Custom Grid World")
        print("4. SARSA on CartPole")
        print("5. Double Q-Learning on CartPole")
        print("6. Exit")
        print("\n" + "="*60)
        
        try:
            choice = input("\nEnter your choice (1-6): ")
            
            if choice == '1':
                run_dqn_breakout()
            elif choice == '2':
                run_policy_gradient_lunar_lander()
            elif choice == '3':
                run_q_learning_grid_world()
            elif choice == '4':
                run_sarsa_cartpole()
            elif choice == '5':
                run_double_q_learning_cartpole()
            elif choice == '6':
                log("Exiting program. Thank you!", "SUCCESS")
                break
            else:
                log("Invalid choice. Please enter a number between 1 and 6.", "WARNING")
                
        except KeyboardInterrupt:
            log("\nProgram interrupted by user", "WARNING")
            break
        except Exception as e:
            log(f"An error occurred: {str(e)}", "ERROR")
            log("Please check the error and try again.", "WARNING")

def main():
    """Main entry point"""
    log("REINFORCEMENT LEARNING TUTORIAL SCRIPT", "HEADER")
    log("Based on Machine Learning with Phil tutorials", "INFO")
    
    # Check and install dependencies
    if check_and_install_dependencies():
        import_libraries()
        
        # Create necessary directories
        os.makedirs('tmp', exist_ok=True)
        log("Created temporary directory for checkpoints", "SUCCESS")
        
        # Display main menu
        main_menu()
    else:
        log("Failed to install dependencies. Exiting.", "ERROR")
        sys.exit(1)

if __name__ == "__main__":
    main()