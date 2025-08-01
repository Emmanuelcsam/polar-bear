#!/usr/bin/env python3

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path
import logging
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random

# Import the original detection module
from detection import OmniFiberAnalyzer, OmniConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s'
)


class DQN(nn.Module):
    """Deep Q-Network for learning optimal detection parameters"""
    
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class DetectionEnvironment:
    """RL Environment wrapper for the anomaly detection system"""
    
    def __init__(self, analyzer: OmniFiberAnalyzer, validation_data: Dict):
        self.analyzer = analyzer
        self.validation_data = validation_data  # Ground truth labels
        self.logger = logging.getLogger(__name__)
        
        # Define parameter ranges for RL to optimize
        self.param_ranges = {
            'confidence_threshold': (0.1, 0.9),
            'min_defect_size': (5, 50),
            'max_defect_size': (1000, 10000),
            'anomaly_threshold_multiplier': (1.5, 4.0),
            'morph_kernel_size': (3, 11),  # For morphological operations
            'canny_low': (20, 80),
            'canny_high': (80, 200),
        }
        
        # State representation
        self.state_size = 15  # Current params + performance metrics
        self.action_size = 15  # Increase/decrease/keep each parameter
        
        # Performance tracking
        self.current_metrics = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
        }
        
        self.reset()
        
    def reset(self):
        """Reset environment to initial state"""
        # Initialize with default parameters
        self.current_params = {
            'confidence_threshold': 0.3,
            'min_defect_size': 10,
            'max_defect_size': 5000,
            'anomaly_threshold_multiplier': 2.5,
            'morph_kernel_size': 5,
            'canny_low': 30,
            'canny_high': 100,
        }
        
        # Update analyzer config
        self._update_analyzer_config()
        
        # Reset metrics
        self.episode_detections = []
        self.steps = 0
        
        return self._get_state()
        
    def _update_analyzer_config(self):
        """Update analyzer configuration with current parameters"""
        self.analyzer.config.confidence_threshold = self.current_params['confidence_threshold']
        self.analyzer.config.min_defect_size = int(self.current_params['min_defect_size'])
        self.analyzer.config.max_defect_size = int(self.current_params['max_defect_size'])
        self.analyzer.config.anomaly_threshold_multiplier = self.current_params['anomaly_threshold_multiplier']
        
    def _get_state(self):
        """Get current state representation"""
        state = []
        
        # Add normalized parameter values
        for param, value in self.current_params.items():
            min_val, max_val = self.param_ranges[param]
            normalized = (value - min_val) / (max_val - min_val)
            state.append(normalized)
        
        # Add performance metrics
        state.extend([
            self.current_metrics['precision'],
            self.current_metrics['recall'],
            self.current_metrics['f1_score'],
            self.current_metrics['true_positives'] / (self.steps + 1),
            self.current_metrics['false_positives'] / (self.steps + 1),
            self.current_metrics['false_negatives'] / (self.steps + 1),
            min(self.steps / 100, 1.0),  # Normalized episode progress
            np.random.random(),  # Noise for exploration
        ])
        
        return np.array(state, dtype=np.float32)
        
    def step(self, action):
        """Execute action and return (state, reward, done, info)"""
        # Decode action (3 options per parameter: decrease, keep, increase)
        param_actions = self._decode_action(action)
        
        # Apply parameter changes
        for param, change in param_actions.items():
            if change == 'increase':
                self._adjust_parameter(param, 1.1)  # Increase by 10%
            elif change == 'decrease':
                self._adjust_parameter(param, 0.9)  # Decrease by 10%
            # 'keep' means no change
        
        # Update analyzer configuration
        self._update_analyzer_config()
        
        # Evaluate on a batch of validation images
        batch_reward = 0
        batch_size = min(5, len(self.validation_data['images']) - self.steps)
        
        for i in range(batch_size):
            if self.steps >= len(self.validation_data['images']):
                break
                
            image_path = self.validation_data['images'][self.steps]
            ground_truth = self.validation_data['labels'][self.steps]
            
            # Run detection
            detection_results = self._run_detection(image_path)
            
            # Calculate reward based on detection performance
            reward = self._calculate_reward(detection_results, ground_truth)
            batch_reward += reward
            
            self.steps += 1
            
        # Average reward over batch
        reward = batch_reward / max(batch_size, 1)
        
        # Check if episode is done
        done = self.steps >= len(self.validation_data['images'])
        
        # Get new state
        new_state = self._get_state()
        
        # Additional info
        info = {
            'metrics': self.current_metrics.copy(),
            'params': self.current_params.copy(),
        }
        
        return new_state, reward, done, info
        
    def _decode_action(self, action):
        """Decode discrete action into parameter changes"""
        param_actions = {}
        param_names = list(self.param_ranges.keys())
        
        # Each parameter can have 3 actions: decrease, keep, increase
        # We'll use a simple encoding where action determines all parameters
        # In practice, you might want more sophisticated action encoding
        
        # Simple approach: use action to determine change pattern
        action_pattern = action % 7  # 7 different patterns
        
        for i, param in enumerate(param_names):
            if action_pattern == 0:  # Increase all
                param_actions[param] = 'increase'
            elif action_pattern == 1:  # Decrease all
                param_actions[param] = 'decrease'
            elif action_pattern == 2:  # Alternate
                param_actions[param] = 'increase' if i % 2 == 0 else 'decrease'
            elif action_pattern == 3:  # Focus on thresholds
                if 'threshold' in param:
                    param_actions[param] = 'increase' if action > 7 else 'decrease'
                else:
                    param_actions[param] = 'keep'
            elif action_pattern == 4:  # Focus on sizes
                if 'size' in param:
                    param_actions[param] = 'increase' if action > 7 else 'decrease'
                else:
                    param_actions[param] = 'keep'
            else:
                param_actions[param] = 'keep'
                
        return param_actions
        
    def _adjust_parameter(self, param, factor):
        """Adjust parameter value within valid range"""
        min_val, max_val = self.param_ranges[param]
        current_val = self.current_params[param]
        new_val = current_val * factor
        
        # Clamp to valid range
        self.current_params[param] = max(min_val, min(max_val, new_val))
        
    def _run_detection(self, image_path):
        """Run anomaly detection on image"""
        try:
            # Create temporary output directory
            temp_dir = Path("temp_rl_detection")
            temp_dir.mkdir(exist_ok=True)
            
            # Run detection
            results = self.analyzer.analyze_end_face(image_path, str(temp_dir))
            
            # Clean up temp files
            for file in temp_dir.glob("*"):
                file.unlink()
            temp_dir.rmdir()
            
            return results
        except Exception as e:
            self.logger.error(f"Detection failed: {e}")
            return None
            
    def _calculate_reward(self, detection_results, ground_truth):
        """Calculate reward based on detection accuracy"""
        if detection_results is None:
            return -10.0  # Penalty for failed detection
            
        # Extract detected defects
        detected_defects = detection_results.get('defects', [])
        true_defects = ground_truth.get('defects', [])
        
        # Simple matching based on location proximity
        tp = 0  # True positives
        matched_true = set()
        
        for detected in detected_defects:
            det_x, det_y = detected['location_xy']
            
            for i, true_def in enumerate(true_defects):
                if i in matched_true:
                    continue
                    
                true_x, true_y = true_def['location_xy']
                distance = np.sqrt((det_x - true_x)**2 + (det_y - true_y)**2)
                
                if distance < 20:  # Within 20 pixels
                    tp += 1
                    matched_true.add(i)
                    break
        
        fp = len(detected_defects) - tp  # False positives
        fn = len(true_defects) - tp      # False negatives
        
        # Update metrics
        self.current_metrics['true_positives'] += tp
        self.current_metrics['false_positives'] += fp
        self.current_metrics['false_negatives'] += fn
        
        # Calculate precision, recall, F1
        total_detected = self.current_metrics['true_positives'] + self.current_metrics['false_positives']
        total_true = self.current_metrics['true_positives'] + self.current_metrics['false_negatives']
        
        if total_detected > 0:
            self.current_metrics['precision'] = self.current_metrics['true_positives'] / total_detected
        else:
            self.current_metrics['precision'] = 0.0
            
        if total_true > 0:
            self.current_metrics['recall'] = self.current_metrics['true_positives'] / total_true
        else:
            self.current_metrics['recall'] = 0.0
            
        if self.current_metrics['precision'] + self.current_metrics['recall'] > 0:
            self.current_metrics['f1_score'] = 2 * (self.current_metrics['precision'] * self.current_metrics['recall']) / \
                                               (self.current_metrics['precision'] + self.current_metrics['recall'])
        else:
            self.current_metrics['f1_score'] = 0.0
        
        # Calculate reward
        # Reward = weighted combination of precision, recall, and penalties
        reward = 0.0
        
        # F1 score component (main objective)
        reward += 10.0 * self.current_metrics['f1_score']
        
        # Penalties for false positives and negatives
        reward -= 2.0 * fp
        reward -= 3.0 * fn  # False negatives are worse
        
        # Bonus for true positives
        reward += 5.0 * tp
        
        # Small penalty for extreme parameter values
        for param, value in self.current_params.items():
            min_val, max_val = self.param_ranges[param]
            normalized = (value - min_val) / (max_val - min_val)
            if normalized < 0.1 or normalized > 0.9:
                reward -= 0.5
        
        return reward


class RLDetectionAgent:
    """Deep Q-Learning agent for optimizing detection parameters"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.gamma = 0.95  # Discount factor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Neural networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Update target network
        self.update_target_network()
        
    def update_target_network(self):
        """Copy weights from main network to target network"""
        self.target_network.load_state_dict(self.q_network.state_dict())
        
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if random.random() <= self.epsilon:
            return random.randrange(self.action_size)
            
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        q_values = self.q_network(state_tensor)
        return np.argmax(q_values.cpu().data.numpy())
        
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
            
        batch = random.sample(self.memory, batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in batch]).to(self.device)
        
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_network(next_states).max(1)[0].detach()
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
        }, filepath)
        
    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']


def create_validation_data(image_dir, labels_file):
    """Load validation data with ground truth labels"""
    # This is a placeholder - you'll need to implement based on your data format
    validation_data = {
        'images': [],
        'labels': []
    }
    
    # Example: Load from JSON file
    if os.path.exists(labels_file):
        with open(labels_file, 'r') as f:
            labels_data = json.load(f)
            
        for item in labels_data:
            image_path = os.path.join(image_dir, item['filename'])
            if os.path.exists(image_path):
                validation_data['images'].append(image_path)
                validation_data['labels'].append(item)
    
    return validation_data


def train_rl_detection(episodes=100, checkpoint_dir="rl_checkpoints"):
    """Train the RL-enhanced detection system"""
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Initialize base analyzer
    config = OmniConfig()
    analyzer = OmniFiberAnalyzer(config)
    
    # Load validation data (you'll need to prepare this)
    validation_data = create_validation_data("validation_images", "validation_labels.json")
    
    if not validation_data['images']:
        logging.error("No validation data found!")
        return
    
    # Create environment and agent
    env = DetectionEnvironment(analyzer, validation_data)
    agent = RLDetectionAgent(env.state_size, env.action_size)
    
    # Training metrics
    episode_rewards = []
    episode_f1_scores = []
    
    # Training loop
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            total_reward += reward
            
            # Train model
            if len(agent.memory) > 32:
                agent.replay()
        
        # Update target network every 10 episodes
        if episode % 10 == 0:
            agent.update_target_network()
        
        # Save checkpoint every 20 episodes
        if episode % 20 == 0:
            checkpoint_path = Path(checkpoint_dir) / f"checkpoint_ep{episode}.pth"
            agent.save(checkpoint_path)
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_f1_scores.append(info['metrics']['f1_score'])
        
        # Log progress
        logging.info(f"Episode {episode + 1}/{episodes}")
        logging.info(f"Total Reward: {total_reward:.2f}")
        logging.info(f"F1 Score: {info['metrics']['f1_score']:.3f}")
        logging.info(f"Epsilon: {agent.epsilon:.3f}")
        logging.info(f"Best Params: {info['params']}")
        logging.info("-" * 50)
    
    # Plot training progress
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    
    plt.subplot(1, 2, 2)
    plt.plot(episode_f1_scores)
    plt.title('F1 Score Progress')
    plt.xlabel('Episode')
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig('rl_training_progress.png')
    plt.close()
    
    # Save final model
    final_path = Path(checkpoint_dir) / "final_model.pth"
    agent.save(final_path)
    
    # Return best parameters
    return info['params']


def apply_rl_optimized_detection(image_path, checkpoint_path, output_dir):
    """Apply detection with RL-optimized parameters"""
    # Initialize base analyzer
    config = OmniConfig()
    analyzer = OmniFiberAnalyzer(config)
    
    # Load dummy validation data (just for environment init)
    validation_data = {'images': [image_path], 'labels': [{}]}
    
    # Create environment and agent
    env = DetectionEnvironment(analyzer, validation_data)
    agent = RLDetectionAgent(env.state_size, env.action_size)
    
    # Load trained model
    agent.load(checkpoint_path)
    agent.epsilon = 0  # No exploration during inference
    
    # Get optimal parameters
    state = env.reset()
    action = agent.act(state)
    env.step(action)
    
    # Apply optimized parameters
    optimized_params = env.current_params
    logging.info(f"Using RL-optimized parameters: {optimized_params}")
    
    # Update analyzer with optimized parameters
    analyzer.config.confidence_threshold = optimized_params['confidence_threshold']
    analyzer.config.min_defect_size = int(optimized_params['min_defect_size'])
    analyzer.config.max_defect_size = int(optimized_params['max_defect_size'])
    analyzer.config.anomaly_threshold_multiplier = optimized_params['anomaly_threshold_multiplier']
    
    # Run detection
    results = analyzer.analyze_end_face(image_path, output_dir)
    
    return results, optimized_params


def main():
    """Main function demonstrating RL-enhanced detection"""
    print("\n" + "="*80)
    print("RL-ENHANCED OMNIFIBER ANALYZER".center(80))
    print("="*80)
    
    # Training mode
    mode = input("\nSelect mode:\n1. Train RL agent\n2. Apply trained model\nChoice (1/2): ").strip()
    
    if mode == '1':
        # Train the RL agent
        print("\nStarting RL training...")
        best_params = train_rl_detection(episodes=50)
        print(f"\nTraining complete! Best parameters: {best_params}")
        
    elif mode == '2':
        # Apply trained model
        image_path = input("\nEnter image path: ").strip().strip('"\'')
        checkpoint_path = input("Enter checkpoint path (or press Enter for latest): ").strip()
        
        if not checkpoint_path:
            # Find latest checkpoint
            checkpoint_dir = Path("rl_checkpoints")
            if checkpoint_dir.exists():
                checkpoints = list(checkpoint_dir.glob("checkpoint_ep*.pth"))
                if checkpoints:
                    checkpoint_path = str(max(checkpoints, key=lambda p: int(p.stem.split('ep')[1])))
                else:
                    checkpoint_path = str(checkpoint_dir / "final_model.pth")
        
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
        
        # Create output directory
        output_dir = f"rl_detection_output_{Path(image_path).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
        
        # Run detection
        print(f"\nAnalyzing {image_path} with RL-optimized parameters...")
        results, params = apply_rl_optimized_detection(image_path, checkpoint_path, output_dir)
        
        print(f"\nResults saved to: {output_dir}/")
        print(f"Optimized parameters used: {params}")
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
