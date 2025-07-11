# RL-Enhanced Fiber Anomaly Detection System

## Overview

This system enhances the original OmniFiberAnalyzer with Reinforcement Learning (RL) capabilities, using Deep Q-Learning to automatically optimize detection parameters based on performance feedback.

## Key Components

### 1. **Deep Q-Network (DQN)**
- A neural network that learns to predict the value of different parameter configurations
- 4-layer fully connected network with ReLU activations
- Implements the Double DQN technique to reduce overestimation bias

### 2. **Detection Environment**
- Wraps the original anomaly detector as an RL environment
- **State Space**: Current parameters + performance metrics (15 dimensions)
- **Action Space**: Parameter adjustments (15 discrete actions)
- **Reward Function**: Based on F1 score, with penalties for false positives/negatives

### 3. **RL Agent**
- Uses epsilon-greedy exploration strategy
- Experience replay for stable learning
- Target network updates every 10 episodes

## Parameters Optimized by RL

1. **confidence_threshold**: (0.1 - 0.9) - Minimum confidence for anomaly detection
2. **min_defect_size**: (5 - 50 pixels) - Minimum defect area
3. **max_defect_size**: (1000 - 10000 pixels) - Maximum defect area
4. **anomaly_threshold_multiplier**: (1.5 - 4.0) - Statistical threshold multiplier
5. **morph_kernel_size**: (3 - 11) - Morphological operation kernel size
6. **canny_low/high**: Edge detection thresholds

## How It Works

### Training Phase
1. Agent starts with default parameters
2. For each episode:
   - Runs detection on validation images
   - Compares results with ground truth
   - Receives rewards based on detection accuracy
   - Updates neural network to improve parameter selection
3. Gradually reduces exploration (epsilon decay)
4. Saves checkpoints periodically

### Inference Phase
1. Loads trained model
2. Selects optimal parameters based on learned policy
3. Applies detection with optimized parameters

## Reward Design

The reward function balances multiple objectives:
- **Positive rewards**:
  - +10 × F1 score (main objective)
  - +5 per true positive detection
- **Penalties**:
  - -2 per false positive
  - -3 per false negative (more severe)
  - -0.5 for extreme parameter values
  - -10 for detection failures

## Usage

### Training the RL Agent

```python
# Prepare validation data with ground truth labels
validation_data = {
    'images': ['path/to/image1.jpg', 'path/to/image2.jpg', ...],
    'labels': [
        {
            'defects': [
                {'location_xy': [x1, y1], 'type': 'scratch'},
                {'location_xy': [x2, y2], 'type': 'dig'},
                ...
            ]
        },
        ...
    ]
}

# Train the agent
best_params = train_rl_detection(episodes=100)
```

### Applying Trained Model

```python
# Load and apply optimized parameters
results, params = apply_rl_optimized_detection(
    image_path="test_image.jpg",
    checkpoint_path="rl_checkpoints/final_model.pth",
    output_dir="output"
)
```

## Advantages Over Fixed Parameters

1. **Adaptive Optimization**: Learns optimal parameters for your specific dataset
2. **Continuous Improvement**: Can be retrained as new data becomes available
3. **Robust Performance**: Explores parameter space systematically
4. **Automated Tuning**: No manual parameter tweaking required

## Integration with Original System

The RL system is designed as a wrapper around the original `OmniFiberAnalyzer`:
- Preserves all original functionality
- Can switch between RL-optimized and manual parameters
- Backward compatible with existing pipelines

## Performance Considerations

- Training requires GPU for faster neural network updates
- Initial training may take 1-2 hours for 100 episodes
- Inference adds minimal overhead (<100ms per image)
- Memory usage: ~2GB for replay buffer

## Future Enhancements

1. **Multi-objective RL**: Optimize for multiple metrics simultaneously
2. **Continuous Actions**: Use DDPG for fine-grained parameter control
3. **Online Learning**: Adapt parameters during deployment
4. **Transfer Learning**: Pre-train on synthetic data
5. **Ensemble Methods**: Combine multiple RL agents

## Troubleshooting

### Low F1 Scores
- Increase training episodes
- Adjust reward weights
- Ensure sufficient validation data diversity

### Unstable Training
- Reduce learning rate
- Increase batch size
- Add gradient clipping

### Parameter Oscillation
- Decrease action magnitude
- Increase epsilon decay rate
- Add momentum to parameter updates
