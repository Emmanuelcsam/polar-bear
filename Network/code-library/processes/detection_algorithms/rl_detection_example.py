#!/usr/bin/env python3
"""
Example script showing how to use the RL-enhanced detection system
"""

import json
import os
from pathlib import Path
import numpy as np
from rl_detection import train_rl_detection, apply_rl_optimized_detection

def prepare_validation_data():
    """
    Example of preparing validation data with ground truth labels
    """
    # Create validation data structure
    validation_data = []
    
    # Example: Add labeled images
    # In practice, you would load this from your annotation tool
    validation_data.append({
        'filename': 'fiber_sample_001.jpg',
        'defects': [
            {
                'defect_id': 'DEF_001',
                'defect_type': 'SCRATCH',
                'location_xy': [125, 89],
                'bbox': [120, 85, 10, 8],
                'area_px': 80,
                'severity': 'MEDIUM'
            },
            {
                'defect_id': 'DEF_002',
                'defect_type': 'DIG',
                'location_xy': [200, 150],
                'bbox': [195, 145, 10, 10],
                'area_px': 100,
                'severity': 'HIGH'
            }
        ]
    })
    
    validation_data.append({
        'filename': 'fiber_sample_002.jpg',
        'defects': [
            {
                'defect_id': 'DEF_003',
                'defect_type': 'CONTAMINATION',
                'location_xy': [50, 50],
                'bbox': [40, 40, 20, 20],
                'area_px': 400,
                'severity': 'LOW'
            }
        ]
    })
    
    # Add more labeled examples...
    
    # Save to JSON file
    with open('validation_labels.json', 'w') as f:
        json.dump(validation_data, f, indent=2)
    
    print(f"Created validation data with {len(validation_data)} labeled images")
    
    
def demonstrate_rl_training():
    """
    Demonstrate how to train the RL agent
    """
    print("\n" + "="*60)
    print("RL TRAINING DEMONSTRATION")
    print("="*60)
    
    # First, prepare validation data
    prepare_validation_data()
    
    # Train the RL agent
    print("\nStarting RL training...")
    print("This will optimize detection parameters based on validation data")
    
    # Run training for 30 episodes (reduced for demo)
    best_params = train_rl_detection(episodes=30, checkpoint_dir="demo_checkpoints")
    
    print("\nTraining complete!")
    print("Best parameters found:")
    for param, value in best_params.items():
        print(f"  {param}: {value:.3f}")
    
    return best_params


def demonstrate_inference():
    """
    Demonstrate how to use trained RL model for detection
    """
    print("\n" + "="*60)
    print("RL INFERENCE DEMONSTRATION")
    print("="*60)
    
    # Example test image
    test_image = "test_fiber_image.jpg"
    
    # Check if we have a trained model
    checkpoint_path = "demo_checkpoints/final_model.pth"
    if not os.path.exists(checkpoint_path):
        print("No trained model found. Running training first...")
        demonstrate_rl_training()
    
    # Apply RL-optimized detection
    print(f"\nAnalyzing {test_image} with RL-optimized parameters...")
    
    output_dir = "demo_output"
    results, optimized_params = apply_rl_optimized_detection(
        test_image, 
        checkpoint_path, 
        output_dir
    )
    
    # Display results
    print("\nDetection Results:")
    print(f"Total defects found: {len(results['defects'])}")
    print(f"Overall quality score: {results['overall_quality_score']:.1f}")
    
    for i, defect in enumerate(results['defects'][:5]):  # Show first 5
        print(f"\nDefect {i+1}:")
        print(f"  Type: {defect['defect_type']}")
        print(f"  Location: {defect['location_xy']}")
        print(f"  Confidence: {defect['confidence']:.2f}")
        print(f"  Severity: {defect['severity']}")
    
    print(f"\nFull results saved to: {output_dir}/")


def compare_with_baseline():
    """
    Compare RL-optimized detection with default parameters
    """
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    
    from detection import OmniFiberAnalyzer, OmniConfig
    
    test_image = "test_fiber_image.jpg"
    
    # Run with default parameters
    print("\n1. Running with DEFAULT parameters...")
    default_config = OmniConfig()
    default_analyzer = OmniFiberAnalyzer(default_config)
    default_results = default_analyzer.analyze_end_face(test_image, "default_output")
    
    # Run with RL-optimized parameters
    print("\n2. Running with RL-OPTIMIZED parameters...")
    rl_results, rl_params = apply_rl_optimized_detection(
        test_image,
        "demo_checkpoints/final_model.pth",
        "rl_output"
    )
    
    # Compare results
    print("\n" + "-"*40)
    print("COMPARISON RESULTS:")
    print("-"*40)
    print(f"Default - Defects found: {len(default_results['defects'])}")
    print(f"RL-Optimized - Defects found: {len(rl_results['defects'])}")
    print(f"\nDefault - Quality score: {default_results['overall_quality_score']:.1f}")
    print(f"RL-Optimized - Quality score: {rl_results['overall_quality_score']:.1f}")
    
    # Show parameter differences
    print("\nParameter differences:")
    print(f"Confidence threshold: {default_config.confidence_threshold} -> {rl_params['confidence_threshold']:.3f}")
    print(f"Min defect size: {default_config.min_defect_size} -> {int(rl_params['min_defect_size'])}")
    print(f"Anomaly multiplier: {default_config.anomaly_threshold_multiplier} -> {rl_params['anomaly_threshold_multiplier']:.2f}")


def continuous_learning_example():
    """
    Example of how to implement continuous learning
    """
    print("\n" + "="*60)
    print("CONTINUOUS LEARNING EXAMPLE")
    print("="*60)
    
    # Load existing model
    from rl_detection import RLDetectionAgent, DetectionEnvironment
    from detection import OmniFiberAnalyzer, OmniConfig
    
    # Initialize components
    config = OmniConfig()
    analyzer = OmniFiberAnalyzer(config)
    
    # Load new validation data (simulated)
    new_validation_data = {
        'images': ['new_image_1.jpg', 'new_image_2.jpg'],
        'labels': [
            {'defects': [{'location_xy': [100, 100], 'type': 'scratch'}]},
            {'defects': [{'location_xy': [150, 150], 'type': 'dig'}]}
        ]
    }
    
    # Create environment with new data
    env = DetectionEnvironment(analyzer, new_validation_data)
    agent = RLDetectionAgent(env.state_size, env.action_size)
    
    # Load previous model
    checkpoint_path = "demo_checkpoints/final_model.pth"
    if os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        print("Loaded existing model for continuous learning")
    
    # Continue training with reduced learning rate
    agent.learning_rate = 0.0001  # Lower learning rate for fine-tuning
    agent.epsilon = 0.1  # Some exploration, but mostly exploitation
    
    print("\nFine-tuning model with new data...")
    
    # Train for fewer episodes since we're fine-tuning
    for episode in range(10):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > 32:
                agent.replay()
        
        print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, F1 = {info['metrics']['f1_score']:.3f}")
    
    # Save updated model
    agent.save("demo_checkpoints/updated_model.pth")
    print("\nContinuous learning complete! Model updated with new data.")


def main():
    """
    Main demonstration menu
    """
    print("\n" + "="*80)
    print("RL-ENHANCED DETECTION SYSTEM EXAMPLES".center(80))
    print("="*80)
    
    while True:
        print("\nSelect a demonstration:")
        print("1. Train RL agent from scratch")
        print("2. Run inference with trained model")
        print("3. Compare RL vs default parameters")
        print("4. Continuous learning example")
        print("5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == '1':
            demonstrate_rl_training()
        elif choice == '2':
            demonstrate_inference()
        elif choice == '3':
            compare_with_baseline()
        elif choice == '4':
            continuous_learning_example()
        elif choice == '5':
            print("\nExiting...")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()
