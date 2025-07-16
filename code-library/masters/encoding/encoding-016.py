#!/usr/bin/env python3
"""
Enhanced Training Script with Full Integration Support
Maintains backward compatibility and independent running capability
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import time
import os

# Try to import integration capabilities
try:
    from integration_wrapper import *
    inject_integration()
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    # Define stubs for independent running
    def report_progress(p, m=''): print(f'Progress: {p*100:.1f}% - {m}')
    def get_parameter(n, d=None): return globals().get(n, d)
    def set_parameter(n, v): globals()[n] = v
    def update_state(k, v): pass
    def get_state(k, d=None): return d
    def publish_event(t, d, target=None): pass
    def integrated(f): return f

# Define PixelGenerator locally to avoid import issues
class PixelGenerator(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(size, size, 3, 256))

    def get_probs(self):
        return torch.softmax(self.logits, dim=3)

# --- Configuration ---
IMG_SIZE = get_parameter('IMG_SIZE', 128)
LEARNING_RATE = get_parameter('LEARNING_RATE', 0.1)
MAX_ITERATIONS = get_parameter('MAX_ITERATIONS', 1000)
SAVE_INTERVAL = get_parameter('SAVE_INTERVAL', 100)
VISUALIZATION_INTERVAL = get_parameter('VISUALIZATION_INTERVAL', 10)
HEADLESS_MODE = get_parameter('HEADLESS_MODE', False)

@integrated
def load_model():
    """Load or create the model"""
    model = PixelGenerator(IMG_SIZE)
    
    model_path = get_parameter('MODEL_PATH', 'generator_model.pth')
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path))
            publish_event('info', {'message': f'Loaded model from {model_path}'})
        except Exception as e:
            publish_event('warning', {'message': f'Could not load model: {e}'})
    else:
        publish_event('info', {'message': 'Created new model'})
    
    # Store in shared state
    update_state('model', model)
    return model

@integrated
def load_target_data():
    """Load target data"""
    target_path = get_parameter('TARGET_PATH', 'target_data.pt')
    
    if os.path.exists(target_path):
        target = torch.load(target_path)
        publish_event('info', {'message': f'Loaded target from {target_path}'})
    else:
        # Create random target for testing
        target = torch.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3))
        publish_event('warning', {'message': 'Created random target data'})
    
    # Store in shared state
    update_state('target', target)
    return target

@integrated
def train_step(model, optimizer, loss_fn, target):
    """Perform a single training step"""
    optimizer.zero_grad()
    
    # Calculate Loss
    logits_flat = model.logits.view(-1, 256)
    target_flat = target.view(-1)
    loss = loss_fn(logits_flat, target_flat)
    
    # Backpropagation and Optimization
    loss.backward()
    optimizer.step()
    
    return loss.item()

@integrated
def visualize(model, target, iteration):
    """Visualize current generation"""
    with torch.no_grad():
        probs = model.get_probs()
        dist = torch.distributions.Categorical(probs)
        sample = dist.sample()
        
        # Prepare visualization
        generated_np = sample.cpu().numpy().astype(np.uint8)
        target_np = target.cpu().numpy().astype(np.uint8)
        
        # Create side-by-side comparison
        comparison = np.hstack([generated_np, target_np])
        
        if not HEADLESS_MODE:
            cv2.imshow('Training Progress (Generated | Target)', comparison)
            key = cv2.waitKey(1)
            return key == ord('q')
        else:
            # Save to file in headless mode
            cv2.imwrite(f'training_progress_{iteration}.png', comparison)
            return False

@integrated
def save_model(model, iteration):
    """Save model checkpoint"""
    checkpoint_path = f'checkpoint_iter_{iteration}.pth'
    torch.save(model.state_dict(), checkpoint_path)
    publish_event('info', {'message': f'Saved checkpoint: {checkpoint_path}'})
    
    # Also update the main model file
    model_path = get_parameter('MODEL_PATH', 'generator_model.pth')
    torch.save(model.state_dict(), model_path)

@integrated
def train_model():
    """Main training function"""
    publish_event('status', {'status': 'initializing'})
    
    # Setup
    model = load_model()
    target = load_target_data()
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()
    
    # Store in shared state
    update_state('optimizer', optimizer)
    update_state('loss_fn', loss_fn)
    
    publish_event('status', {'status': 'training'})
    print(f"Starting training... Max iterations: {MAX_ITERATIONS}")
    if not HEADLESS_MODE:
        print("Press 'q' in the window to stop.")
    
    iteration = 0
    start_time = time.time()
    losses = []
    
    try:
        while iteration < MAX_ITERATIONS:
            # Training step
            loss = train_step(model, optimizer, loss_fn, target)
            losses.append(loss)
            
            # Progress reporting
            progress = iteration / MAX_ITERATIONS
            avg_loss = np.mean(losses[-100:]) if losses else 0
            report_progress(progress, f'Iter {iteration}, Loss: {avg_loss:.4f}')
            
            # Visualization
            if iteration % VISUALIZATION_INTERVAL == 0:
                should_stop = visualize(model, target, iteration)
                if should_stop:
                    publish_event('info', {'message': 'Training stopped by user'})
                    break
            
            # Save checkpoint
            if iteration % SAVE_INTERVAL == 0 and iteration > 0:
                save_model(model, iteration)
            
            iteration += 1
            
            # Check for external stop signal
            if get_state('stop_training', False):
                publish_event('info', {'message': 'Training stopped by external signal'})
                break
                
    except KeyboardInterrupt:
        publish_event('info', {'message': 'Training interrupted'})
    except Exception as e:
        publish_event('error', {'message': f'Training error: {e}'})
        raise
    finally:
        # Final save
        save_model(model, iteration)
        
        # Cleanup
        if not HEADLESS_MODE:
            cv2.destroyAllWindows()
        
        # Report completion
        duration = time.time() - start_time
        final_loss = np.mean(losses[-100:]) if losses else 0
        
        publish_event('status', {
            'status': 'completed',
            'iterations': iteration,
            'duration': duration,
            'final_loss': final_loss
        })
        
        print(f"\nTraining completed:")
        print(f"  Iterations: {iteration}")
        print(f"  Duration: {duration:.2f} seconds")
        print(f"  Final loss: {final_loss:.4f}")

# Allow direct execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Enhanced PyTorch Training Script')
    parser.add_argument('--size', type=int, default=128, help='Image size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--iterations', type=int, default=1000, help='Max iterations')
    parser.add_argument('--headless', action='store_true', help='Run without display')
    parser.add_argument('--model', type=str, default='generator_model.pth', help='Model path')
    parser.add_argument('--target', type=str, default='target_data.pt', help='Target data path')
    
    args = parser.parse_args()
    
    # Update parameters from command line
    set_parameter('IMG_SIZE', args.size)
    set_parameter('LEARNING_RATE', args.lr)
    set_parameter('MAX_ITERATIONS', args.iterations)
    set_parameter('HEADLESS_MODE', args.headless)
    set_parameter('MODEL_PATH', args.model)
    set_parameter('TARGET_PATH', args.target)
    
    # Run training
    train_model()