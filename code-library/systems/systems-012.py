import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import json
import os

class GeneratorNet(nn.Module):
    """Neural network for generating pixel patterns"""
    def __init__(self, feature_dim=10):
        super(GeneratorNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 256)
        self.fc4 = nn.Linear(256, 100)  # Generate 100 pixels at a time
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))  # Output between 0 and 1
        return x

def generate_neural_image():
    """Generate images using neural network and learned patterns"""
    
    try:
        # Initialize generator
        generator = GeneratorNet()
        
        # Load multiple data sources
        features = extract_features()
        
        if not features:
            print("[NEURAL_GEN] No features found, using random initialization")
            features = np.random.rand(10)
        
        # Convert to tensor
        feature_tensor = torch.FloatTensor(features).unsqueeze(0)
        
        # Generate pixel blocks
        print("[NEURAL_GEN] Generating image blocks...")
        
        generated_pixels = []
        
        # Generate multiple blocks with variations
        for i in range(100):  # 100 blocks of 100 pixels = 10,000 pixels (100x100 image)
            # Add noise for variation
            noise = torch.randn(1, 10) * 0.1
            noisy_features = feature_tensor + noise
            
            # Generate block
            with torch.no_grad():
                pixel_block = generator(noisy_features)
                pixels = (pixel_block.squeeze().numpy() * 255).astype(np.uint8)
                generated_pixels.extend(pixels)
        
        # Create image
        size = (100, 100)
        img = Image.new('L', size)
        img.putdata(generated_pixels[:size[0]*size[1]])
        
        filename = f'neural_generated_{int(np.random.rand()*10000)}.jpg'
        img.save(filename)
        
        print(f"[NEURAL_GEN] Generated {filename}")
        
        # Save generation info
        generation_info = {
            'filename': filename,
            'method': 'neural_network',
            'features_used': features.tolist() if isinstance(features, np.ndarray) else features,
            'statistics': {
                'mean': float(np.mean(generated_pixels)),
                'std': float(np.std(generated_pixels)),
                'min': int(np.min(generated_pixels)),
                'max': int(np.max(generated_pixels))
            }
        }
        
        with open('neural_generation.json', 'w') as f:
            json.dump(generation_info, f)
        
        # Train generator if we have target data
        train_generator(generator, features)
        
    except Exception as e:
        print(f"[NEURAL_GEN] Error: {e}")

def extract_features():
    """Extract features from various analysis results"""
    features = []
    
    # 1. From neural results
    if os.path.exists('neural_results.json'):
        with open('neural_results.json', 'r') as f:
            neural = json.load(f)
            
        if 'predictions' in neural:
            # Use statistics of predictions
            preds = neural['predictions'][:100]
            features.extend([
                np.mean(preds) / 255,
                np.std(preds) / 128,
                np.min(preds) / 255,
                np.max(preds) / 255
            ])
    
    # 2. From vision results
    if os.path.exists('vision_results.json'):
        with open('vision_results.json', 'r') as f:
            vision = json.load(f)
            
        if 'edges' in vision:
            features.append(vision['edges']['canny_edge_ratio'])
        
        if 'texture' in vision:
            features.append(vision['texture']['lbp_entropy'] / 10)  # Normalize
    
    # 3. From pattern analysis
    if os.path.exists('patterns.json'):
        with open('patterns.json', 'r') as f:
            patterns = json.load(f)
            
        if 'statistics' in patterns and patterns['statistics']:
            stats = patterns['statistics'][0]
            features.append(stats.get('mean', 128) / 255)
            features.append(stats.get('std', 50) / 128)
    
    # 4. From hybrid analysis
    if os.path.exists('hybrid_analysis.json'):
        with open('hybrid_analysis.json', 'r') as f:
            hybrid = json.load(f)
            
        if 'synthesis' in hybrid:
            quality_score = hybrid['synthesis'].get('quality_score', 50)
            features.append(quality_score / 100)
    
    # Pad or trim to exactly 10 features
    if len(features) < 10:
        features.extend([0.5] * (10 - len(features)))
    else:
        features = features[:10]
    
    print(f"[NEURAL_GEN] Extracted {len(features)} features")
    return np.array(features)

def train_generator(generator, features):
    """Train generator to match learned patterns"""
    
    try:
        # Load target pixel data
        if not os.path.exists('pixel_data.json'):
            print("[NEURAL_GEN] No training data available")
            return
        
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            target_pixels = np.array(data['pixels']) / 255.0
        
        # Prepare training data
        num_samples = len(target_pixels) // 100
        if num_samples < 10:
            print("[NEURAL_GEN] Insufficient training data")
            return
        
        print(f"[NEURAL_GEN] Training generator with {num_samples} samples...")
        
        # Training setup
        optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # Simple training loop
        for epoch in range(20):
            total_loss = 0
            
            for i in range(min(num_samples, 50)):
                # Get target block
                start_idx = i * 100
                target_block = target_pixels[start_idx:start_idx + 100]
                
                if len(target_block) < 100:
                    continue
                
                # Add noise to features for variety
                noise = np.random.randn(10) * 0.1
                noisy_features = features + noise
                
                # Forward pass
                input_tensor = torch.FloatTensor(noisy_features).unsqueeze(0)
                target_tensor = torch.FloatTensor(target_block).unsqueeze(0)
                
                output = generator(input_tensor)
                loss = criterion(output, target_tensor)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 5 == 0:
                avg_loss = total_loss / min(num_samples, 50)
                print(f"[NEURAL_GEN] Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Save trained model
        torch.save(generator.state_dict(), 'generator_model.pth')
        print("[NEURAL_GEN] Saved trained generator")
        
        # Generate sample with trained model
        generate_trained_sample(generator, features)
        
    except Exception as e:
        print(f"[NEURAL_GEN] Training error: {e}")

def generate_trained_sample(generator, base_features):
    """Generate image with trained generator"""
    
    generator.eval()
    
    # Generate coherent image
    size = (100, 100)
    generated_pixels = []
    
    # Use sliding features for coherence
    for i in range(100):
        # Modify features gradually
        progress = i / 100
        modified_features = base_features * (1 - progress * 0.5)
        modified_features += np.random.randn(10) * 0.05
        
        input_tensor = torch.FloatTensor(modified_features).unsqueeze(0)
        
        with torch.no_grad():
            pixel_block = generator(input_tensor)
            pixels = (pixel_block.squeeze().numpy() * 255).astype(np.uint8)
            generated_pixels.extend(pixels)
    
    # Create and save image
    img = Image.new('L', size)
    img.putdata(generated_pixels[:size[0]*size[1]])
    
    filename = 'neural_trained_generated.jpg'
    img.save(filename)
    
    print(f"[NEURAL_GEN] Generated trained sample: {filename}")

def generate_style_transfer():
    """Generate image that transfers learned style"""
    
    try:
        # Load source features from multiple images
        all_features = []
        
        # Process batch results if available
        for file in os.listdir('.'):
            if file.startswith('batch_') and file.endswith('.json'):
                with open(file, 'r') as f:
                    batch_data = json.load(f)
                    
                if 'stats' in batch_data:
                    stats = batch_data['stats']
                    features = [
                        stats.get('mean', 128) / 255,
                        stats.get('std', 50) / 128,
                        stats.get('unique_values', 100) / 256
                    ]
                    all_features.append(features)
        
        if not all_features:
            print("[NEURAL_GEN] No style sources found")
            return
        
        # Average features for style
        avg_features = np.mean(all_features, axis=0)
        
        # Pad to 10 features
        style_features = np.zeros(10)
        style_features[:len(avg_features)] = avg_features
        
        # Generate with style
        generator = GeneratorNet()
        
        size = (100, 100)
        styled_pixels = []
        
        for i in range(100):
            # Interpolate between styles
            t = i / 100
            interpolated = style_features * t + np.random.rand(10) * (1 - t) * 0.2
            
            input_tensor = torch.FloatTensor(interpolated).unsqueeze(0)
            
            with torch.no_grad():
                pixel_block = generator(input_tensor)
                pixels = (pixel_block.squeeze().numpy() * 255).astype(np.uint8)
                styled_pixels.extend(pixels)
        
        # Create styled image
        img = Image.new('L', size)
        img.putdata(styled_pixels[:size[0]*size[1]])
        
        filename = 'neural_styled.jpg'
        img.save(filename)
        
        print(f"[NEURAL_GEN] Generated styled image: {filename}")
        
    except Exception as e:
        print(f"[NEURAL_GEN] Style transfer error: {e}")

if __name__ == "__main__":
    # Generate basic neural image
    generate_neural_image()
    
    # Try style transfer if we have multiple sources
    generate_style_transfer()