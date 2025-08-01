import torch
import torch.nn as nn
import numpy as np
import json
import os

class PixelNet(nn.Module):
    """Simple neural network for learning pixel patterns"""
    def __init__(self):
        super(PixelNet, self).__init__()
        self.fc1 = nn.Linear(10, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def prepare_sequences(pixels, seq_len=10):
    """Prepare pixel sequences for training"""
    sequences = []
    targets = []
    
    for i in range(len(pixels) - seq_len):
        seq = pixels[i:i+seq_len]
        target = pixels[i+seq_len]
        sequences.append(seq)
        targets.append(target)
    
    return np.array(sequences), np.array(targets)

def train_neural_model():
    try:
        # Load pixel data
        if not os.path.exists('pixel_data.json'):
            print("[NEURAL] No pixel data found")
            return
            
        with open('pixel_data.json', 'r') as f:
            data = json.load(f)
            pixels = np.array(data['pixels']) / 255.0  # Normalize
        
        print(f"[NEURAL] Loaded {len(pixels)} pixels")
        
        # Prepare training data
        sequences, targets = prepare_sequences(pixels)
        
        if len(sequences) < 100:
            print("[NEURAL] Not enough data for training")
            return
        
        # Convert to tensors
        X = torch.FloatTensor(sequences)
        y = torch.FloatTensor(targets).reshape(-1, 1)
        
        # Create model
        model = PixelNet()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        print("[NEURAL] Training neural network...")
        epochs = 50
        batch_size = 32
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Mini-batch training
            for i in range(0, len(X), batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                avg_loss = total_loss / (len(X) / batch_size)
                print(f"[NEURAL] Epoch {epoch}, Loss: {avg_loss:.4f}")
        
        # Save model
        torch.save(model.state_dict(), 'pixel_model.pth')
        print("[NEURAL] Model saved to pixel_model.pth")
        
        # Generate predictions
        model.eval()
        with torch.no_grad():
            # Use last sequence to predict next values
            last_seq = torch.FloatTensor(pixels[-10:]).unsqueeze(0)
            predictions = []
            
            for _ in range(100):
                pred = model(last_seq)
                predictions.append(float(pred.item()))
                
                # Shift sequence and add prediction
                last_seq = torch.cat([last_seq[:, 1:], pred.unsqueeze(0)], dim=1)
        
        # Save predictions
        neural_results = {
            'training_loss': float(total_loss / (len(X) / batch_size)),
            'sequences_trained': len(sequences),
            'predictions': [int(p * 255) for p in predictions],  # Denormalize
            'model_architecture': {
                'input_size': 10,
                'hidden_layers': [32, 16],
                'output_size': 1
            }
        }
        
        with open('neural_results.json', 'w') as f:
            json.dump(neural_results, f)
        
        print(f"[NEURAL] Generated {len(predictions)} predictions")
        print(f"[NEURAL] First 10 predictions: {neural_results['predictions'][:10]}")
        
        # Analyze learned patterns
        analyze_neural_patterns(model, sequences[:100])
        
    except Exception as e:
        print(f"[NEURAL] Error: {e}")

def analyze_neural_patterns(model, sample_sequences):
    """Analyze what patterns the network learned"""
    model.eval()
    
    patterns = {
        'increasing': [],
        'decreasing': [],
        'stable': [],
        'volatile': []
    }
    
    with torch.no_grad():
        for seq in sample_sequences:
            seq_tensor = torch.FloatTensor(seq).unsqueeze(0)
            pred = model(seq_tensor).item()
            
            # Classify pattern
            seq_mean = np.mean(seq)
            seq_std = np.std(seq)
            
            if seq[-1] > seq[0] + 0.1:
                patterns['increasing'].append(pred)
            elif seq[-1] < seq[0] - 0.1:
                patterns['decreasing'].append(pred)
            elif seq_std < 0.05:
                patterns['stable'].append(pred)
            else:
                patterns['volatile'].append(pred)
    
    # Save pattern analysis
    pattern_stats = {
        pattern: {
            'count': len(preds),
            'mean_prediction': float(np.mean(preds)) if preds else 0,
            'std_prediction': float(np.std(preds)) if preds else 0
        }
        for pattern, preds in patterns.items()
    }
    
    with open('neural_patterns.json', 'w') as f:
        json.dump(pattern_stats, f)
    
    print("[NEURAL] Pattern analysis saved")

def load_and_predict():
    """Load saved model and make predictions"""
    try:
        if not os.path.exists('pixel_model.pth'):
            print("[NEURAL] No saved model found")
            return
        
        model = PixelNet()
        model.load_state_dict(torch.load('pixel_model.pth'))
        model.eval()
        
        # Load recent correlations if available
        if os.path.exists('correlations.json'):
            with open('correlations.json', 'r') as f:
                correlations = json.load(f)
                
            if len(correlations) >= 10:
                # Use last 10 correlation values
                recent_values = [c['value'] / 255.0 for c in correlations[-10:]]
                input_seq = torch.FloatTensor(recent_values).unsqueeze(0)
                
                with torch.no_grad():
                    prediction = model(input_seq).item() * 255
                
                print(f"[NEURAL] Predicted next value: {int(prediction)}")
                
                # Save prediction
                with open('neural_prediction.json', 'w') as f:
                    json.dump({
                        'predicted_value': int(prediction),
                        'confidence': 0.8,  # Placeholder
                        'based_on': recent_values
                    }, f)
                    
    except Exception as e:
        print(f"[NEURAL] Prediction error: {e}")

if __name__ == "__main__":
    # Train if we have data
    if os.path.exists('pixel_data.json'):
        train_neural_model()
    
    # Make predictions if model exists
    load_and_predict()