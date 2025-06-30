"""
Example Neural Network Project for Neural Nexus IDE
This demonstrates how multiple scripts can work together
Save each section as a separate file and load them in the IDE
"""

# ===== File 1: data_loader.py =====
"""
Data loader module for MNIST dataset
"""
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class MNISTDataLoader:
    def __init__(self, batch_size=64, data_dir='./data'):
        self.batch_size = batch_size
        self.data_dir = data_dir
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
    def get_train_loader(self):
        """Get training data loader"""
        train_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.transform
        )
        
        return DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2
        )
        
    def get_test_loader(self):
        """Get test data loader"""
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.transform
        )
        
        return DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2
        )

# ===== File 2: model.py =====
"""
Neural network model definition
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleConvNet(nn.Module):
    """Simple convolutional neural network for MNIST"""
    
    def __init__(self, num_classes=10):
        super(SimpleConvNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        
        # Pooling
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # Conv block 1
        x = self.pool(F.relu(self.conv1(x)))
        
        # Conv block 2
        x = self.pool(F.relu(self.conv2(x)))
        
        # Flatten
        x = x.view(-1, 64 * 7 * 7)
        
        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ===== File 3: trainer.py =====
"""
Training script for the neural network
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import our modules
from data_loader import MNISTDataLoader
from model import SimpleConvNet

class Trainer:
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.train_losses = []
        self.val_accuracies = []
        
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(train_loader)
        self.train_losses.append(avg_loss)
        return avg_loss
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
        accuracy = 100 * correct / total
        self.val_accuracies.append(accuracy)
        return accuracy
    
    def plot_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.val_accuracies)
        ax2.set_title('Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()

def main():
    """Main training script"""
    print("Starting Neural Network Training")
    print("-" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Initialize data loader
    print("Loading data...")
    data_loader = MNISTDataLoader(batch_size=64)
    train_loader = data_loader.get_train_loader()
    test_loader = data_loader.get_test_loader()
    
    # Initialize model
    print("Initializing model...")
    model = SimpleConvNet(num_classes=10)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Initialize trainer
    trainer = Trainer(model)
    print(f"Using device: {trainer.device}")
    
    # Training loop
    num_epochs = 10
    print(f"\nTraining for {num_epochs} epochs...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        accuracy = trainer.evaluate(test_loader)
        print(f"Validation Accuracy: {accuracy:.2f}%")
    
    # Plot results
    print("\nPlotting training history...")
    trainer.plot_history()
    
    # Save model
    print("\nSaving model...")
    torch.save(model.state_dict(), 'mnist_model.pth')
    print("Model saved as 'mnist_model.pth'")
    
    print("\nTraining complete!")

if __name__ == "__main__":
    main()

# ===== File 4: inference.py =====
"""
Inference script for making predictions
"""
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Import our model
from model import SimpleConvNet

class Predictor:
    def __init__(self, model_path='mnist_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = SimpleConvNet(num_classes=10)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
    def preprocess_image(self, image_path):
        """Preprocess image for inference"""
        # Load and convert to grayscale
        img = Image.open(image_path).convert('L')
        
        # Resize to 28x28
        img = img.resize((28, 28))
        
        # Convert to numpy array and normalize
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Normalize with MNIST stats
        img_array = (img_array - 0.1307) / 0.3081
        
        # Convert to tensor and add batch dimension
        img_tensor = torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
        
        return img_tensor
    
    def predict(self, image_path):
        """Make prediction on single image"""
        # Preprocess
        img_tensor = self.preprocess_image(image_path).to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(img_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, prediction].item()
            
        return prediction, confidence, probabilities[0].cpu().numpy()
    
    def visualize_prediction(self, image_path):
        """Visualize prediction with confidence scores"""
        prediction, confidence, probs = self.predict(image_path)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        img = Image.open(image_path).convert('L')
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f'Predicted: {prediction} (Confidence: {confidence:.2%})')
        ax1.axis('off')
        
        # Show probability distribution
        classes = list(range(10))
        ax2.bar(classes, probs)
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Probability')
        ax2.set_title('Prediction Probabilities')
        ax2.set_xticks(classes)
        
        plt.tight_layout()
        plt.show()
        
        return prediction

# Example usage
if __name__ == "__main__":
    # Create a test image (you would load a real image here)
    print("Creating test image...")
    test_img = Image.new('L', (28, 28), color=255)
    # Draw a simple digit (this is just an example)
    from PIL import ImageDraw, ImageFont
    draw = ImageDraw.Draw(test_img)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((10, 5), "7", fill=0, font=font)
    test_img.save('test_digit.png')
    
    # Make prediction
    print("Loading model and making prediction...")
    predictor = Predictor()
    
    try:
        prediction = predictor.visualize_prediction('test_digit.png')
        print(f"Predicted digit: {prediction}")
    except FileNotFoundError:
        print("Model file not found. Please run trainer.py first to train the model.")
