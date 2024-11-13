import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# Define the model architecture (must match the one used during training)
class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 7 * 7 * 64)  # Flatten the output
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and load the weights
 # Set to evaluation mode

# Model is ready for evaluation or inference


def load_model(filepath):
    model = torch.load(filepath)
    model.eval()  # Setting the model to evaluation mode to ignore dropout and batch normalization layers
    print(f"Model loaded from {filepath}")
    return model

def evaluate_model(model, test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def create_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return test_loader


model = FashionMNISTNet()
model.load_state_dict(torch.load('fashion_mnist_model_weights.pth'))
model.eval() 
# Now to load the model and evaluate
if __name__ == '__main__':
    test_loader = create_data_loaders(batch_size=64)
    
    model = load_model('model.pt')
    
    accuracy = evaluate_model(model, test_loader)
    print(f'Test Accuracy: {accuracy:.2f}%')
