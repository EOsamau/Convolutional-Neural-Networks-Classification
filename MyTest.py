import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import numpy as np

class FashionMNISTDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data.numpy()
        self.targets = targets
        self.transform = transform
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image = self.data[idx].astype(np.uint8)
        label = int(self.targets[idx])
        
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.FloatTensor(image) / 255.0
            image = image.unsqueeze(0)
            
        return image, label

class FashionMNISTNet(nn.Module):
    def __init__(self):
        super(FashionMNISTNet, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)
        )
        
        self.flatten = nn.Flatten()
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x

def load_and_evaluate_model(model_path='model.pt', batch_size=64):
    # Create a new instance of the model
    model = FashionMNISTNet()
    
    # Load the saved model state
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    
    print(f"Loaded model from epoch {epoch}")
    
    # Prepare the test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=None
    )
    
    custom_test_dataset = FashionMNISTDataset(
        test_dataset.data,
        test_dataset.targets,
        transform=transform
    )
    
    test_loader = DataLoader(
        custom_test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Evaluate the model
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    print(f'Model Accuracy: {accuracy:.2f}%')
    return accuracy

if __name__ == "__main__":
    load_and_evaluate_model()