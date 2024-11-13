import torch
from model_definition import FashionMNISTNet  # Import the model architecture
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Load the trained model weights
def load_model_weights(path='model_weights.pt'):
    model = FashionMNISTNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

# Load test data
def load_test_data(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.FashionMNIST(root='./data', train=False, transform=transform, download=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')

if __name__ == "__main__":
    # Load model weights
    model = load_model_weights('model_weights.pt')
    
    test_loader = load_test_data(batch_size=32)
    
    evaluate_model(model, test_loader)
