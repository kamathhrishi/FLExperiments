import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import numpy as np
import os
import gdown
import urllib.error
from typing import Tuple
from collections import OrderedDict


def download_mnist_files():
    """
    Download MNIST files from Google Drive mirror
    """
    # Google Drive file IDs for MNIST dataset
    files = {
        'train-images-idx3-ubyte.gz': '1xPdqzo11y5bq5PiPLyA3dUSYCQg8BG8S',
        'train-labels-idx1-ubyte.gz': '1RS2GxvB8O6dNO4IYZd8YIbg4_E5_hjYZ',
        't10k-images-idx3-ubyte.gz': '1PUw2AP9RCTQHb_8JBbtXMKJ3YKqXbCAP',
        't10k-labels-idx1-ubyte.gz': '1XQ2VT_w8J1NOjK-KQego4GjC0h9F2z7E'
    }
    
    # Create data directory if it doesn't exist
    os.makedirs('./data/MNIST/raw', exist_ok=True)
    
    for filename, file_id in files.items():
        output_path = os.path.join('./data/MNIST/raw', filename)
        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            url = f'https://drive.google.com/uc?id={file_id}'
            gdown.download(url, output_path, quiet=False)
            print(f"Successfully downloaded {filename}")
        else:
            print(f"File {filename} already exists")

def load_datasets(num_clients: int = 10) -> Tuple[list, DataLoader]:
    """
    Load and split MNIST dataset
    """
    # Create data directory
    os.makedirs('./data', exist_ok=True)
    
    # Define transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        # Try to load MNIST dataset
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    except (urllib.error.URLError, urllib.error.HTTPError) as e:
        print(f"Error downloading through torchvision: {e}")
        print("Attempting alternative download method...")
        
        # Download files manually
        download_mnist_files()
        
        # Try loading again with download=False
        train_dataset = datasets.MNIST('./data', train=True, download=False, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=False, transform=transform)
    
    # Split training set into `num_clients` partitions
    partition_size = len(train_dataset) // num_clients
    lengths = [partition_size] * num_clients
    dataset_splits = random_split(train_dataset, lengths, torch.Generator().manual_seed(42))
    
    # Create dataloaders for each partition
    train_loaders = []
    for ds in dataset_splits:
        train_loaders.append(DataLoader(ds, batch_size=32, shuffle=True))
    
    # Create a single test loader
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    return train_loaders, test_loader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def train(model, train_loader, optimizer, device, epochs=1):
    model.train()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
    return loss.item()

def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)
    return test_loss, accuracy

class MnistClient(fl.client.NumPyClient):
    def __init__(self, train_loader, test_loader, client_id):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.client_id = client_id
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        loss = train(self.model, self.train_loader, self.optimizer, self.device)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {"loss": loss}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(self.model, self.test_loader, self.device)
        print(f"\nClient {self.client_id} - Current Accuracy: {accuracy:.4f}")
        
        # Save final model parameters (only client 0)
        if self.client_id == 0 and config.get("server_round", 0) == 5:
            torch.save(self.model.state_dict(), "global_model.pt")
            print("\nGlobal model saved after final round!")
            
            # Print final test accuracy
            print(f"\nFinal Test Accuracy: {accuracy:.4f}")
            
        return float(loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

def client_fn(client_id: int):
    train_loaders, test_loader = load_datasets()
    return MnistClient(train_loaders[client_id], test_loader, client_id)

def start_server():
    # Define evaluation aggregation strategy
    def weighted_average(metrics):
        accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        return {"accuracy": sum(accuracies) / sum(examples)}

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=10,
        min_evaluate_clients=10,
        min_available_clients=10,
        evaluate_metrics_aggregation_fn=weighted_average
    )
    
    # Start server
    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )


# Run these in separate terminals:

# Terminal 1 (Server):
start_server()

# Terminal 2 (Client 0):
#fl.client.start_numpy_client(server_address="0.0.0.0:8081", client=client_fn(0))

# Terminal 3 (Client 1):
#fl.client.start_numpy_client(server_address="0.0.0.0:8081", client=client_fn(1))

# Terminal 4 (Client 2):
#fl.client.start_numpy_client(server_address="0.0.0.0:8081", client=client_fn(2))

#fl.client.start_numpy_client(server_address="0.0.0.0:8081", client=client_fn(3))

#fl.client.start_numpy_client(server_address="0.0.0.0:8081", client=client_fn(4))
