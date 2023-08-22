# %%
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import wandb


# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download the entire MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Define the size of the validation set
validation_size = 10000

# Split the trainset into training and validation sets
trainset, validationset = torch.utils.data.random_split(trainset, [len(trainset) - validation_size, validation_size])

# Create DataLoader objects for training, validation, and test sets
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
validationloader = torch.utils.data.DataLoader(validationset, batch_size=64, shuffle=False)

# Download the test set
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# %%
#Define a simple neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Instantiate the model

# %%
def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


#for lr_rate in config["lr"] :
def main():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument(
        "--lr", type=float, default=0.1, help="Learning rate."
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Epochs to train."
    )



    args = parser.parse_args()
    config_dict={"epochs": args.epochs,
        "lr": args.lr}
    config = dict2namespace(config_dict)

    print(config)
    wandb.init(
        project="shu7",
        config = config_dict,
        name=f"lr = {args.lr}"
        )

    # every time you make a new trial you need to reset the model AND THE OPTIMIZER (SGD)
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=config.lr, momentum=0.9)
    for epoch in range(config.epochs):
        # training:
        net.train() # tells model we are training.
        for i, data in enumerate(trainloader, 0):
            
            inputs, labels = data
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss})
        # validation
        net.eval() # tells model we are testing.
        correct = 0
        total = 0
        with torch.no_grad():
            for data in validationloader:
                inputs, labels = data
                outputs = net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            validation_acc = 100 * correct/total
            wandb.log({'validation accuracy': validation_acc})
    wandb.finish()

# %%
if __name__ == "__main__":
    main()
