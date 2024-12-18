import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchsummary import torchsummary

# Define the CNN architecture
class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # First block with residual connection
        self.conv1 = nn.Conv2d(1, 12, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        
        # Second block
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        
        # Third block
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(12)
        
        # Cross connection with channel attention
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(12, 6, 1),
            nn.ReLU(),
            nn.Conv2d(6, 12, 1),
            nn.Sigmoid()
        )
        
        # Channel expansion
        self.expand = nn.Conv2d(12, 16, kernel_size=1)
        
        # Fourth block
        self.conv4 = nn.Conv2d(16, 16, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(16)
        
        # Fifth block with SE
        self.conv5 = nn.Conv2d(16, 16, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(16)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 8, 1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 1),
            nn.Sigmoid()
        )
        
        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Classification head with dropout
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(16, 10)

    def forward(self, x):
        # First block with residual
        identity = F.pad(x, (0,0,0,0,0,11))  # Pad channels to match
        x = F.relu(self.bn1(self.conv1(x)))
        x = x + identity
        x = F.max_pool2d(x, 2)
        
        # Second block
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Third block
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Channel attention
        att = self.ca(x)
        x = x * att
        
        # Channel expansion
        x = self.expand(x)
        
        # Fourth block
        x = F.relu(self.bn4(self.conv4(x)))
        
        # Fifth block with SE
        x = self.conv5(x)
        x = self.bn5(x)
        se_weight = self.se(x)
        x = F.relu(x * se_weight)
        
        # Global pooling and classification
        x = self.gap(x)
        x = x.view(-1, 16)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

# Training function
def train(model, device, train_loader, optimizer, scheduler, epoch):
    model.train()
    final_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        final_loss = loss.item()
    return final_loss

# Testing function
def test(model, device, test_loader):
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
    accuracy = 100. * correct / len(test_loader.dataset)
    return test_loss, accuracy

def main():
    # Training settings
    batch_size = 128  # Increased batch size for faster convergence
    epochs = 15  # Reduced epochs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Optimized data augmentation - reduced intensity for faster convergence
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=(-15, 15), translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomPerspective(distortion_scale=0.15, p=0.4),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])

    # Test transform without augmentation
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000)

    # Initialize model and optimizer
    model = LightMNIST().to(device)
    
    # Add model summary
    print("\nModel Summary:")
    torchsummary.summary(model, (1, 28, 28))
    
    # Print total parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nTotal parameters: {total_params}')
    
    # Modified optimizer settings for faster convergence
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.03)  # Increased base lr
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,  # Increased max lr
        epochs=epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.2,  # Reduced warmup time
        anneal_strategy='cos',
        div_factor=20,
        final_div_factor=500
    )

    # Add learning rate warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.3,
        end_factor=1.0,
        total_iters=len(train_loader)  # One epoch warmup
    )

    # Training loop
    best_accuracy = 0
    print("\nEpoch  Train Loss    Test Loss    Accuracy    Best")
    print("-" * 50)
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, device, train_loader, optimizer, scheduler, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        
        is_best = accuracy > best_accuracy
        if is_best:
            best_accuracy = accuracy
            torch.save(model.state_dict(), 'mnist_model.pth')
        
        print(f"{epoch:3d}    {train_loss:.6f}    {test_loss:.6f}    {accuracy:.2f}%    {'*' if is_best else ''}")

    print("-" * 50)
    print(f"Final best accuracy: {best_accuracy:.2f}%")

if __name__ == '__main__':
    main() 