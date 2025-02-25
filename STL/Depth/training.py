import os
import torch
from UNet_model import UNet
from Datasets import CityScapesDataset, ImageTransform, CityDepthTransform, SynScapesDataset
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Absolute Relative Depth Error
def RDE(pred, gt):
    mask = gt > 0
    pred, gt = pred[mask], gt[mask]
    
    return torch.mean(torch.abs(pred - gt) / gt)


def train(model, dataloader, criterion, optimizer, device):
    model.train()
    print("\nTraining...")
    running_loss = 0.0
    
    for rgb_images, depth_maps in dataloader:
        rgb_images, depth_maps = rgb_images.to(device), depth_maps.to(device)

        optimizer.zero_grad()
        outputs = model(rgb_images)
        loss = criterion(outputs, depth_maps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #print(f"Loss: {loss.item()}")

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    print("Validating...")
    running_loss = 0.0
    total_rde = 0.0
    
    with torch.no_grad():
        for rgb_images, depth_maps in dataloader:
            rgb_images, depth_maps = rgb_images.to(device), depth_maps.to(device)
            outputs = model(rgb_images)
            
            loss = criterion(outputs, depth_maps)
            rde = RDE(outputs, depth_maps)

            running_loss += loss.item()
            total_rde += rde.item()

    return running_loss / len(dataloader), total_rde / len(dataloader)


if __name__ == "__main__":
    train_dir = '/CityScapes/train'
    val_dir = '/CityScapes/val'
    
    BATCH_SIZE = 8
    LEARNING_RATE = 5e-4
    EPOCHS = 200
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing Device: {device}")
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    
    train_dataset = CityScapesDataset(train_dir, ImageTransform(), CityDepthTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = CityScapesDataset(val_dir, ImageTransform(), CityDepthTransform())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = UNet(in_channels=3, init_features=64)
    if torch.cuda.device_count() > 1:
        print(f"\nUsing {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, factor=0.5)
    
    best_epoch = 0
    best_val_loss = float('inf')
    best_rde = float('inf')
    
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, avg_rde = validate(model, val_loader, criterion, device)
        
        print(f"Epoch: {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss (ADE): {val_loss:.4f}, ARDE: {avg_rde:.4f}")
        
        lr_scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_rde = avg_rde
            best_epoch = epoch + 1
            print("Model Improved...")
            print("Saving Model...")
            torch.save(model.state_dict(), 'CityScapes_depth_model.pth')
            
    print(f"Training complete, Best model found at epoch {best_epoch} with Val Loss (ADE): {best_val_loss:.4f} and ARDE: {best_rde:.4f}")
