import torch
import torch.nn as nn
import torch.optim as optim
from UNet_model import UNet
from Datasets import CityScapesDataset, ImageTransform, SegTransform, SynScapesDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb


def evaluate(pred, gt, num_classes=19):
    assert pred.shape == gt.shape
    
    pred = torch.flatten(pred)
    gt = torch.flatten(gt)
    valid_mask = (gt != 255) & (gt != -1) #remaining : 0-18
    pred = pred[valid_mask]
    gt = gt[valid_mask]
    
    mIoU = 0
    for i in range(num_classes):
        #if i in target
        if i not in gt:
            continue
        
        TP = torch.sum((pred==i)&(gt==i))
        FP = torch.sum((pred==i)&(gt!=i))
        FN = torch.sum((pred!=i)&(gt==i))
        mIoU += TP/(TP+FP+FN+1e-6)
    
    class_count_target = len(torch.unique(gt))
    mIoU = mIoU/class_count_target
    
    pixel_acc = torch.sum(pred == gt) / len(gt)
    
    return mIoU.item(), pixel_acc.item()

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    print("Training...")
    running_loss = 0.0
    
    for rgb_images, seg_maps in dataloader:
        rgb_images, seg_maps = rgb_images.to(device), seg_maps.to(device)
        seg_maps = seg_maps.squeeze(1).long()

        optimizer.zero_grad()
        outputs = model(rgb_images)
        loss = criterion(outputs, seg_maps)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    print("Validating...")
    running_loss = 0.0
    total_mIoU = 0.0
    total_pixel_acc = 0.0
    
    with torch.no_grad():
        for rgb_images, seg_maps in dataloader:
            rgb_images, seg_maps = rgb_images.to(device), seg_maps.to(device)
            seg_maps = seg_maps.squeeze(1).long()
            outputs = model(rgb_images)
            loss = criterion(outputs, seg_maps)

            running_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            seg_maps = seg_maps.squeeze(1).long()
            batch_mIoU, batch_pixel_acc = evaluate(pred=predictions, gt=seg_maps, num_classes=19)
            
            total_mIoU += batch_mIoU
            total_pixel_acc += batch_pixel_acc

    return running_loss / len(dataloader), total_mIoU / len(dataloader), total_pixel_acc / len(dataloader)


if __name__ == "__main__":
    train_dir = '/CityScapes/train'
    val_dir = '/CityScapes/val'
    
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-4
    EPOCHS = 200
    
    wandb.init(
    # set the wandb project where this run will be logged
    project="BTP",

    # track hyperparameters and run metadata
    config={
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "architecture": "UNet",
    "dataset": "CityScapes",
    "epochs": EPOCHS,
    "optimizer": "Adam",
    }
)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing Device: {device}")
    #device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    torch.cuda.empty_cache()
    
    train_dataset = CityScapesDataset(train_dir, ImageTransform(), SegTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = CityScapesDataset(val_dir, ImageTransform(), SegTransform())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = UNet(num_classes=19, in_channels=3, init_features=64).to(device)
    if torch.cuda.device_count() > 1:
        print(f"\n{torch.cuda.device_count()} GPUs Available!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, factor=0.75)
    
    best_epoch = 0
    best_val_loss = float('inf')
    best_mIoU = 0.0
    best_pixel_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion, optimizer, device)
        val_loss, mIoU, pixel_acc = validate(model, val_loader, criterion, device)
        
        print(f"Epoch: {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, mIoU: {mIoU:.4f}, pixel_acc: {pixel_acc:.4f}")
        wandb.log({"train/loss": train_loss, "validate/loss": val_loss, "validate/mIoU": mIoU, "validate/pixel_acc": pixel_acc})
        
        lr_scheduler.step(val_loss)
        
        if mIoU > best_mIoU:
            best_mIoU = mIoU
            best_val_loss = val_loss
            best_pixel_acc = pixel_acc
            best_epoch = epoch + 1
            print("Model Improved...")
            print("Saving Model...")
            torch.save(model.state_dict(), 'CityScapes_seg_model.pth')
        
    print(f"Training complete, Best model found at epoch {best_epoch} with Val Loss (CE): {best_val_loss:.4f}, mIoU: {best_mIoU:.4f}, pixel_acc: {best_pixel_acc}")
    wandb.finish()
