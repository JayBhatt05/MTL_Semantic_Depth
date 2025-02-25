import torch
import torch.nn as nn
import torch.optim as optim
from UNet_model import UNet
from Datasets import CityScapesDataset, ImageTransform, SegTransform, CityDepthTransform, SynScapesDataset
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb


class AutomaticWeightedLoss(nn.Module):
    '''
    Taken from - https://github.com/Mikoto10032/AutomaticWeightedLoss/tree/master
    '''
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction='none')
        
    def forward(self, logits, targets):
        ce_loss = self.ce_loss(logits, targets)  # Compute CE loss first
        pt = torch.exp(-ce_loss)  # Probabilities of true classes
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # Apply Focal Loss formula
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# mIoU and Pixel Accuracy
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

# Absolute Relative Depth Error
def RDE(pred, gt):
    mask = gt > 0
    pred, gt = pred[mask], gt[mask]
    
    return torch.mean(torch.abs(pred - gt) / gt)

def train(model, dataloader, seg_criterion, depth_criterion, optimizer, device):
    model.train()
    print("Training...")
    seg_running_loss = 0.0
    depth_running_loss = 0.0
    combined_running_loss = 0.0
    
    for rgb_images, seg_maps, depth_maps in dataloader:
        rgb_images, seg_maps, depth_maps = rgb_images.to(device), seg_maps.to(device), depth_maps.to(device)
        seg_maps = seg_maps.squeeze(1).long()

        optimizer.zero_grad()
        seg_outputs, depth_outputs = model(rgb_images)
        seg_loss = seg_criterion(seg_outputs, seg_maps)
        depth_loss = depth_criterion(depth_outputs, depth_maps)
        #combined_loss = seg_loss + depth_loss
        combined_loss = awl(seg_loss, depth_loss)
        combined_loss.backward()
        optimizer.step()

        seg_running_loss += seg_loss.item()
        depth_running_loss += depth_loss.item()
        combined_running_loss += combined_loss.item()

    return seg_running_loss / len(dataloader), depth_running_loss / len(dataloader), combined_running_loss / len(dataloader)


def validate(model, dataloader, seg_criterion, depth_criterion, device):
    model.eval()
    print("Validating...")
    seg_running_loss = 0.0
    depth_running_loss = 0.0
    combined_running_loss = 0.0
    total_mIoU = 0.0
    total_pixel_acc = 0.0
    total_rde = 0.0
    
    with torch.no_grad():
        for rgb_images, seg_maps, depth_maps in dataloader:
            rgb_images, seg_maps, depth_maps = rgb_images.to(device), seg_maps.to(device), depth_maps.to(device)
            seg_maps = seg_maps.squeeze(1).long()
            
            seg_outputs, depth_outputs = model(rgb_images)
            seg_loss = seg_criterion(seg_outputs, seg_maps)
            predictions = torch.argmax(seg_outputs, dim=1)
            #seg_maps = seg_maps.squeeze(1).long()
            batch_mIoU, batch_pixel_acc = evaluate(pred=predictions, gt=seg_maps, num_classes=19)
            
            depth_loss = depth_criterion(depth_outputs, depth_maps)
            rde = RDE(depth_outputs, depth_maps)
            
            #combined_loss = seg_loss + depth_loss
            combined_loss = awl(seg_loss, depth_loss)
            
            seg_running_loss += seg_loss.item()
            depth_running_loss += depth_loss.item()
            combined_running_loss += combined_loss.item()
            total_mIoU += batch_mIoU
            total_pixel_acc += batch_pixel_acc
            total_rde += rde.item()

    return (seg_running_loss / len(dataloader), depth_running_loss / len(dataloader), combined_running_loss / len(dataloader),
            total_mIoU / len(dataloader), total_pixel_acc / len(dataloader), total_rde / len(dataloader))


if __name__ == "__main__":
    train_dir = '/home/22ucs095/BTP/CityScapes/train'
    val_dir = '/home/22ucs095/BTP/CityScapes/val'
    
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
    
    train_dataset = CityScapesDataset(train_dir, ImageTransform(), SegTransform(), CityDepthTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = CityScapesDataset(val_dir, ImageTransform(), SegTransform(), CityDepthTransform())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = UNet(num_classes=19, in_channels=3, init_features=64).to(device)
    if torch.cuda.device_count() > 1:
        print(f"\n{torch.cuda.device_count()} GPUs Available!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    awl = AutomaticWeightedLoss(2)
    #seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    seg_criterion = FocalLoss(alpha=0.25, gamma=2.0, ignore_index=255, reduction='mean')
    depth_criterion = nn.L1Loss()
    #optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    optimizer = optim.Adam([
                {'params': model.parameters(), 'weight_decay': 1e-5},
                {'params': awl.parameters(), 'weight_decay': 0}
            ], lr=LEARNING_RATE)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, factor=0.75)
    
    best_epoch = 0
    best_seg_val_loss = float('inf')
    best_depth_val_loss = float('inf')
    best_combined_val_loss = float('inf')
    best_mIoU = 0.0
    best_pixel_acc = 0.0
    best_arde = float('inf')
    
    for epoch in range(EPOCHS):
        seg_train_loss, depth_train_loss, combined_train_loss = train(model, train_loader, seg_criterion, depth_criterion, optimizer,
                                                                      device)
        print(f"Epoch: {epoch+1}/{EPOCHS}, Seg. Train Loss: {seg_train_loss:.4f}, Depth Train Loss: {depth_train_loss:.4f}, \
              Combined Train Loss: {combined_train_loss:.4f}")
        seg_val_loss, depth_val_loss, combined_val_loss, mIoU, pixel_acc, avg_rde = validate(model, val_loader, seg_criterion,
                                                                                        depth_criterion, device)
        
        print(f"Epoch: {epoch+1}/{EPOCHS}, Seg. Val Loss: {seg_val_loss:.4f}, Depth Val Loss: {depth_val_loss:.4f}, \
              Combined Val Loss: {combined_val_loss:.4f}, mIoU: {mIoU:.4f}, pixel_acc: {pixel_acc:.4f}, ARDE: {avg_rde:.4f}")
        wandb.log({"train/combined_loss": combined_train_loss, "val/combined_loss": combined_val_loss})
        wandb.log({"Segmentation/train/seg_loss": seg_train_loss, "Segmentation/val/seg_loss": seg_val_loss, 
                   "Segmentation/val/mIoU": mIoU, "Segmentation/val/pixel_acc": pixel_acc})
        wandb.log({"Depth/train/depth_loss": depth_train_loss, "Depth/val/depth_loss": depth_val_loss, "Depth/val/ARDE": avg_rde})
        
        lr_scheduler.step(combined_val_loss)
        
        if combined_val_loss < best_combined_val_loss:
            best_combined_val_loss = combined_val_loss
            best_seg_val_loss = seg_val_loss
            best_depth_val_loss = depth_val_loss
            best_mIoU = mIoU
            best_pixel_acc = pixel_acc
            best_arde = avg_rde
            best_epoch = epoch + 1
            print("Model Improved...")
            print("Saving Model...")
            torch.save(model.state_dict(), 'CityScapes_UNetMTL_FocalLoss_model.pth')
        
    print(f"Training complete, Best model found at epoch {best_epoch} with Combined Val Loss (CE): {best_combined_val_loss:.4f}, \
          mIoU: {best_mIoU:.4f}, pixel_acc: {best_pixel_acc:.4f}, ARDE: {best_arde:.4f}")
    wandb.finish()