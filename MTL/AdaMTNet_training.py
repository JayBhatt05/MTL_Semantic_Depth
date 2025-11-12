import torch
import torch.nn as nn
import torch.optim as optim
from AdaMTNet_model import AdaMTNet
# from UNet_model import UNet
from Datasets import CityScapesDataset, ImageTransform, SegTransform, CityDepthTransform, SynScapesDataset, SynDepthTransform
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from Losses import FocalLoss
import wandb


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


def train(model, dataloader, seg_criterion, depth_criterion, optimizer, device, epoch):
    model.train()
    print("Training...")
    seg_running_loss = 0.0
    depth_running_loss = 0.0
    combined_running_loss = 0.0
    latest_seg_loss_weight = 0.0
    latest_depth_loss_weight = 0.0

    for batch_idx, (rgb_images, seg_maps, depth_maps) in enumerate(dataloader):
        rgb_images, seg_maps, depth_maps = rgb_images.to(device), seg_maps.to(device), depth_maps.to(device)
        seg_maps = seg_maps.squeeze(1).long()

        # Access underlying model if using DataParallel
        actual_model = model.module if hasattr(model, 'module') else model
        optimizer.zero_grad()

        # ---- Forward pass ----
        seg_outputs, depth_outputs = model(rgb_images)
        seg_loss = seg_criterion(seg_outputs, seg_maps)
        depth_loss = depth_criterion(depth_outputs, depth_maps)

        # ---- Compute gradient magnitudes for each task ----
        # Use only decoder parameters to estimate gradients as in paper
        seg_decoder_params = [p for p in actual_model.seg_decoder_block4.parameters() if p.requires_grad]
        depth_decoder_params = [p for p in actual_model.depth_decoder_block4.parameters() if p.requires_grad]

        seg_grads = torch.autograd.grad(
            seg_loss, seg_decoder_params, retain_graph=True, create_graph=False
        )
        depth_grads = torch.autograd.grad(
            depth_loss, depth_decoder_params, retain_graph=True, create_graph=False
        )

        seg_grad_mag = sum(g.abs().mean() for g in seg_grads)
        depth_grad_mag = sum(g.abs().mean() for g in depth_grads)

        # ---- Normalize to get adaptive weights ----
        total = seg_grad_mag + depth_grad_mag + 1e-8
        seg_weight = (seg_grad_mag / total).detach()
        depth_weight = (depth_grad_mag / total).detach()

        latest_seg_loss_weight = seg_weight.item()
        latest_depth_loss_weight = depth_weight.item()

        # ---- Combined weighted loss ----
        combined_loss = seg_weight * seg_loss + depth_weight * depth_loss

        # ---- Backward + update ----
        combined_loss.backward()
        optimizer.step()

        seg_running_loss += seg_loss.item()
        depth_running_loss += depth_loss.item()
        combined_running_loss += combined_loss.item()

        # if batch_idx % 10 == 0:
        #     print(f"[Batch {batch_idx}/{len(dataloader)}] "
        #           f"SegW: {latest_seg_loss_weight:.4f}, DepthW: {latest_depth_loss_weight:.4f}, "
        #           f"SegL: {seg_loss.item():.4f}, DepthL: {depth_loss.item():.4f}")

    return (seg_running_loss / len(dataloader), depth_running_loss / len(dataloader), combined_running_loss / len(dataloader),
            latest_seg_loss_weight, latest_depth_loss_weight)


def validate(model, dataloader, seg_criterion, depth_criterion, device):
    model.eval()
    print("Validating...")
    seg_running_loss = 0.0
    depth_running_loss = 0.0
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
            
            seg_running_loss += seg_loss.item()
            depth_running_loss += depth_loss.item()
            total_mIoU += batch_mIoU
            total_pixel_acc += batch_pixel_acc
            total_rde += rde.item()

    return (seg_running_loss / len(dataloader), depth_running_loss / len(dataloader), total_mIoU / len(dataloader),
            total_pixel_acc / len(dataloader), total_rde / len(dataloader))
    

if __name__ == "__main__":
    train_dir = '/SynScapes/train'
    val_dir = '/SynScapes/val'
    
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
    "architecture": "AdaMT-Net",
    "dataset": "SynScapes",
    "epochs": EPOCHS,
    "optimizer": "Adam",
    "Height": 256,
    "Width": 512
    }
)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing Device: {device}")
    torch.cuda.empty_cache()
    
    train_dataset = SynScapesDataset(train_dir, ImageTransform(), SegTransform(), SynDepthTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = SynScapesDataset(val_dir, ImageTransform(), SegTransform(), SynDepthTransform())
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    model = AdaMTNet(num_classes=19, in_channels=3)
    if torch.cuda.device_count() > 1:
        print(f"\n{torch.cuda.device_count()} GPUs Available!")
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    
    seg_criterion = nn.CrossEntropyLoss(ignore_index=255)
    depth_criterion = nn.L1Loss()
    
    optimizer = optim.Adam(model.parameters(), weight_decay=1e-5)
    lr_scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True, factor=0.75)
    
    best_epoch = 0
    best_seg_val_loss = float('inf')
    best_depth_val_loss = float('inf')
    best_mIoU = 0.0
    best_pixel_acc = 0.0
    best_arde = float('inf')
    
    for epoch in range(EPOCHS):
        seg_train_loss, depth_train_loss, combined_train_loss, latest_seg_loss_weight, latest_depth_loss_weight = train(
            model, train_loader, seg_criterion, depth_criterion, optimizer, device, epoch)
        print(f"Epoch: {epoch+1}/{EPOCHS}, Seg. Train Loss: {seg_train_loss:.4f}, Depth Train Loss: {depth_train_loss:.4f}, \
              Combined Train Loss: {combined_train_loss:.4f}, Seg. Loss Weight: {latest_seg_loss_weight:.4f}, \
              Depth Loss Weight: {latest_depth_loss_weight:.4f}")
        seg_val_loss, depth_val_loss, mIoU, pixel_acc, avg_rde = validate(model, val_loader, seg_criterion,
                                                                                        depth_criterion, device)
        
        print(f"Epoch: {epoch+1}/{EPOCHS}, Seg. Val Loss: {seg_val_loss:.4f}, Depth Val Loss: {depth_val_loss:.4f}, \
              mIoU: {mIoU:.4f}, pixel_acc: {pixel_acc:.4f}, ARDE: {avg_rde:.4f}")
        wandb.log({"train/combined_loss": combined_train_loss})
        wandb.log({"Segmentation/train/seg_loss": seg_train_loss, "Segmentation/val/seg_loss": seg_val_loss, 
                   "Segmentation/val/mIoU": mIoU, "Segmentation/val/pixel_acc": pixel_acc})
        wandb.log({"Depth/train/depth_loss": depth_train_loss, "Depth/val/depth_loss": depth_val_loss, "Depth/val/ARDE": avg_rde})
        
        lr_scheduler.step(mIoU)
        
        if mIoU > best_mIoU:
            best_seg_val_loss = seg_val_loss
            best_depth_val_loss = depth_val_loss
            best_mIoU = mIoU
            best_pixel_acc = pixel_acc
            best_arde = avg_rde
            best_epoch = epoch + 1
            print("Model Improved...")
            print("Saving Model...")
            torch.save(model.state_dict(), 'SynScapes_AdaMTNet_model_256.pth')
        
    print(f"Training complete, Best model found at epoch {best_epoch} with \
          mIoU: {best_mIoU:.4f}, pixel_acc: {best_pixel_acc:.4f}, ARDE: {best_arde:.4f}")
    wandb.finish()
