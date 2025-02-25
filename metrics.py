import torch

# Absolute Relative Depth Error
def RDE(pred, gt):
    mask = gt > 0
    pred, gt = pred[mask], gt[mask]
    
    return torch.mean(torch.abs(pred - gt) / gt)

# Mean IoU and Pixel Accuracy
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

