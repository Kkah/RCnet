import torch
import torch.nn.functional as F

def run(model, device, train_loader, optimizer):
    # set model as training mode
    model.train()

    train_loss = 0
    train_mae = 0
    
    for image, targets in train_loader:
        image = image.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        
        outputs = model(image)
        
        loss = F.mse_loss(outputs, targets, reduction = 'sum')
        
        train_loss += loss.item()
        
        mae = torch.sum(torch.abs(outputs - targets))
        train_mae += mae.item()
        loss.backward()
        optimizer.step()
        
    train_loss /= len(train_loader.dataset)
    train_mae /= len(train_loader.dataset)
    return train_loss, train_mae