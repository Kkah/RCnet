import torch
import torch.nn.functional as F

def run(model, device, valid_loader):
    # set model as validing mode
    model.eval()

    valid_loss = 0
    valid_mae = 0
    
    all_y = []
    all_y_pred = []

    with torch.no_grad():
        for image, targets in valid_loader:
            image = image.to(device)
            targets = targets.to(device)
            outputs = model(image)
            
            loss = F.mse_loss(outputs, targets,reduction= 'sum')
            
            valid_loss += loss.item()    
            
            mae = torch.sum(torch.abs(outputs - targets))
            
            valid_mae += mae.item()
        
            y_pred = outputs

            # collect all y and y_pred in all batches                                                                                                                       
            all_y.extend(targets)
            all_y_pred.extend(y_pred)

    valid_loss /= len(valid_loader.dataset)
    valid_mae /= len(valid_loader.dataset)
    
    # torch.save(model.state_dict(), os.path.join(save_model_path, \
    #     'cnn_encoder_epoch{}.pth'.format(epoch + 1)))  # save spatial_encoder
    # torch.save(optimizer.state_dict(), os.path.join(save_model_path, \
    #     'optimizer_epoch{}.pth'.format(epoch + 1)))      # save optimizer

    return valid_loss, valid_mae