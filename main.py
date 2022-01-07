import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from utils import *
from opt import parse_args
from model import gen_model
import train
import valid

def main(args):
    
    # Check GPU to use CUDA
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")   # use CPU or GPU
    
    model = gen_model(args, device)
    
    # Define optimizer      
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr) 
    
    # Load Data
    train_loader, valid_loader = load_data(args)
    
    # record training process
    epoch_train_losses = []
    epoch_valid_losses = []

    # Evaluate Best loss
    best_loss = 1e5
    
    # start training
    print("\nStart Training")
    for epoch in range(args.epochs):
        # train, valid model
        train_loss, train_mae = train.run(model, device, train_loader, optimizer)
        valid_loss, val_mae = valid.run(model, device, valid_loader)

        best_loss = min(valid_loss, best_loss)
        
        if valid_loss == best_loss:
            os.makedirs(args.save_model_path, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.save_model_path,'best_model.pth'))  # save best model
            
        print('Epoch: {} loss: {:.5f}, mae: {:.5f} / val_loss: {:.5f}, val_mae: {:.5f}'
            .format(epoch+1, train_loss, train_mae, valid_loss, val_mae))
        
        # save results
        epoch_train_losses.append(train_loss)
        epoch_valid_losses.append(valid_loss)

        # save all train valid results
        T = np.array(epoch_train_losses)
        V = np.array(epoch_valid_losses)

        save_loss_path = os.path.join(args.save_model_path, 'loss')
        os.makedirs(save_loss_path, exist_ok=True)
        np.save(os.path.join(save_loss_path,'CRNN_epoch_training_loss.npy'), T)
        np.save(os.path.join(save_loss_path,'CRNN_epoch_valid_loss.npy'), V)

    # Plot Loss
    fig = plt.figure(figsize=(8, 4))
    plt.plot(np.arange(1, args.epochs + 1), T)  # train loss (on epoch end)
    plt.plot(np.arange(1, args.epochs + 1), V)         #  valid loss (on epoch end)
    plt.title("model loss")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend(['train', 'valid'], loc="upper left")
    
    title = os.path.join(save_loss_path,"fig_Loss_RCnet.png")
    plt.savefig(title, dpi=600)


if __name__=='__main__':

    args = parse_args()
    
    print('\nTraining Conditions')
    for arg in vars(args):
        print (arg, ' : ', getattr(args, arg))
        
    main(args)