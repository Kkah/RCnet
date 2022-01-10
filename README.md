# RCnet
Robotic Colonoscopy Autonomous Driving with supervised learning.   
Predicting X and Y axis steering input using CNN and CNNLSTM

## 0. Requirements
- [Python 3.6](https://www.python.org/)
- [PyTorch 1.8.2](https://pytorch.org/)
- [Numpy 1.19.2](http://www.numpy.org/)
- [opencv 3.4.2](https://opencv.org/)
- [Sklearn 0.24.2](https://scikit-learn.org/stable/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm)

### Install requirements
To install all the requirements type following command in the command line  

    conda env create -f train_env.yaml

Be aware to change the prefix at last line of 'train_env.yaml'

## 1. Dataset

## 2. Training

To start train,

    python main.py --model cnnlstm --data_path <YOUR_DATAPATH_HERE> --save_model_path <SAVE_MODEL_PATH> --batch_size 64 --epochs 20 --lr 1e-4 --seq_len 5 --target_len 1 

Be aware of sequence length and target length while training

### 2.1. Model
Resnet 18 based CNN and CNN-LSTM model

### 2.2. Loss
### 2.2.1. Loss function
Loss function is MSE loss with summation of two axis error

    import torch.nn.functional as F
    loss = F.mse_loss(outputs, targets, reduction = 'sum')

### 2.2.2. Training Result
Your training result will be saved in 'save_model_path'  
Only the best model will be saved as 'best_model.pth'

## 3. Test
To test prediction in testset,

    python plot_prediction.py --model cnnlstm --data_path <YOUR_DATAPATH_HERE> --save_model_path <SAVE_MODEL_PATH> --seq_len 5 --target_len 1

Be aware of sequence length and target length while testing

### 3.1. Prediction

### 3.2. Video
