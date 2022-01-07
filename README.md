# RCnet
Robotic Colonoscopy Autonomous Driving with supervised learning.   
Predicting X and Y axis steering input using CNN.

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

    cd RCnet
    pip install -r requirements.txt

## 1. Dataset

## 2. Training
### 2.1. Model
Resnet 18 based CNN and CNN-LSTM model

### 2.2. Loss
### 2.2.1. Loss function
Loss function is MSE loss with summation of two axis error

    import torch.nn.functional as F
    loss = F.mse_loss(outputs, targets, reduction = 'sum')

### 2.2.2. Result

## 3. Test
### 3.1. Prediction

### 3.2. Video
