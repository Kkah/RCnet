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
Resnet 18 based CNN

### 2.2. Loss
### 2.2.1. Loss function
Loss function is MSE loss with summation of two axis error

    import torch.nn.functional as F
    loss = F.mse_loss(outputs, targets, reduction = 'sum')

### 2.2.2. Result
![fig_Loss_RCnet](https://user-images.githubusercontent.com/86364359/147040679-cd61ab66-b13e-4fea-a90a-29ce281763be.png)

## 3. Test
### 3.1. Prediction
#### exp_20211208-142740
![Prediction_exp_20211208-142740](https://user-images.githubusercontent.com/86364359/147041682-29febaa6-986b-4d77-955b-1041c679fe8b.png)
#### exp_20211208-132354
![Prediction_exp_20211208-132354](https://user-images.githubusercontent.com/86364359/147041760-ab04946e-98e3-4398-8bd9-e99a67777ee5.png)
#### exp_20211208-142936
![Prediction_exp_20211208-142936](https://user-images.githubusercontent.com/86364359/147041765-c2fcb073-e383-47f3-8fe9-c42a0cca595b.png)
#### exp_20211207-214442
![Prediction_exp_20211207-214442](https://user-images.githubusercontent.com/86364359/147041811-53f76f52-e21c-4dbd-a8b5-5c2acb124d5e.png)

### 3.2. Video
