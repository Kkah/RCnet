import numpy as np
import matplotlib.pyplot as plt


A = np.load('CRNN_epoch_training_loss.npy')
C = np.load('CRNN_epoch_valid_loss.npy')
epochs = len(A)
# plot
fig = plt.figure(figsize=(8, 4))
plt.plot(np.arange(1, epochs + 1), A)  # train loss (on epoch end)
plt.plot(np.arange(1, epochs + 1), C)         #  valid loss (on epoch end)
plt.title("model loss")
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim([0, 0.2])
plt.legend(['train', 'valid'], loc="upper left")
title = "./fig_RC_ResNetCRNN.png"
plt.savefig(title, dpi=600)
# plt.close(fig)
plt.show()