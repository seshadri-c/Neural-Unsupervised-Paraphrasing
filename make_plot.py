import os
from matplotlib import pyplot as plt
import numpy as np

# path = "train_msr_valid_loss.txt"

# with open(path,"r") as f:
# 	lines = f.readlines()
# f.close()

# train_loss = []
# validation_loss = []

# for l in lines:
# 	train_loss.append(float(l.split("\t")[1].split()[-1]))
# 	validation_loss.append(float(l.split("\t")[2].split()[-1]))

# plt.plot(train_loss, label = "Loss on Train Set")
# plt.plot(validation_loss, label = "Loss on Test Set\nBest Checkpoint : Checkpoint_{}".format(np.argmin(validation_loss)))
# plt.xlabel("Number of Epochs")
# plt.ylabel("Loss Value")
# plt.legend()
# plt.title("Train Validation Loss \n Micrsoft Research Paraphrase Dataset")
# plt.show()
# plt.savefig("msr_train_loss.png")
# plt.close()

path = "train_paws_valid_loss.txt"

with open(path,"r") as f:
	lines = f.readlines()
f.close()

train_loss = []
validation_loss = []

for l in lines:
	train_loss.append(float(l.split("\t")[1].split()[-1]))
	validation_loss.append(float(l.split("\t")[2].split()[-1]))

plt.plot(train_loss, label = "Loss on Train Set")
plt.plot(validation_loss, label = "Loss on Test Set\nBest Checkpoint : Checkpoint_{}".format(np.argmin(validation_loss)))
plt.xlabel("Number of Epochs")
plt.ylabel("Loss Value")
plt.legend()
plt.title("Train Validation Loss \n Paraphrase Adversaries from Word Scrambling Dataset")
plt.show()
plt.savefig("paws_train_loss.png")
plt.close()
