import numpy as np


load_conv = np.load("conv_data/train_4a_01728.npy")

print(load_conv.shape)
print(load_conv[:10])