epoches = 50

from keras.callbacks import ModelCheckpoint

from keras.layers import Dense, Input, Dropout, GlobalAveragePooling2D, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model, Sequential
from keras.optimizers import Adam

from keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np # linear algebra
import pandas as pd 

# number of images to feed into the NN for every batch
batch_size = 2

datagen_train = ImageDataGenerator()
datagen_validation = ImageDataGenerator()

base_path = os.getcwd()
pic_size = 300
pic_size2 = 200

test_df = pd.read_csv('/kaggle/input/sign-language-mnist/sign_mnist_test.csv')