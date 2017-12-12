# 32x32 imagenet util

import numpy as np
import pickle
from PIL import Image
import sys, os

picks = [210, 2] #huskey, german shepard
        # ['75', '205'] # tiger, cheeta
      #[759, 985]:
      #[10, 75,54, 205, 189]:
      #[829]: 
      # [947, 232]:

def make_directory(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)

# Note that this will work with Python3
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo)
    return dict

def load_databatch(d, img_size=32):
    x = d['data']
    y = d['labels']

    x = x/np.float32(255)

    # Labels are indexed from 1, shift it so that indexes start at 0
    y = [i-1 for i in y]
    data_size = x.shape[0]

    # x -= mean_image

    img_size2 = img_size * img_size

    x = np.dstack((x[:, :img_size2], x[:, img_size2:2*img_size2], x[:, 2*img_size2:]))
    x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)

    # create mirrored images
    X_train = x[0:data_size, :, :, :]
    Y_train = y[0:data_size]
    X_train_flip = X_train[:, :, :, ::-1]
    Y_train_flip = Y_train
    X_train = np.concatenate((X_train, X_train_flip), axis=0)
    Y_train = np.concatenate((Y_train, Y_train_flip), axis=0)

    X_train = X_train[np.logical_or(Y_train == picks[0], Y_train == picks[1])]
    Y_train = Y_train[np.logical_or(Y_train == picks[0], Y_train == picks[1])]

    print (len(X_train), len(Y_train))
    
    return dict(
        X_train=(X_train),
        Y_train=Y_train.astype(np.int32))


folder = 'Data_ImageNet/fast_data/IN64_train'
make_directory(folder)
for i in range(1000):
    make_directory(folder + '/' + str(i))

counts = np.zeros(1000, dtype=np.int)

for batch in range(1,10+1):

    d = unpickle("Data_ImageNet/IN64/train_data_batch_" + str(batch))
    b = load_databatch(d, img_size=64)

    for i in range(len(b['X_train'])):
      x = Image.fromarray(np.transpose(b['X_train'][i,:,:,:] * 255, (1,2,0)).astype(np.uint8), 'RGB')
      y = b['Y_train'][i]

      if y in picks:

          x.save(folder + '/' + str(y) + '/' + str(counts[y]).zfill(7) + '.png')

          counts[y] += 1

          print (i)

# i = 320
# x0 = np.transpose(a['X_train'][i,:,:,:], (1,2,0))
# y0 = a['Y_train'][i]
# print (np.shape(x0))
# print (np.max(x0))
# print (y0)

# b = Image.fromarray(x0, 'RGB')
# b.resize((512,512), Image.BILINEAR).show()
