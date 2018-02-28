from future import absolute_import
from future import print_function
import os
os.environ['KERAS_BACKEND'] = 'theano'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'
#os.environ['THEANO_FLAGS']='mode=FAST_RUN,device=gpu0,floatX=float32,optimizer=fast_compile'
os.environ[CUDA_VISIBLE_DEVICES] = ""
import sys
import pylab as pl
import matplotlib.cm as cm
import itertools
import numpy as np
import theano.tensor as T
np.random.seed(1337) # for reproducibility

from keras.datasets import mnist
from keras.layers.noise import GaussianNoise
import keras.models as models
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Permute
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
#from keras.regularizers import ActivityRegularizer
#from keras.utils.visualize_util import plot

from keras import backend as K
K.set_image_dim_ordering('th')

import cv2
import numpy as np
from tqdm import tqdm

class UnPooling2D(Layer):
A 2D Repeat layer
def init(self, poolsize=(2, 2)):
super(UnPooling2D, self).init()
self.poolsize = poolsize

@property
def output_shape(self):
    input_shape = self.input_shape
    return (input_shape[0], input_shape[1],
            self.poolsize[0] * input_shape[2],
            self.poolsize[1] * input_shape[3])

def get_output(self, train):
    X = self.get_input(train)
    s1 = self.poolsize[0]
    s2 = self.poolsize[1]
    output = X.repeat(s1, axis=2).repeat(s2, axis=3)
    return output

def get_config(self):
    return {"name":self.__class__.__name__,
        "poolsize":self.poolsize}
def create_encoding_layers():
kernel = 3
filter_size = 64
pad = 1
pool_size = 2
return [
ZeroPadding2D(padding=(pad,pad)),
Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
BatchNormalization(),
Activation('relu'),
MaxPooling2D(pool_size=(pool_size, pool_size)),

    ZeroPadding2D(padding=(pad,pad)),
    Convolution2D(128, kernel, kernel, border_mode='valid'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    ZeroPadding2D(padding=(pad,pad)),
    Convolution2D(256, kernel, kernel, border_mode='valid'),
    BatchNormalization(),
    Activation('relu'),
    MaxPooling2D(pool_size=(pool_size, pool_size)),

    ZeroPadding2D(padding=(pad,pad)),
    Convolution2D(512, kernel, kernel, border_mode='valid'),
    BatchNormalization(),
    Activation('relu'),
    #MaxPooling2D(pool_size=(pool_size, pool_size)),
]
def create_decoding_layers():
kernel = 3
filter_size = 64
pad = 1
pool_size = 2
return[
UnPooling2D(),
#UpSampling2D (size=(pool_size,pool_size)),
ZeroPadding2D(padding=(pad,pad)),
Convolution2D(512, kernel, kernel, border_mode='valid'),
BatchNormalization(),

    UpSampling2D(size=(pool_size,pool_size)),
    ZeroPadding2D(padding=(pad,pad)),
    Convolution2D(256, kernel, kernel, border_mode='valid'),
    BatchNormalization(),

    UpSampling2D(size=(pool_size,pool_size)),
    ZeroPadding2D(padding=(pad,pad)),
    Convolution2D(128, kernel, kernel, border_mode='valid'),
    BatchNormalization(),

    UpSampling2D(size=(pool_size,pool_size)),
    ZeroPadding2D(padding=(pad,pad)),
    Convolution2D(filter_size, kernel, kernel, border_mode='valid'),
    BatchNormalization(),
]


def getModel():
#can load model from uploaded folder
autoencoder = models.Sequential()
#Add a noise layer to get a denoising autoencoder. This helps avoid overfitting
autoencoder.add(Layer(input_shape=(3, 360, 480)))
autoencoder.encoding_layers = create_encoding_layers()
autoencoder.decoding_layers = create_decoding_layers()
data_shape = 360*480
for l in autoencoder.encoding_layers:
autoencoder.add(l)
for l in autoencoder.decoding_layers:
autoencoder.add(l)
autoencoder.add(Convolution2D(12, 1, 1, border_mode='valid'))
autoencoder.add(Reshape((12,360*480),input_shape=(12,360,480) ))
autoencoder.add(Permute((2, 1)))
autoencoder.add(Activation('softmax'))
autoencoder.compile(loss=categorical_crossentropy, optimizer='adadelta')
autoencoder.load_weights('model_weight_ep100.hdf5')
loaded_model = autoencoder

model=loaded_model # this model will be a trained classifier model. e.g. LinearSVM
return model
def visualize(temp, plot=True):
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
label_colours = np.array([Sky, Building, Pole, Road, Pavement,
Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
r = temp.copy()
g = temp.copy()
b = temp.copy()
for l in range(0,11):
r[temp==l]=label_colours[l,0]
g[temp==l]=label_colours[l,1]
b[temp==l]=label_colours[l,2]

rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
rgb[:,:,0] = (r)#[:,:,0]
rgb[:,:,1] = (g)#[:,:,1]
rgb[:,:,2] = (b)#[:,:,2]
if plot:
    plt.imshow(rgb)
else:
    return rgb.astype(int)
def normalized(rgb):
#return rgb/255.0
norm=np.zeros((rgb.shape[0], rgb.shape[1], 3),np.float32)

b=rgb[:,:,0]
g=rgb[:,:,1]
r=rgb[:,:,2]

norm[:,:,0]=cv2.equalizeHist(b)
norm[:,:,1]=cv2.equalizeHist(g)
norm[:,:,2]=cv2.equalizeHist(r)

return norm
from PIL import Image
import base64,cStringIO
def runModel(data,model):
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road_marking = [255,69,0]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]
label_colours = np.array([Sky, Building, Pole, Road, Pavement,
Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
ogOut = sys.stdout
sys.stdout = sys.stderr
lst = data['files']
for item in lst:
temp_data = []
im = cv2.imread(item[0]['file-uri'])
im = cv2.resize(im, (480, 360)) 
image = [np.rollaxis(normalized(im),2)]
output = model.predict_proba(np.array(image))
pred = visualize(np.argmax(output[0],axis=1).reshape((360,480)), False)
print(pred)
#im = Image.fromarray(np.array(pred).astype('uint8'))
#buf = cStringIO.StringIO()
#im.save(buf,format='jpeg')
#im_str = base64.b64encode(buf.getvalue())
label = {}
#label ['type'] = 'classification'
#label ['value'] = '<img style="width: 100%" src="data:image/png;base64, ' + im_str +'"/>'
return [label]

data = {'files': [], 'data':''}
dataitem = {'file-uri':'scene1.png'}
data['files'].append([dataitem])
dataitem = {'file-uri' :'scene2.png'}
data['files'].append([dataitem])

model = getModel()
pred = runModel(data,model)
print(pred)