#!/usr/bin/env python3
# import packages
import os
from conf import myConfig as config
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D,Activation,Input,Add,Subtract,AveragePooling2D,Multiply,Concatenate,Conv2DTranspose,Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow.keras.backend as K
import numpy as np
from numpy import array
from numpy.linalg import norm
import tensorflow as tf
from numpy import *
import random
from skimage.util import random_noise
#os.environ["CUDA_VISIBLE_DEVICES"]="7"
#tf_device='/gpu:7'

# define the power law transformation function
def power_law_transform(x, gamma=1.0):
    c = 255 / (tf.reduce_max(x) ** gamma)
    return c * (x ** gamma)

# create CNN model
input_img=Input(shape=(None,None,1))

#POWER LAW TRANSFORMATION
pwl = Lambda(power_law_transform, arguments={'gamma': 1.0})(input_img)

pwl_rc = Activation('relu')(pwl)
pwl_rc = Conv2D(64,(3,3), dilation_rate=1, padding="same")(pwl_rc)

#1st relu on input image
x=Activation('relu')(input_img)

x=Conv2D(64,(3,3), dilation_rate=1,padding="same")(x)

x1 = Add()([x,pwl_rc])

#After X1 1st RC
x=Activation('relu')(x1)
x=Conv2D(64,(3,3), dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)


#FEATURE DENOISING BLOCK
x=Conv2D(64,(3,3), padding="same")(x)
x_temp2=Activation('relu')(x)
x=AveragePooling2D(pool_size=(1, 1))(x_temp2)
x=Conv2D(64,(1,1), padding="same")(x)
x_temp3 = Add()([x, x_temp2])

#Dilated 3-2-1
x=Activation('relu')(x_temp3)
x=Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)

x=Activation('relu')(x)
x=Conv2D(64,(3,3), dilation_rate=2,padding="same")(x)

x=Activation('relu')(x)
x2=Conv2D(64,(3,3), dilation_rate=1,padding="same")(x)

x = Add()([x2, x1])

x = Concatenate()([x,x1])

x3 = Conv2D(64,(3,3), padding="same")(x)

# PAB - Pixel Attention Block
x=Activation('relu')(x3)
x = Conv2D(64,(3,3), dilation_rate=1,padding="same")(x)
x = Conv2D(64,(3,3), dilation_rate=3,padding="same")(x)
x = Conv2D(64,(3,3), dilation_rate=1,padding="same")(x)
x=Activation('sigmoid')(x)
x4 = Multiply()([x,x3])

x = Conv2D(64,(3,3), padding="same")(x4)
x = Add()([x, x1])

x=Conv2D(1,(3,3),padding="same")(x)

final_img = Add()([x, input_img])
model = Model(inputs=input_img, outputs=final_img)

# load the data and normalize it
cleanImages=np.load(config.data)
print(cleanImages.dtype)
cleanImages=cleanImages/255.0
cleanImages=cleanImages.astype('float32')

# define augmentor and create custom flow
aug = ImageDataGenerator(rotation_range=30, fill_mode="nearest", validation_split=0.2)

def myFlow(generator,X):
    for batch in generator.flow(x=X,batch_size=config.batch_size,seed=0):
        noise=random.randint(5,26)
        noisyImagesBatch=random_noise(batch, mode='speckle',var=noise/255.0)
        yield(noisyImagesBatch,batch)

# create custom learning rate scheduler
def lr_decay(epoch):
    initAlpha=0.0001
    factor=0.5
    dropEvery=5
    alpha=initAlpha*(factor ** np.floor((1+epoch)/dropEvery))
    return float(alpha)
callbacks=[LearningRateScheduler(lr_decay)]

# create custom loss, compile the model
print("[INFO] compilingTheModel")
opt=optimizers.Adam(learning_rate=0.0001)
def custom_loss(y_true,y_pred):
    diff=abs(y_true-y_pred)
    #l1=K.sum(diff)/(config.batch_size)
    l1=(diff)/(config.batch_size)
    return l1
model.compile(loss=custom_loss,optimizer=opt)

# train
model.fit_generator(myFlow(aug,cleanImages),
epochs=24,steps_per_epoch=len(cleanImages)//config.batch_size,callbacks=callbacks,verbose=1)

# save the model
model.save('B5.h5')
