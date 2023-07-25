#!/usr/bin/env python
# coding: utf-8

# ## Transfer Learning ResNet50 modified by Noman

# Here the environment name is class. Performed on datasets containing test, training and validation

# In[3]:


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)


# In[4]:


# import the libraries as shown below

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet import ResNet101
#from keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.models import Sequential
import numpy as np
from glob import glob
#import matplotlib.pyplot as plt


# In[5]:


import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print(physical_devices)


# In[6]:


# re-size all the images to this
IMAGE_SIZE = [224, 224]

train_path = 'datasets/training'
valid_path = 'Datasets/validation'


# In[7]:


# Import the Vgg 16 library as shown below and add preprocessing layer to the front of VGG
# Here we will be using imagenet weights

resnet = ResNet101(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)



# In[8]:


# don't train existing weights
for layer in resnet.layers:
    layer.trainable = False


# In[9]:


# useful for getting number of output classes
folders = glob('datasets/training/*')


# In[10]:


# our layers - you can add more if you want
x = Flatten()(resnet.output)


# In[11]:


prediction = Dense(len(folders), activation='softmax')(x)

# create a model object
model = Model(inputs=resnet.input, outputs=prediction)


# In[12]:



# view the structure of the model
model.summary()


# In[13]:


# tell the model what cost and optimization method to use
model.compile(
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)


# In[14]:


# Use the Image Data Generator to import the images from the dataset
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)


# In[15]:


# Make sure you provide the same target size as initialied for the image size
training_set = train_datagen.flow_from_directory('datasets/training',
                                                 target_size = (224, 224),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')


# In[18]:


valid_set = validation_datagen.flow_from_directory('datasets/validation',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')


# In[19]:


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 3)
cb_checkpointer = ModelCheckpoint(filepath = 'working', monitor = 'val_loss', save_best_only = True, mode = 'auto')


# In[ ]:


# fit the model
# Run the cell. It will take some time to execute
r = model.fit_generator(
  training_set,
  validation_data=valid_set,
  epochs=200,
  steps_per_epoch=len(training_set),
  validation_steps=len(valid_set)
)
model.save_weights("working/model_resnet101.h5")


# In[ ]:


model.save("ResNet101.h5")


# In[ ]:


test_path = 'datasets/test'
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Datasets/test',
                                            target_size = (224, 224),
                                            batch_size = 32,
                                            class_mode = 'categorical')
result1=model.evaluate(training_set)
result2=model.evaluate(valid_set)
result3=model.evaluate(test_set)


# In[ ]:


print("Training data accuracy:", result1)
print("Validation data accuracy:", result2)
print("Test data accuracy:", result3)


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


print(r.history.keys())


# In[ ]:



plt.figure(1, figsize = (15,8)) 
    
plt.subplot(221)  
plt.plot(r.history['accuracy']) 
plt.plot(r.history['val_accuracy'])  
plt.title('model accuracy')  
plt.ylabel('accuracy')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 
    
plt.subplot(222),  
plt.plot(r.history['loss'])  
plt.plot(r.history['val_loss'])  
plt.title('model loss')  
plt.ylabel('loss')  
plt.xlabel('epoch')  
plt.legend(['train', 'valid']) 

plt.show()


# In[ ]:


# plot the loss
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')




