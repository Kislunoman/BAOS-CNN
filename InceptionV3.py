#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt


# In[2]:


train_dir = os.path.join('Train')
validation_dir = os.path.join('Validate')


# In[3]:


train_Background_dir = os.path.join(train_dir, 'Background')

# Directory with our training dog pictures
train_Ferny_dir = os.path.join(train_dir, 'Ferny')
train_Rounded_dir = os.path.join(train_dir, 'Rounded')
train_Strappy_dir = os.path.join(train_dir, 'Strappy')

# Directory with our validation cat pictures
validation_Background_dir = os.path.join(validation_dir, 'Background')

# Directory with our validation dog pictures
validation_Ferny_dir = os.path.join(validation_dir, 'Ferny')
validation_Rounded_dir = os.path.join(validation_dir, 'Rounded')
validation_Strappy_dir = os.path.join(validation_dir, 'Strappy')


# In[4]:


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )


# In[5]:


# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'categorical', target_size = (224, 224))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 20, class_mode = 'categorical', target_size = (224, 224))


# In[6]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(input_shape = (224, 224, 3), include_top = False, weights = 'imagenet')


# In[7]:


for layer in base_model.layers:
    layer.trainable = False


# In[8]:


from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(4, activation='softmax')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])


# In[9]:


from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

cb_early_stopper = EarlyStopping(monitor = 'val_loss', patience = 3)
cb_checkpointer = ModelCheckpoint(filepath = 'working', monitor = 'val_loss', save_best_only = True, mode = 'auto')


# In[ ]:


inc_history = model.fit_generator(train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 10)


# In[ ]:


model.save_weights("InceptionNewV3.h5")


# In[ ]:


test_path = 'Test'
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Test',
                                            target_size = (224, 224),
                                            batch_size = 4,
                                            class_mode = 'categorical')


# In[ ]:


result3=model.evaluate(test_set)


# In[ ]:


result1=model.evaluate(train_generator)


# In[ ]:


result2=model.evaluate(validation_generator)


# In[ ]:


print("Train accuracy = ", result1)
print("Valdation accuracy = ", result2)
print("Test accuracy = ", result3)

