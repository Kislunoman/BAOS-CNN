
pip install tensorflow
pip install keras
Now, let's create the deep CNN model:


import tensorflow as tf
from tensorflow.keras import layers, models

def create_deep_cnn_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional Block 1
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Convolutional Block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten the output of the previous layer
    model.add(layers.Flatten())

    # Fully connected layers
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

input_shape = (128, 128, 3)  # Change this to your image size
num_classes = 10             # Change this to the number of classes in your dataset

model = create_deep_cnn_model(input_shape, num_classes)
model.summary()  # To see the summary of the model