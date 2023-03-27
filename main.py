import numpy as np

import my_show_methods
import show_methods
import tensorflow as tf

from load_smallnorb import load_smallnorb

class_names = ['four-legged animal','human','airplane','truck','car']


print("Loading Data \n")
(train_images, train_labels),(test_images,test_labels) = load_smallnorb()

classif_train_images = train_images[:,:,:,0]
classif_train_labels = train_labels[:,2]

print("TRAINING:")
print("Images Shape:", train_images.shape)
print("Label Shape:", train_labels.shape)

print("\nTESTING:")
print("Images Shape:", test_images.shape)
print("Label Shape:", test_labels.shape)

# TODO: Implement my own show_methods
show_methods.show_data_images(images=train_images[:16,:,:,0],labels=train_labels[:16,2],class_names=class_names,blocking=True)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1,1), activation='relu', input_shape=(96, 96,1),padding='same'))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=5, activation='softmax'))

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

train_info = model.fit(classif_train_images, classif_train_labels, epochs=2, validation_split=.33, shuffle=True)








