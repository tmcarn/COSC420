import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import gzip
import pickle
from load_smallnorb import load_smallnorb

# Load in Dataset
(train_images, train_labels), (test_images, test_labels) = load_smallnorb()

# Trains only on first image of the pair
classif_train_images = train_images[:, :, :, 0]
classif_train_labels = train_labels[:, 2]

classif_test_images = test_images[:, :, :, 0]
classif_test_labels = test_labels[:, 2]

category_labels = ['animal', 'human', 'airplane', 'truck', 'car']
n_classes = len(category_labels)

print("Image Shape:", classif_train_images.shape)
print("Label Shape:", classif_train_labels.shape)


if not os.path.isdir("saved"):
      os.mkdir('saved')

runPretrained = input("Do you want to use pretrained weights? y/n")

history=[]
if runPretrained=='y':
    fname = input("What is the name of the network? ")
    path = os.path.join('saved',fname)
    weights = path+'_weights.h5'
    hist = path+'_history.hist'

    if os.path.isfile(weights):
        print("Loading weights from: ", weights)
        net = tf.keras.models.load_model(weights)

        if os.path.isfile(hist):
            print("Loading history from:", hist)
            with gzip.open(hist) as f:
                history = pickle.load(f)
    else:
        print("That file dne")

else:
    print("Training new weights...")


    #Train new network

    # Create feed-forward network
    net = tf.keras.models.Sequential()
    net.add(tf.keras.layers.Conv2D(filters=32,
                                   kernel_size=(3, 3),
                                   strides=(1,1),
                                   activation='relu',
                                   input_shape=(96, 96, 1),
                                   padding='same'))

    net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),
                                         strides=(2,2)))

    net.add(tf.keras.layers.Flatten())

    net.add(tf.keras.layers.Dense(units=n_classes, activation='softmax'))


    net.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    train_info = net.fit(classif_train_images, classif_train_labels,
                         validation_split=0.33,
                         epochs=2,
                         shuffle=True)

    willSave = input("Would you like to save the weights and history from this training? y/n")

    if willSave=='y':
        fname = input("Enter a name for the weights: ")
        save_name = os.path.join('saved', fname)
        saved_weights_name = save_name + "_weights.h5"
        saved_hist_name = save_name + "_history.hist"
        net.save(saved_weights_name)
        history = train_info.history

        with gzip.open(saved_hist_name, 'w') as f:
            pickle.dump(history, f)



if history != []:
    print("plotting...")
    # figure, axis = plt.subplots(2,2)
    plt.plot(history['accuracy'], label='accuracy')
    plt.plot(history['val_accuracy'], label = 'val_accuracy')
    plt.legend(loc='lower right')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0, 1])
    plt.title('Training vs. Validation Accuracy')
    plt.show()


# *********************************************************
# * Evaluating the neural network model within tensorflow *
# *********************************************************

loss_train, accuracy_train = net.evaluate(classif_train_images,  classif_train_labels, verbose=0)
loss_test, accuracy_test = net.evaluate(classif_test_images, classif_test_labels, verbose=0)

print("Train accuracy (tf): %.2f" % accuracy_train)
print("Test accuracy  (tf): %.2f" % accuracy_test)

# Compute output for 16 test images
y_test = net.predict(classif_test_images[:16])
y_test = np.argmax(y_test, axis=1)
print(y_test)



