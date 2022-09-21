#File to house base version of image classification model using a custom CNN architecture
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import sparse_categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
from keras.models import load_model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, UpSampling2D, Dropout,BatchNormalization,GlobalAveragePooling2D, MaxPooling2D, Activation
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical

class BaseModel:

    def __init__(self):
        pass

    def load_data(self):
        cifar100 = tf.keras.datasets.cifar100
        (train_images, train_labels), (test_images, test_labels) = cifar100.load_data()
        return train_images, train_labels, test_images, test_labels

    def process_images(self, train_images, test_images):
        #Normalizing the image
        train_images = train_images.astype('float32')
        test_images = test_images.astype('float32')

        return train_images / 255, test_images / 255

    def create_baseline_model(self):
        tf.compat.v1.reset_default_graph()
        baseline_model = Sequential()
        baseline_model.add(layers.Conv2D(16, (7, 7), activation='relu', input_shape=(32, 32, 3), strides=1))
        baseline_model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        baseline_model.add(layers.Conv2D(32, (5, 5), activation='relu', strides=1))
        baseline_model.add(layers.MaxPooling2D(pool_size=2, strides=2))
        baseline_model.add(layers.Flatten())
        baseline_model.add(layers.Dense(128, activation='relu'))
        baseline_model.add(layers.Dense(100))

        return baseline_model

    def baseline_model_train(self, baseline_model, train_images, train_labels):
        batch_size = 32
        validation_split = 0.25
        verbosity = 1
        no_epochs = 30
        initial_learning_rate = 0.001

        # Early stopping the runs incase the validation loss does not improve
        earlystopper = EarlyStopping(monitor='val_loss', mode='min',
                                     verbose=1, patience=7)
        # Checkpointing the model to local storage on the basis of validation accuracy
        modelcheckpointer = ModelCheckpoint('best_model_baseline.h5',
                                            monitor='val_accuracy',
                                            mode='max',
                                            verbose=1, save_best_only=True)
        # Reducing learning rate on the basis of valiation loss
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=4, min_lr=0.0001)

        baseline_model.compile(
            optimizer=tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"), ], )

        history = baseline_model.fit(train_images, train_labels,
                                     batch_size=batch_size,
                                     epochs=no_epochs,
                                     verbose=verbosity,
                                     validation_split=validation_split,
                                     callbacks=[modelcheckpointer, reduce_lr, earlystopper])

        # Load the model and evaluate it
        saved_model = load_model('best_model_baseline.h5')
        _, test_accuracy = saved_model.evaluate(test_images, test_labels, verbose=1)
        print('Accuracy on test data : %.2f' % (test_accuracy))

        return history

    # Method to plot the accuracy over training set and validation set
    def create_accuracy_plot(self, model_hist):
        plt.plot(model_hist.history["accuracy"])
        plt.plot(model_hist.history["val_accuracy"])
        plt.title("model performance")
        plt.ylabel("accuracy score")
        plt.xlabel("num_epoch")
        plt.legend(["train_set", "validation_set"], loc="upper left")
        plt.show()


    if __name__ == "__main__":
        baseline_model = create_baseline_model()
        print(baseline_model.summary())

        tf.keras.utils.plot_model(
            baseline_model, to_file='baseline_model.png', show_shapes=False, show_dtype=False,
            show_layer_names=True, rankdir='TB', expand_nested=False, dpi=96
        )

        train_hist_baseline = baseline_model_train(baseline_model, train_images, train_labels)
        create_accuracy_plot(train_hist_baseline)