from basemodel import BaseModel
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

class ImprovedCNN(BaseModel):

    def __init__(self):
        pass

    def create_improved_model(self):
        tf.compat.v1.reset_default_graph()
        improved_model = Sequential()
        improved_model.add(Conv2D(32, kernel_size=(3, 3), activation=LeakyReLU(0.1), input_shape=(32, 32, 3)))
        improved_model.add(tf.keras.layers.BatchNormalization(momentum=0.8, epsilon=0.003))
        improved_model.add(Conv2D(64, kernel_size=(3, 3), padding="same", activation=LeakyReLU(0.1)))
        improved_model.add(tf.keras.layers.BatchNormalization(momentum=0.8, epsilon=0.003))
        improved_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        improved_model.add(Conv2D(128, kernel_size=(3, 3), padding="same", activation=LeakyReLU(0.1)))
        improved_model.add(tf.keras.layers.BatchNormalization(momentum=0.8, epsilon=0.003))
        improved_model.add(MaxPooling2D(pool_size=(2, 2), strides=2))
        improved_model.add(Dropout(0.2))
        improved_model.add(Flatten())
        improved_model.add(Dense(128, activation=LeakyReLU(0.1)))
        # Output layer with softmax activation
        improved_model.add(Dense(100, activation='softmax'))

        return improved_model

    def improved_model_train(self, improved_model,
                             train_images,
                             train_labels,
                             test_images,
                             test_labels):
        batch_size = 40
        validation_split = 0.25
        no_epochs = 50
        initial_learning_rate = 0.001

        # Early stopping the runs incase the validation loss does not improve
        earlystopper = EarlyStopping(monitor='val_loss',
                                     mode='min', verbose=1,
                                     patience=10)

        # Checkpointing the model to local storage on the basis of validation accuracy
        modelcheckpointer = ModelCheckpoint('best_model_improved.h5',
                                            monitor='val_accuracy',
                                            mode='max', verbose=1,
                                            save_best_only=True)

        # Reducing the learning rate when the validation accuracy does not increase
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                                      patience=3, min_lr=0.00001)

        optimizer = tf.keras.optimizers.Adamax(learning_rate=initial_learning_rate)

        improved_model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=[
                tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            ],
        )

        # Data Augmentation using Image DataGenerator to add some variablity to the training set
        train_data_generator = ImageDataGenerator(rotation_range=10,
                                                  zoom_range=0.3,
                                                  horizontal_flip=True,
                                                  vertical_flip=False,
                                                  fill_mode='constant')

        generated_train_data = train_data_generator.flow(train_images,
                                                         train_labels,
                                                         batch_size=batch_size,
                                                         shuffle=True)

        test_data_generator = ImageDataGenerator()
        generated_test_data = test_data_generator.flow(test_images,
                                                       test_labels,
                                                       shuffle=True)

        # Training the improved model with augmented data
        history = improved_model.fit(generated_train_data,
                                     steps_per_epoch=len(generated_train_data),
                                     epochs=no_epochs,
                                     validation_steps=len(generated_test_data),
                                     verbose=1, validation_data=generated_test_data,
                                     callbacks=[modelcheckpointer, reduce_lr, earlystopper])
        # Evaluate the model
        saved_model = load_model('best_model_improved.h5')
        test_loss, test_accuracy = saved_model.evaluate_generator(generated_test_data,
                                                                  steps=len(test_images) // batch_size)
        print('Accuracy on test data : %.3f' % (test_accuracy))

        return history


