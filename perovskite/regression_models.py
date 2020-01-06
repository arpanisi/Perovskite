from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.layers import BatchNormalization
from keras.models import Model, model_from_json
from keras import optimizers
from keras.regularizers import l1, l2, l1_l2
import keras.backend as K
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.utils import plot_model
import numpy as np
import json


class RegressionModel:

    def __init__(self):

        self.model = None

    def build_model(self):

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu',
                          input_shape=(32, 32, 1)))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Flatten())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dense(4, activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Dense(1))

    def compile_regression_model(self):

        if self.model is None:
            raise ValueError('Model needs to be built first')
        self.model.compile(loss='mse', optimizer=Adam(lr=0.01))

    def r_squared(self, X_test, y_test):

        y_true = y_test
        y_pred = self.model.predict(X_test).reshape(len(y_true))
        SS_res = np.sum((y_true - y_pred) ** 2)
        SS_T = np.sum((y_true - np.mean(y_true)) ** 2)

        return 1 - SS_res / SS_T, y_test - y_pred, y_pred

    def detect_outliers(self, X, y, tol=1):

        y_pred = self.model.predict(X)
        e = y_pred - y

        return np.abs(e) > tol

    def save_model(self, value=None, filename=None, weight_file=None):

        if value is None:
            raise ValueError('Must pass value name')
        model_json = self.model.to_json()
        if filename is None:
            filename = 'regression_model_'+value+'.json'
        with open(filename, 'w') as json_file:
            json_file.write(model_json)

        if weight_file is None:
            return

        self.model.save_weights('regression_model_'+value+'.h5')

    def from_json_file(self, json_file_name=None, weight_file_name=None):

        with open(json_file_name, 'r') as f:
            self.model = model_from_json(f.read())

        self.model.load_weights(weight_file_name)

    def plot_regression_model(self, filename='architecture.png', show_shapes=True, show_layer_names=True):

        plot_model(self.model, to_file=filename, show_shapes=show_shapes, show_layer_names=show_layer_names)

    def plot_training_history(self, history):

        plt.rcParams['font.size'] = 20
        fig, ax = plt.subplots()
        ax.grid('on')
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_ylabel('Training Loss')
        ax.set_xlabel('Iterations')
        ax.legend()