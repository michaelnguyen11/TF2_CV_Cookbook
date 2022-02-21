import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import tensorflow.keras as keras

def load_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # normalize data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # reshape grayscale to include channel dimension
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # process labels
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    return x_train, y_train, x_test, y_test

# define architecture
def build_network():
    input_layer = keras.layers.Input(shape=(28, 28, 1))
    conv1 = keras.layers.Conv2D(kernel_size=(2, 2), padding='same', strides=(2, 2), filters=32)(input_layer)
    activation1 = keras.layers.ReLU()(conv1)
    bn1 = keras.layers.BatchNormalization()(activation1)
    pooling1 = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(bn1)
    dropout = keras.layers.Dropout(rate=0.5)(pooling1)

    flatten = keras.layers.Flatten()(dropout)
    dense1 = keras.layers.Dense(units=128)(flatten)
    activation2 = keras.layers.ReLU()(dense1)
    dense2 = keras.layers.Dense(units=10)(activation2)
    output = keras.layers.Softmax()(dense2)

    network = keras.Model(inputs=input_layer, outputs=output)

    return network

# evaluate a network using the test set
def evaluate(model, x_test, y_test):
    _, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print("Accuracy: {}".format(accuracy))

x_train, y_train, x_test, y_test = load_data()
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, train_size=0.8)

model = build_network()
print(model.summary())

# plot a diagram of the network architecture
keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='test_model.jpg')
model_diagram = Image.open('test_model.jpg')
#
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
#           epochs=50, batch_size=1024, verbose=0)
#
# # saving model and weights as H5
# model.save('model_and_weights.h5')
#
# loaded_model = keras.models.load_model('model_and_weights.h5')
#
# # predicting using load model
# evaluate(loaded_model, x_test, y_test)
