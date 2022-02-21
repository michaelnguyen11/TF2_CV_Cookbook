import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_docs as tfdocs
import tensorflow_docs.plots

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

def load_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    # normalize the data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # reshape grayscale to include channel dimension
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)

    # one-hot encode the labels
    label_binarizer = LabelBinarizer()
    y_train = label_binarizer.fit_transform(y_train)
    y_test = label_binarizer.fit_transform(y_test)

    (x_train, x_val, y_train, y_val) = train_test_split(x_train, y_train, train_size=0.8)

    # feed data to tensorflow dataset
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    return train_ds, val_ds, test_ds

def build_network():
    input_layer = keras.layers.Input(shape=(28, 28, 1))
    x = keras.layers.Conv2D(filters=20, kernel_size=(5, 5), padding='same', strides=(1, 1))(input_layer)
    x = keras.layers.ELU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Conv2D(filters=50, kernel_size=(5, 5), padding='same', strides=(1, 1))(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(units=500)(x)
    x = keras.layers.ELU()(x)
    x = keras.layers.Dropout(0.5)(x)

    x = keras.layers.Dense(10)(x)
    output = keras.layers.Softmax()(x)

    model = keras.Model(inputs=input_layer, outputs=output)

    return model

# function that takes a model's training history, along with metrics of interest
# as well as  create a plot corresponding to the training and validation of the curves of such a metric
def plot_model_history(model_history, metric, ylim=None):
    plt.style.use('seaborn-darkgrid')
    plotter = tfdocs.plots.HistoryPlotter()
    plotter.plot({'Model': model_history}, metric=metric)

    plt.title(f'{metric.upper()}')
    if ylim is None:
        plt.ylim([0, 1])
    else:
        plt.ylim(ylim)

    plt.savefig(f'{metric}.png')
    plt.close()

if __name__ == '__main__':
    # dataset setup
    BATCH_SIZE = 256
    BUFFER_SIZE = 1024

    train_dataset, val_dataset, test_dataset = load_dataset()

    # the prefetch() method spawns a background thread that populates a buffer of BUFFER_SIZE with image batches
    train_dataset = train_dataset.shuffle(buffer_size=BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size= BUFFER_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(buffer_size=BUFFER_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # build and train the network
    EPOCHS = 100
    model = build_network()
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model_history = model.fit(train_dataset, validation_data=val_dataset, epochs=EPOCHS, verbose=0)

    # plot the training and validation loss and accuracy
    plot_model_history(model_history, 'loss', [0., 2.0])
    plot_model_history(model_history, 'accuracy')

    # Visualize the model architecture
    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')
    model.save('basic_image_classifier.h5')
    # load and evaluate the model
    loaded_model = keras.models.load_model('basic_image_classifier.h5')
    results = loaded_model.evaluate(test_dataset, verbose=0)
    print(f'Loss: {results[0]}, Accuracy: {results[1]}')

