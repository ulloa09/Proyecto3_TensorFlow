import tensorflow as tf
import mlflow

# Load an preprocess the CIFAR-10 dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

print("x_train shape:", x_train.shape)