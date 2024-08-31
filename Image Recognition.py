import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import random

(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

y_train[:5]
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


X_train = X_train / 255.0
X_test = X_test / 255.0

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(100, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=20)

test_loss, test_acc = cnn.evaluate(X_test, y_test)
print("Test accuracy:", test_acc)

y_pred = cnn.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

def plot_random_sample(X, y):
    index = random.randint(0, len(X) - 1)
    plt.figure(figsize=(3, 3))
    plt.imshow(X[index])
    plt.xlabel(f"Pred: {classes[y_pred_classes[index]]}, Actual: {classes[y[index]]}")
    plt.show()
    print("Predicted class:", classes[y_pred_classes[index]])
    print("Actual class:", classes[y_test[index]])

plot_random_sample(X_test, y_test)
