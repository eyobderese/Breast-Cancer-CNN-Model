import os
import numpy as np
import matplotlib.pyplot as plt
from imutils import paths
from model import config
import tensorflow as tf

from model.cancernet import CancerNet
from sklearn.metrics import confusion_matrix, classification_report
from keras.utils import to_categorical
from keras.optimizers import Adagrad
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use("Agg")


# Define plot path
plot_path = "plot.png"

# Set training parameters
NUM_EPOCHS = 40
INIT_LR = 1e-2
BS = 32

# Calculate image paths in training, validation, and testing directories
trainPaths = list(paths.list_images(config.TRAIN_PATH))
totalTrain = len(trainPaths)
totalVal = len(list(paths.list_images(config.VAL_PATH)))
totalTest = len(list(paths.list_images(config.TEST_PATH)))

# Calculate class weights based on training labels
trainLabels = [int(p.split(os.path.sep)[-2]) for p in trainPaths]
trainLabels = to_categorical(trainLabels)
classTotals = trainLabels.sum(axis=0)
classWeight = {i: classTotals.max(
) / classTotals[i] for i in range(len(classTotals))}

# Initialize data augmentation
trainAug = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.05,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest")
valAug = ImageDataGenerator(rescale=1 / 255.0)

# Initialize training, validation, and testing generators
trainGen = trainAug.flow_from_directory(
    config.TRAIN_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=True,
    batch_size=BS)
valGen = valAug.flow_from_directory(
    config.VAL_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)
testGen = valAug.flow_from_directory(
    config.TEST_PATH,
    class_mode="categorical",
    target_size=(48, 48),
    color_mode="rgb",
    shuffle=False,
    batch_size=BS)


batch_x, batch_y = next(trainGen)
print("Image batch shape:", batch_x.shape)   # Should be (BS, 48, 48, 3)
# Should be (BS, num_classes) if "categorical"
print("Label batch shape:", batch_y.shape)
print("Image batch dtype:", batch_x.dtype)   # Should be float32
print("Label batch dtype:", batch_y.dtype)

# # Initialize and compile model
model = CancerNet.build(width=48, height=48, depth=3, classes=2)
opt = Adagrad(learning_rate=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
print(model.input_shape)

# Test MOdel
# H = model.fit(
#     x=trainGen,
#     steps_per_epoch=2,
#     validation_data=valGen,
#     validation_steps=2,
#     epochs=2
# )

# Train the model
H = model.fit(
    x=trainGen,
    steps_per_epoch=totalTrain // BS,
    validation_data=valGen,
    validation_steps=totalVal // BS,
    epochs=NUM_EPOCHS)

# Evaluate the model
print("[INFO] evaluating network...")
testGen.reset()
predIdxs = model.predict(x=testGen, steps=(totalTest // BS) + 1)
predIdxs = np.argmax(predIdxs, axis=1)
print(classification_report(testGen.classes, predIdxs,
      target_names=testGen.class_indices.keys()))

# Calculate confusion matrix, accuracy, sensitivity, and specificity
cm = confusion_matrix(testGen.classes, predIdxs)
total = sum(sum(cm))
acc = (cm[0, 0] + cm[1, 1]) / total
sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
print(cm)
print("acc: {:.4f}".format(acc))
print("sensitivity: {:.4f}".format(sensitivity))
print("specificity: {:.4f}".format(specificity))

# Save the model
model.save("cancer_detection_model.h5")
print("[INFO] Model saved to disk.")

# Plot the training loss and accuracy
N = NUM_EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(plot_path)
