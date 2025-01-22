import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.utils import image_dataset_from_directory, plot_model
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report,confusion_matrix, ConfusionMatrixDisplay
import time
import os
import psutil
import matplotlib.pyplot as plt

# Paths to the dataset directories
base_dir = r"C:\Users\User\.cache\kagglehub\datasets\thitikornindee\dr-resized-2015-and-2019\versions\3\DR_2Classes_CLAHE_LAB\DR_2Classes_CLAHE_LAB"
train_dir = os.path.join(base_dir, "train")
valid_dir = os.path.join(base_dir, "valid")
test_dir = os.path.join(base_dir, "test")

# Hyperparameters used for the training
image_size = (299, 299)
batch_size = 32
epochs = 30
learning_rate = 0.0001

# Load train/valid/test datasets
train_dataset = image_dataset_from_directory(train_dir, image_size=image_size, batch_size=batch_size)
valid_dataset = image_dataset_from_directory(valid_dir, image_size=image_size, batch_size=batch_size)
test_dataset = image_dataset_from_directory(test_dir, image_size=image_size, batch_size=batch_size)

# Preprocess train/valid/test datasets
train_dataset = train_dataset.map(lambda image, label: (tf.keras.applications.inception_v3.preprocess_input(image), label))
valid_dataset = valid_dataset.map(lambda image, label: (tf.keras.applications.inception_v3.preprocess_input(image), label))
test_dataset = test_dataset.map(lambda image, label: (tf.keras.applications.inception_v3.preprocess_input(image), label))

# Prefetch train/valid/test datasets for better performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
valid_dataset = valid_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Pre-trained model InceptionV3
base_model = InceptionV3(input_shape=(299, 299, 3), include_top=False, weights="imagenet")
base_model.trainable = False

# Custom model to replace top layers of pre-trained model
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(526, activation='relu'),
    Dropout(0.1),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Evaluation Metrics Function
class MetricsCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.train_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        val_true = []
        val_pred = []

        # Collect ground truth and predictions for the validation dataset
        for images, labels in valid_dataset:
            preds = tf.round(self.model.predict(images)).numpy().flatten()
            val_true.extend(labels.numpy())
            val_pred.extend(preds)

        # Calculate precision, recall, and F1-score
        f1 = f1_score(val_true, val_pred)
        precision = precision_score(val_true, val_pred)
        recall = recall_score(val_true, val_pred)

        print(f"Epoch {epoch + 1} - F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    def on_train_end(self, logs=None):
        elapsed_time = time.time() - self.train_start_time
        print(f"Training completed in {elapsed_time / 60:.2f} minutes.")

# Save model architecture to an image file to get architectures image. Available in the report.
plot_model(model, to_file="model_architecture.png", show_shapes=True, show_layer_names=True)

# Monitor memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB
    print(f"Memory Usage: {memory_usage:.2f} MB")

# Callbacks
metrics_callback = MetricsCallback()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate * 0.1 ** (epoch // 5))
callbacks = [metrics_callback, lr_schedule]

# Model training
log_memory_usage()
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=epochs,
    callbacks=callbacks
)
log_memory_usage()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# F1-Score calculation
test_true = []
test_pred = []
for images, labels in test_dataset:
    preds = tf.round(model.predict(images)).numpy().flatten()
    test_true.extend(labels.numpy())
    test_pred.extend(preds)

f1 = f1_score(test_true, test_pred)
precision = precision_score(test_true, test_pred)
recall = recall_score(test_true, test_pred)

print(f"Test F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
print("Classification Report:")
print(classification_report(test_true, test_pred, target_names=["No_DR", "DR"]))

# Traing curves graph. Available in report.
def plot_training_curves(history):
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(accuracy) + 1)

    plt.figure()
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_validation_accuracy.png")

    plt.figure()
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_validation_loss.png")

plot_training_curves(history)

# Confusion matrix graph. Available in report.
def plot_confusion_matrix(test_true, test_pred, class_names=["No_DR", "DR"]):
    cm = confusion_matrix(test_true, test_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    plt.figure()
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")

plot_confusion_matrix(test_true, test_pred)
