import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os

# Define the number of features in the dataset
num_features = 39

# Number of records in the dataset
num_lines = 100000

# Define the path to the dataset folder
data_folder = f'proj/datasets/sized_data/multiclass/{num_lines}_lines'

# Function to load and preprocess the dataset
def load_dataset(data_folder):
    # List all CSV files in the data folder
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]

    # Create a dataset from the CSV files
    dataset = tf.data.experimental.make_csv_dataset(
        csv_files,
        batch_size=32,
        label_name='label',
        num_epochs=1,
        ignore_errors=False
    )

    return dataset

# Function to preprocess the dataset
def preprocess(features, label): 
    # Convert the dictionary of features to a single tensor
    features = tf.concat([tf.cast(tf.expand_dims(tensor, axis=-1), tf.float32) for tensor in features.values()], axis=-1)

    # Check for NaN values and replace them with 0
    features = tf.where(tf.math.is_nan(features), tf.zeros_like(features), features)

    # Standardize the features
    mean = tf.reduce_mean(features, axis=0)
    std = tf.math.reduce_std(features, axis=0)
    features = (features - mean) / std

    # Add a time dimension
    features = tf.expand_dims(features, axis=1)  # Add time dimension (batch_size, 1, num_features)

    return features, label

# Load the dataset
dataset = load_dataset(data_folder)

# Shuffle the dataset
dataset = dataset.shuffle(buffer_size=num_lines, reshuffle_each_iteration=False)

# Preprocess the dataset
dataset = dataset.map(preprocess)

# Split the dataset into training, validation, and test sets
dataset_size = sum(1 for _ in dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size).repeat()
val_dataset = dataset.skip(train_size).take(val_size).repeat()
test_dataset = dataset.skip(train_size + val_size)

# CNN-LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1, num_features)),  # Input shape: (batch_size, 1, num_features)
    tf.keras.layers.Conv1D(256, 1, activation='relu'),  
    tf.keras.layers.MaxPooling1D(1),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(256, return_sequences=True),  
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(128),  #
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(4, activation='softmax')  
])

model.compile(optimizer=tf.keras.optimizers.Adagrad(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

# Ensure the directory exists
os.makedirs('proj/models', exist_ok=True)

# Define the checkpoint callback to save the best model based on validation accuracy
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath='proj/models/best_multiclass_model.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# Train the model with the checkpoint callback
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    steps_per_epoch=train_size // 32,
    validation_steps=val_size // 32,
    callbacks=[checkpoint_callback]
)

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_size // 32)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plot loss and accuracy
plt.figure(figsize=(12, 6))

# Subplot for training
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Train Accuracy'], loc='upper right')

# Subplot for validation
plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Val Loss', 'Val Accuracy'], loc='upper right')

plt.tight_layout()
plt.show()