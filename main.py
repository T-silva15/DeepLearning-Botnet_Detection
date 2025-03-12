import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the number of features in the dataset
num_features = 39

# Number of records in the dataset
num_lines = 100000

# Define the path to the dataset folder
data_folder = f'proj/datasets/sized_data/{num_lines}_lines'

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
    
    # Check for NaN or Inf values and replace them with 0
    features = tf.where(tf.math.is_nan(features), tf.zeros_like(features), features)
    features = tf.where(tf.math.is_inf(features), tf.zeros_like(features), features)
    
    # Standardize the features
    mean = tf.reduce_mean(features, axis=0)
    std = tf.math.reduce_std(features, axis=0)
    features = (features - mean) / std
    
    # Add a time dimension
    features = tf.expand_dims(features, axis=0)  # Change axis to 0 to increase the time dimension
    
    label = tf.where(label == 'BENIGN', 0, 1)
    
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
    tf.keras.layers.Input(shape=(None, num_features)),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.LSTM(64),  
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model with a lower learning rate
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(train_dataset, epochs=25, validation_data=val_dataset, steps_per_epoch=train_size // 32, validation_steps=val_size // 32) 

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_size // 32)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Plot loss and accuracy
plt.plot(history.history['loss'])
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_accuracy'])
plt.title('Model loss and accuracy')
plt.ylabel('Loss/Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy'], loc='upper right')
plt.show()