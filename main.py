import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

# Define the number of features in the dataset
num_features = 39

# Number of records in the dataset
num_lines = 10000

# Define the path to the dataset folder
data_folder = f'proj/datasets/sized_data/{num_lines}_lines'

# Function to load and preprocess the dataset
def load_dataset(data_folder):
    # List all CSV files in the data folder
    csv_files = [os.path.join(data_folder, f) for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    # Create a dataset from the CSV files
    dataset = tf.data.experimental.make_csv_dataset(
        csv_files,
        batch_size=100,  
        label_name='label',  
        num_epochs=1,
        ignore_errors=False
    )
    
    return dataset.shuffle(buffer_size=num_lines, reshuffle_each_iteration=False)

# Load the dataset
dataset = load_dataset(data_folder)

# Split the dataset into training, validation, and test sets
dataset_size = sum(1 for _ in dataset)
train_size = int(0.7 * dataset_size)
val_size = int(0.2 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

# Very basic LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(None, num_features)),  
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Train the model
history = model.fit(train_dataset, epochs=50, validation_data=val_dataset) 

# Evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)
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

