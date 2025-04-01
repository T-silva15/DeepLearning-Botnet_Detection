import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
import random
from binaryFalsePositiveRate import BinaryFalsePositiveRate



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Dataset parameters
    NUM_FEATURES = 39
    NUM_LINES = 250000
    BATCH_SIZE = 32
    DATA_FOLDER = f'proj/datasets/sized_data/binaryclass/{NUM_LINES}_lines'
    MODEL_PATH = 'proj/models/best_binary_model.keras'
    
    # Training parameters
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    EPOCHS = 50
    LEARNING_RATE = 0.0001 
    
    # Data processing
    SHUFFLE_BUFFER = 1000
    CLIP_VALUE = 10.0 
    ENSURE_CLASS_BALANCE = True
    
# Ensure data folder exists
cfg = Config()
if not os.path.exists(cfg.DATA_FOLDER):
    raise FileNotFoundError(f"Data folder not found: {cfg.DATA_FOLDER}")

def check_class_distribution(dataset):
    """Check the distribution of classes in the dataset"""
    labels = []
    
    # Create a copy of the dataset to avoid consuming the original
    dataset_copy = dataset.map(lambda x, y: (x, y))
    
    # Unbatch first to get individual samples
    if hasattr(dataset_copy, "_batch_size") and dataset_copy._batch_size is not None:
        dataset_copy = dataset_copy.unbatch()
        
    try:
        # Collect all labels and convert to numpy
        for _, label in dataset_copy.as_numpy_iterator():
            if isinstance(label, np.ndarray):
                labels.append(float(label.item()))
            else:
                labels.append(float(label))
    except Exception as e:
        logger.warning(f"Error collecting labels: {str(e)}, using empty distribution")
        return {}
    
    if not labels:
        logger.warning("No labels collected, returning empty distribution")
        return {}
    
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    logger.info(f"Class distribution: {distribution}")
    return distribution

def load_dataset():
    """Load CSV files into a TensorFlow dataset."""
    logger.info(f"Loading data from {cfg.DATA_FOLDER}")
    
    # Find all CSV files in the data folder
    csv_files = [str(f) for f in Path(cfg.DATA_FOLDER).glob("*.csv")]
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cfg.DATA_FOLDER}")
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Create dataset from CSV files - important to use drop_remainder=False initially
    dataset = tf.data.experimental.make_csv_dataset(
        csv_files,
        batch_size=cfg.BATCH_SIZE,
        label_name='label',
        num_epochs=1,
        ignore_errors=True
    )
    
    return dataset

def preprocess_features(features, label):
    """Convert features dictionary to tensor and handle label encoding."""
    # Get feature names and sort them for consistent ordering
    feature_names = sorted(features.keys())
    
    # Convert features dictionary to tensor with explicit shape
    features_list = [tf.cast(features[key], tf.float32) for key in feature_names]
    features_tensor = tf.stack(features_list, axis=1)  # Stack along feature dimension
    
    # Replace NaN values with zeros
    features_tensor = tf.where(tf.math.is_nan(features_tensor), tf.zeros_like(features_tensor), features_tensor)
    
    # Replace inf values with large finite numbers
    features_tensor = tf.where(tf.math.is_inf(features_tensor), 1e6 * tf.ones_like(features_tensor), features_tensor)
    
    # Clip extreme values
    features_tensor = tf.clip_by_value(features_tensor, -1e6, 1e6)
    
    # Add time dimension for LSTM (batch_size, time_steps, features)
    features_tensor = tf.expand_dims(features_tensor, axis=1)
    
    # Explicitly set shape to remove any None dimensions
    features_tensor.set_shape([None, 1, cfg.NUM_FEATURES])
    
    # Convert string labels to binary (0 for BENIGN, 1 for everything else)
    label_tensor = tf.cast(tf.where(label == 'BENIGN', 0, 1), tf.float32)
    
    # Reshape label to match the expected output shape for F1Score (batch_size, 1)
    label_tensor = tf.reshape(label_tensor, [-1, 1])
    
    return features_tensor, label_tensor

def analyze_dataset_stats(dataset, num_samples=1000):
    """Analyze and print detailed statistics about the dataset values."""
    logger.info(f"Analyzing dataset statistics on {num_samples} samples...")
    
    # Collect samples
    feature_samples = []
    for features, _ in dataset.take(num_samples):
        feature_samples.append(features.numpy())
    
    if not feature_samples:
        logger.warning("No samples collected for analysis")
        return
    
    # Concatenate samples
    all_features = np.concatenate(feature_samples, axis=0)
    
    # Calculate statistics
    min_vals = np.min(all_features, axis=0)
    max_vals = np.max(all_features, axis=0)
    mean_vals = np.mean(all_features, axis=0)
    median_vals = np.median(all_features, axis=0)
    std_vals = np.std(all_features, axis=0)
    
    # Check for problematic values
    nan_count = np.isnan(all_features).sum()
    inf_count = np.isinf(all_features).sum()
    zero_count = (all_features == 0).sum()
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Shape: {all_features.shape}")
    logger.info(f"  NaN values: {nan_count}")
    logger.info(f"  Inf values: {inf_count}")
    logger.info(f"  Zero values: {zero_count}")
    logger.info(f"  Global min: {np.min(min_vals)}, max: {np.max(max_vals)}")
    logger.info(f"  Global mean: {np.mean(mean_vals)}, std: {np.mean(std_vals)}")
    
    # Check for extreme values that might cause training issues
    extreme_threshold = 1e5
    extreme_count = ((all_features > extreme_threshold) | (all_features < -extreme_threshold)).sum()
    logger.info(f"  Values beyond ±{extreme_threshold}: {extreme_count}")
    
    return {
        'min': min_vals,
        'max': max_vals,
        'mean': mean_vals,
        'median': median_vals,
        'std': std_vals
    }

def prepare_datasets():
    """Prepare training, validation and test datasets."""
    try:
        # Attempt to use TensorFlow Probability for robust statistics
        global tfp
        import tensorflow_probability as tfp
        logger.info("Using TensorFlow Probability for robust statistics")
        use_robust_stats = True
    except ImportError:
        logger.warning("TensorFlow Probability not available, falling back to standard normalization")
        use_robust_stats = False
    
    # Load raw data
    raw_dataset = load_dataset()
    
    # Apply preprocessing
    preprocessed_dataset = raw_dataset.map(preprocess_features)
    
    # Examine a sample after preprocessing
    for features, labels in preprocessed_dataset.take(1):
        logger.info(f"After preprocessing: features shape={features.shape}, labels shape={labels.shape}")
        logger.info(f"Features min: {tf.reduce_min(features)}, max: {tf.reduce_max(features)}")
    
    # Important: Unbatch the dataset after preprocessing to get individual samples
    preprocessed_dataset = preprocessed_dataset.unbatch()
    
    # Cache preprocessed dataset to avoid recomputing
    preprocessed_dataset = preprocessed_dataset.cache()
    
    # Analyze dataset statistics for potential issues
    stats = analyze_dataset_stats(preprocessed_dataset.batch(100).take(10))
    
    # Normalize features
    if use_robust_stats:
        try:
            # Use robust statistics (median/IQR) for normalization
            # Collect some samples for calculating statistics
            all_features = []
            for features, _ in preprocessed_dataset.take(10000): # subset 
                all_features.append(features)
            
            # Stack the samples
            feature_tensor = tf.stack(all_features)
            
            # Calculate robust statistics
            median = tfp.stats.percentile(feature_tensor, 50.0, axis=0)
            q75 = tfp.stats.percentile(feature_tensor, 75.0, axis=0)
            q25 = tfp.stats.percentile(feature_tensor, 25.0, axis=0)
            iqr = tf.maximum(q75 - q25, 1e-6)  # Ensure no division by zero
            
            logger.info(f"Calculated robust statistics on {len(all_features)} samples")
            
            def normalize_with_robust_stats(x, y):
                # Normalize using robust statistics
                normalized_x = (x - median) / iqr
                # Clip to avoid extreme values
                normalized_x = tf.clip_by_value(normalized_x, -cfg.CLIP_VALUE, cfg.CLIP_VALUE)
                return normalized_x, y
            
            normalized_dataset = preprocessed_dataset.map(normalize_with_robust_stats)
            logger.info("Applied robust normalization using median and IQR")
        except Exception as e:
            logger.error(f"Error with robust normalization: {str(e)}, falling back to standard")
            use_robust_stats = False
    
    if not use_robust_stats:
        # Calculate standard statistics on the unbatched dataset
        logger.info("Computing dataset statistics for standard normalization...")
        
        # Compute statistics on a sample of the data for efficiency
        features_sample = []
        for features, _ in preprocessed_dataset.take(5000):
            features_sample.append(features.numpy())
        
        if not features_sample:
            raise ValueError("No features collected for normalization")
        
        features_array = np.concatenate(features_sample, axis=0)
        mean = np.mean(features_array, axis=0, keepdims=True)
        std = np.std(features_array, axis=0, keepdims=True)
        std = np.maximum(std, 1e-6)  # Avoid division by zero
        
        # Convert to tensors
        mean_tensor = tf.constant(mean, dtype=tf.float32)
        std_tensor = tf.constant(std, dtype=tf.float32)
        
        logger.info(f"Normalization stats - Mean min: {np.min(mean)}, max: {np.max(mean)}")
        logger.info(f"Normalization stats - Std min: {np.min(std)}, max: {np.max(std)}")
        
        def normalize_with_mean_std(x, y):
            # Standard normalization
            normalized_x = (x - mean_tensor) / std_tensor
            # Clip values to avoid extremes
            normalized_x = tf.clip_by_value(normalized_x, -cfg.CLIP_VALUE, cfg.CLIP_VALUE)
            return normalized_x, y
        
        normalized_dataset = preprocessed_dataset.map(normalize_with_mean_std)
        logger.info("Applied standard normalization using mean and std")
    
    # Examine a sample after normalization
    for features, labels in normalized_dataset.take(1):
        logger.info(f"After normalization: features shape={features.shape}, labels shape={labels.shape}")
        logger.info(f"Normalized features min: {tf.reduce_min(features)}, max: {tf.reduce_max(features)}")
    
    # Cache the normalized dataset
    normalized_dataset = normalized_dataset.cache()
    
    # Separate dataset by class to ensure balanced splits
    if cfg.ENSURE_CLASS_BALANCE:
        logger.info("Ensuring class balance across dataset splits...")
        
        # Convert to numpy for easier manipulation
        # samples_list = [(x.numpy(), y.numpy()) for x, y in normalized_dataset.as_numpy_iterator()]
        samples_list = [(x.numpy(), y.numpy()) for x, y in normalized_dataset]
        

        # Separate by class
        class_0_samples = [(x, y) for x, y in samples_list if y[0] == 0]
        class_1_samples = [(x, y) for x, y in samples_list if y[0] == 1]
        
        logger.info(f"Total samples - Class 0: {len(class_0_samples)}, Class 1: {len(class_1_samples)}")
        
        # Shuffle each class separately
        random.seed(42)
        random.shuffle(class_0_samples)
        random.shuffle(class_1_samples)
        
        # Calculate split sizes for each class
        class_0_train_size = int(len(class_0_samples) * cfg.TRAIN_SPLIT)
        class_0_val_size = int(len(class_0_samples) * cfg.VAL_SPLIT)
        
        class_1_train_size = int(len(class_1_samples) * cfg.TRAIN_SPLIT)
        class_1_val_size = int(len(class_1_samples) * cfg.VAL_SPLIT)
        
        # Split the classes
        class_0_train = class_0_samples[:class_0_train_size]
        class_0_val = class_0_samples[class_0_train_size:class_0_train_size+class_0_val_size]
        class_0_test = class_0_samples[class_0_train_size+class_0_val_size:]
        
        class_1_train = class_1_samples[:class_1_train_size]
        class_1_val = class_1_samples[class_1_train_size:class_1_train_size+class_1_val_size]
        class_1_test = class_1_samples[class_1_train_size+class_1_val_size:]
        
        # Combine and shuffle
        train_samples = class_0_train + class_1_train
        val_samples = class_0_val + class_1_val
        test_samples = class_0_test + class_1_test
        
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        logger.info(f"Split sizes after balancing:")
        logger.info(f"  Train: {len(train_samples)} samples (Class 0: {len(class_0_train)}, Class 1: {len(class_1_train)})")
        logger.info(f"  Validation: {len(val_samples)} samples (Class 0: {len(class_0_val)}, Class 1: {len(class_1_val)})")
        logger.info(f"  Test: {len(test_samples)} samples (Class 0: {len(class_0_test)}, Class 1: {len(class_1_test)})")
        
        # Convert back to TensorFlow datasets
        train_dataset = tf.data.Dataset.from_tensor_slices(
            (np.array([x for x, _ in train_samples]), np.array([y for _, y in train_samples]))
        )
        val_dataset = tf.data.Dataset.from_tensor_slices(
            (np.array([x for x, _ in val_samples]), np.array([y for _, y in val_samples]))
        )
        test_dataset = tf.data.Dataset.from_tensor_slices(
            (np.array([x for x, _ in test_samples]), np.array([y for _, y in test_samples]))
        )
    else:
        # Shuffle the entire dataset
        shuffled_dataset = normalized_dataset.shuffle(
            buffer_size=cfg.SHUFFLE_BUFFER, 
            reshuffle_each_iteration=True,
            seed=42  # Set seed for reproducibility
        )
        
        # Calculate dataset size and splits
        dataset_size = tf.data.experimental.cardinality(shuffled_dataset).numpy()
        if dataset_size < 0:  # If cardinality is unknown
            dataset_size = sum(1 for _ in shuffled_dataset)
        
        logger.info(f"Total dataset size: {dataset_size} samples")
        
        train_size = int(dataset_size * cfg.TRAIN_SPLIT)
        val_size = int(dataset_size * cfg.VAL_SPLIT)
        
        logger.info(f"Dataset split: {train_size} train, {val_size} validation, {dataset_size - train_size - val_size} test samples")
        
        # Split the dataset
        train_dataset = shuffled_dataset.take(train_size)
        remaining = shuffled_dataset.skip(train_size)
        val_dataset = remaining.take(val_size)
        test_dataset = remaining.skip(val_size)
    
    # Check class distribution in each split (before batching)
    logger.info("Class distribution in training set (before batching):")
    train_dist = check_class_distribution(train_dataset)
    
    logger.info("Class distribution in validation set (before batching):")
    val_dist = check_class_distribution(val_dataset)
    
    logger.info("Class distribution in test set (before batching):")
    test_dist = check_class_distribution(test_dataset)
    
    # Calculate class weights for training
    if 0.0 in train_dist and 1.0 in train_dist:
        n_samples = train_dist[0.0] + train_dist[1.0]
        class_0_weight = n_samples / (2 * train_dist[0.0])
        class_1_weight = n_samples / (2 * train_dist[1.0])
        class_weights = {0: class_0_weight, 1: class_1_weight}
        logger.info(f"Class weights for handling imbalance: {class_weights}")
    else:
        class_weights = None
        logger.warning("Could not compute class weights - missing classes in distribution")
    
    # Batch the datasets - with drop_remainder=True to ensure consistent shapes
    train_dataset = train_dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)
    val_dataset = val_dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)
    test_dataset = test_dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)
    
    # Optimize with prefetching
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Count test batches
    test_batches = tf.data.experimental.cardinality(test_dataset).numpy()
    if test_batches < 0:  # If cardinality is unknown
        test_batches = sum(1 for _ in test_dataset)
    
    # Count all batches
    train_batches = sum(1 for _ in train_dataset)
    val_batches = sum(1 for _ in val_dataset)
    
    logger.info(f"Final dataset shapes - train: {train_batches} batches, val: {val_batches} batches, test: {test_batches} batches")
    
    # Verify final shapes of batched data
    for features, labels in train_dataset.take(1):
        logger.info(f"Final batched shapes: features={features.shape}, labels={labels.shape}")
    
    return train_dataset, val_dataset, test_dataset, test_batches, class_weights

def create_model():
    """Create and compile the CNN-LSTM model with gradient clipping."""
    logger.info("Creating CNN-LSTM model")
    
    # Define model with explicit input shape
    inputs = tf.keras.layers.Input(shape=(1, cfg.NUM_FEATURES), name="input_layer")
    
    # CNN layers
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # LSTM layers - with recurrent dropout to prevent overfitting
    x = tf.keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(32, recurrent_dropout=0.2)(x)
    
    # Dense layers with L2 regularization
    x = tf.keras.layers.Dense(16, activation='relu', 
                            kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid', name="output_layer")(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Use Adam with gradient clipping
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.LEARNING_RATE,
        clipnorm=1.0  # Clip gradients to prevent NaN values
    )
    
    # Compile model with numerically stable loss
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.BinaryCrossentropy(
            from_logits=False,  
            label_smoothing=0.1 
        ),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(
                name='f1_score',
                threshold=0.5,
                dtype=tf.float32
            ),
            BinaryFalsePositiveRate(name='false_positive_rate')
        ]
    )
    
    return model

def train_model(model, train_dataset, val_dataset, class_weights=None):
    """Train the model with appropriate callbacks."""
    logger.info(f"Training model for {cfg.EPOCHS} epochs")
    
    # Callbacks
    callbacks = [
        # Model checkpoint to save best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cfg.MODEL_PATH,
            monitor='val_accuracy',  
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping with patience
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min'
        ),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, 
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        # Terminate on NaN
        tf.keras.callbacks.TerminateOnNaN(),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='proj/logs',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    # Train the model
    history = model.fit(
        train_dataset,
        epochs=cfg.EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1,
        class_weight=class_weights
    )
    
    return history

def evaluate_model(model, test_dataset, test_batches):
    """Evaluate the model on test data."""
    logger.info("Evaluating model on test data")
    
    # Evaluate the model
    test_results = model.evaluate(test_dataset, verbose=1)
    metrics = dict(zip(model.metrics_names, test_results))
    
    # Log results
    for metric_name, value in metrics.items():
        logger.info(f"Test {metric_name}: {value:.4f}")
    
    # Generate predictions for a small sample to verify
    sample_size = min(5, test_batches)
    logger.info(f"Generating predictions for {sample_size} test batches to verify:")
    
    batch_count = 0
    for features, labels in test_dataset:
        if batch_count >= sample_size:
            break
            
        predictions = model.predict(features, verbose=0)
        
        # Check prediction values
        logger.info(f"Batch {batch_count+1} prediction stats:")
        logger.info(f"  Min: {np.min(predictions)}, Max: {np.max(predictions)}")
        logger.info(f"  Mean: {np.mean(predictions)}, Std: {np.std(predictions)}")
        
        # Check actual vs. predicted
        binary_preds = (predictions > 0.5).astype(np.int32)
        accuracy = np.mean((binary_preds == labels.numpy()).astype(np.float32))
        logger.info(f"  Batch accuracy: {accuracy:.4f}")
        
        batch_count += 1
    
    return metrics

def plot_training_history(history):
    """Plot binary training metrics in separate figures and as a combined grid."""
    logger.info("Plotting training history in separate figures (binary)")
    
    # Create results directory if it doesn't exist
    os.makedirs('proj/src/results', exist_ok=True)
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Binary Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('proj/src/results/binary_model_loss.png', dpi=300)
    plt.show()
    
    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Binary Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('proj/src/results/binary_model_accuracy.png', dpi=300)
    plt.show()
    
    # Plot AUC (if available)
    if 'auc' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['auc'], label='Training AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title("Binary AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.ylim(.9998, 1)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj/src/results/binary_model_auc.png', dpi=300)
        plt.show()
    
    # Plot F1 Score (if available)
    if 'f1_score' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['f1_score'], label='Training F1 Score')
        plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
        plt.title("Binary F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj/src/results/binary_model_f1_score.png', dpi=300)
        plt.show()
        
    # Plot Precision and Recall (if available)
    if 'precision' in history.history and 'recall' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['precision'], label="Training Precision")
        plt.plot(history.history['val_precision'], label="Validation Precision")
        plt.plot(history.history['recall'], label="Training Recall")
        plt.plot(history.history['val_recall'], label="Validation Recall")
        plt.title("Binary Precision and Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj/src/results/binary_model_precision_recall.png', dpi=300)
        plt.show()
    
    # Plot False Positive Rate (if available)
    if 'false_positive_rate' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['false_positive_rate'], label="Training FPR")
        plt.plot(history.history['val_false_positive_rate'], label="Validation FPR")
        plt.title("Binary False Positive Rate")
        plt.xlabel("Epoch")
        plt.ylabel("FPR")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj/src/results/binary_model_fpr.png', dpi=300)
        plt.show()
    
    # Combined Grid Figure (2 x 3 grid)
    # Top row: Loss, Accuracy, AUC; Bottom row: F1 Score, Precision/Recall, FPR.
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss graph (top-left)
    axs[0, 0].plot(history.history['loss'], label='Train Loss')
    axs[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axs[0, 0].set_title("Binary Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Accuracy graph (top-middle)
    axs[0, 1].plot(history.history['accuracy'], label="Train Accuracy")
    axs[0, 1].plot(history.history['val_accuracy'], label="Val Accuracy")
    axs[0, 1].set_title("Binary Accuracy")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # AUC graph (top-right)
    if 'auc' in history.history:
        axs[0, 2].plot(history.history['auc'], label="Train AUC")
        axs[0, 2].plot(history.history['val_auc'], label="Val AUC")
        axs[0, 2].set_title("Binary AUC")
        axs[0, 2].set_ylim(.9998, 1) 
        axs[0, 2].legend()
        axs[0, 2].grid(True)
    else:
        axs[0, 2].set_visible(False)
        
    # F1 Score graph (bottom-left)
    if 'f1_score' in history.history:
        axs[1, 0].plot(history.history['f1_score'], label="Train F1")
        axs[1, 0].plot(history.history['val_f1_score'], label="Val F1")
        axs[1, 0].set_title("Binary F1 Score")
        axs[1, 0].legend()
        axs[1, 0].grid(True)
    else:
        axs[1, 0].set_visible(False)
    
    # Precision & Recall graph (bottom-middle)
    if 'precision' in history.history and 'recall' in history.history:
        axs[1, 1].plot(history.history['precision'], label="Train Precision")
        axs[1, 1].plot(history.history['val_precision'], label="Val Precision")
        axs[1, 1].plot(history.history['recall'], label="Train Recall")
        axs[1, 1].plot(history.history['val_recall'], label="Val Recall")
        axs[1, 1].set_title("Binary Precision & Recall")
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    else:
        axs[1, 1].set_visible(False)
        
    # False Positive Rate graph (bottom-right)
    if 'false_positive_rate' in history.history:
        axs[1, 2].plot(history.history['false_positive_rate'], label="Train FPR")
        axs[1, 2].plot(history.history['val_false_positive_rate'], label="Val FPR")
        axs[1, 2].set_title("Binary False Positive Rate")
        axs[1, 2].legend()
        axs[1, 2].grid(True)
    else:
        axs[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("proj/src/results/binary_model_training_results_combined.png", dpi=300)
    plt.show()

def main():
    """Main function to run the entire training pipeline."""
    try:
        logger.info("Starting binary classification training pipeline")
        
        # Set memory growth for GPUs if available
        physical_devices = tf.config.list_physical_devices('GPU')
        if physical_devices:
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            logger.info(f"Found {len(physical_devices)} GPU(s), memory growth enabled")
        
        # Set up deterministic behavior as much as possible
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        # Prepare datasets with class weights for handling imbalance
        train_dataset, val_dataset, test_dataset, test_batches, class_weights = prepare_datasets()
        
        # Verify datasets before proceeding
        logger.info("Verifying datasets...")
        # Count batches explicitly to catch any issues before training
        train_batches = sum(1 for _ in train_dataset)
        val_batches = sum(1 for _ in val_dataset)
        test_batches_count = sum(1 for _ in test_dataset)
        logger.info(f"Dataset counts: {train_batches} train batches, {val_batches} val batches, {test_batches_count} test batches")
        
        # Create model
        model = create_model()
        model.summary()
        
        # Train model
        history = train_model(model, train_dataset, val_dataset, class_weights)
        
        # Evaluate model
        metrics = evaluate_model(model, test_dataset, test_batches)
        
        # Print summary of final results
        logger.info("========== FINAL TRAINING RESULTS ==========")
        logger.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
        logger.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
        
        logger.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
        logger.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
        
        if 'auc' in history.history:
            logger.info(f"Final training AUC: {history.history['auc'][-1]:.4f}")
            logger.info(f"Final validation AUC: {history.history['val_auc'][-1]:.4f}")
        
        if 'f1_score' in history.history:
            logger.info(f"Final training F1 Score: {history.history['f1_score'][-1]:.4f}")
            logger.info(f"Final validation F1 Score: {history.history['val_f1_score'][-1]:.4f}")
        
        if 'precision' in history.history:
            logger.info(f"Final training Precision: {history.history['precision'][-1]:.4f}")
            logger.info(f"Final validation Precision: {history.history['val_precision'][-1]:.4f}")
        
        if 'recall' in history.history:
            logger.info(f"Final training Recall: {history.history['recall'][-1]:.4f}")
            logger.info(f"Final validation Recall: {history.history['val_recall'][-1]:.4f}")
        
        if 'false_positive_rate' in history.history:
            logger.info(f"Final training FPR: {history.history['false_positive_rate'][-1]:.4f}")
            logger.info(f"Final validation FPR: {history.history['val_false_positive_rate'][-1]:.4f}")
        
        logger.info(f"Test set results: {metrics}")
        logger.info("==========================================")
        

        # Plot results
        plot_training_history(history)
        
        logger.info("Training pipeline completed successfully")
        return model, history, metrics
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()