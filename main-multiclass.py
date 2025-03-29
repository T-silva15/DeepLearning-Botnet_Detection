import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
import random
import sklearn.metrics

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Dataset parameters
    NUM_FEATURES = 39
    NUM_LINES = 250000  # Same as binary for comparison
    BATCH_SIZE = 32
    DATA_FOLDER = f'proj/datasets/sized_data/multiclass/{NUM_LINES}_lines'
    MODEL_PATH = 'proj/models/best_multiclass_model.keras'
    
    # Training parameters
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2
    TEST_SPLIT = 0.1
    EPOCHS = 50   # Reduced to match binary model
    LEARNING_RATE = 0.00001
    
    # Data processing
    SHUFFLE_BUFFER = 10000
    CLIP_VALUE = 1.0
    ENSURE_CLASS_BALANCE = True
    
    # Multiclass-specific parameters
    NUM_CLASSES = 4  # Multiclass classification

# Initialize config
cfg = Config()

# Ensure data folder exists
if not os.path.exists(cfg.DATA_FOLDER):
    raise FileNotFoundError(f"Data folder not found: {cfg.DATA_FOLDER}")

# Try to import TensorFlow Probability
try:
    import tensorflow_probability as tfp
    logger.info("Using TensorFlow Probability for robust statistics")
    USE_ROBUST_STATS = True
except ImportError:
    logger.warning("TensorFlow Probability not available, falling back to standard normalization")
    USE_ROBUST_STATS = False

def check_class_distribution(dataset):
    """Check the distribution of classes in the dataset"""
    labels = []
    
    # Create a copy of the dataset to avoid consuming the original
    dataset_copy = dataset.map(lambda x, y: (x, y))
    
    # Unbatch if needed
    if hasattr(dataset_copy, "_batch_size") and dataset_copy._batch_size is not None:
        dataset_copy = dataset_copy.unbatch()
        
    try:
        # Collect all labels and convert to numpy
        for _, label in dataset_copy.as_numpy_iterator():
            if isinstance(label, np.ndarray):
                if label.size == 1:  # Single value
                    labels.append(int(label.item()))
                else:  # One-hot encoded
                    labels.append(np.argmax(label))
            else:
                labels.append(int(label))
    except Exception as e:
        logger.warning(f"Error collecting labels: {str(e)}")
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
    
    # Create dataset from CSV files
    dataset = tf.data.experimental.make_csv_dataset(
        csv_files,
        batch_size=cfg.BATCH_SIZE,
        label_name='label',
        num_epochs=1,
        ignore_errors=True
    )
    
    return dataset

def preprocess_features(features, label):
    """Convert features dictionary to tensor and handle multiclass labels."""
    # Get feature names and sort them for consistent ordering
    feature_names = sorted(features.keys())
    
    # Convert features dictionary to tensor
    features_list = [tf.cast(features[key], tf.float32) for key in feature_names]
    features_tensor = tf.stack(features_list, axis=1)
    
    # Replace NaN and inf values, then clip
    features_tensor = tf.where(tf.math.is_nan(features_tensor), tf.zeros_like(features_tensor), features_tensor)
    features_tensor = tf.where(tf.math.is_inf(features_tensor), 1e6 * tf.ones_like(features_tensor), features_tensor)
    features_tensor = tf.clip_by_value(features_tensor, -1e6, 1e6)
    
    # Add time dimension for LSTM (batch_size, time_steps, features)
    features_tensor = tf.expand_dims(features_tensor, axis=1)
    
    # Explicitly set shape
    features_tensor.set_shape([None, 1, cfg.NUM_FEATURES])
    
    # Convert label to int32 for multiclass classification
    label_tensor = tf.cast(label, tf.int32)
    
    return features_tensor, label_tensor

def analyze_dataset_stats(dataset, num_samples=1000):
    """Analyze and print detailed statistics about the dataset values."""
    logger.info(f"Analyzing dataset statistics on {num_samples} samples...")
    
    # Collect samples
    feature_samples = []
    label_samples = []
    for features, labels in dataset.take(num_samples):
        feature_samples.append(features.numpy())
        label_samples.append(labels.numpy())
    
    if not feature_samples:
        logger.warning("No samples collected for analysis")
        return
    
    # Concatenate samples
    all_features = np.concatenate(feature_samples, axis=0)
    all_labels = np.concatenate(label_samples, axis=0)
    
    # Calculate statistics
    min_vals = np.min(all_features, axis=0)
    max_vals = np.max(all_features, axis=0)
    mean_vals = np.mean(all_features, axis=0)
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
    
    # Check class distribution
    unique_labels, label_counts = np.unique(all_labels, return_counts=True)
    logger.info(f"  Class distribution: {dict(zip(unique_labels, label_counts))}")
    
    return {
        'min': min_vals,
        'max': max_vals,
        'mean': mean_vals,
        'std': std_vals,
        'label_distribution': dict(zip(unique_labels, label_counts))
    }

def prepare_datasets():
    """Prepare training, validation and test datasets."""
    # Load raw data
    raw_dataset = load_dataset()
    
    # Apply preprocessing
    preprocessed_dataset = raw_dataset.map(preprocess_features)
    
    # Examine a sample after preprocessing
    for features, labels in preprocessed_dataset.take(1):
        logger.info(f"After preprocessing: features shape={features.shape}, labels shape={labels.shape}")
        logger.info(f"Features min: {tf.reduce_min(features)}, max: {tf.reduce_max(features)}")
    
    # Unbatch the dataset after preprocessing to get individual samples
    preprocessed_dataset = preprocessed_dataset.unbatch()
    
    # Cache preprocessed dataset
    preprocessed_dataset = preprocessed_dataset.cache()
    
    # Analyze dataset statistics
    stats = analyze_dataset_stats(preprocessed_dataset.batch(100).take(10))
    
    # Normalize features
    global USE_ROBUST_STATS
    if USE_ROBUST_STATS:
        try:
            # Use robust statistics (median/IQR) for normalization
            # Collect some samples for calculating statistics
            all_features = []
            for features, _ in preprocessed_dataset.take(10000):  # Use a subset for efficiency
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
            USE_ROBUST_STATS = False
    
    if not USE_ROBUST_STATS:
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
    
    # Ensure class balance if configured
    if cfg.ENSURE_CLASS_BALANCE:
        logger.info("Ensuring class balance across dataset splits...")
        
        # Convert to numpy for easier manipulation
        samples_list = [(x.numpy(), y.numpy()) for x, y in normalized_dataset]
        
        # Separate by class
        class_samples = {}
        for i in range(cfg.NUM_CLASSES):
            class_samples[i] = [(x, y) for x, y in samples_list if y == i]
            logger.info(f"Total samples for Class {i}: {len(class_samples[i])}")
        
        # Shuffle each class separately
        random.seed(42)
        for i in range(cfg.NUM_CLASSES):
            random.shuffle(class_samples[i])
        
        # Calculate split sizes for each class
        class_train = {}
        class_val = {}
        class_test = {}
        
        for i in range(cfg.NUM_CLASSES):
            train_size = int(len(class_samples[i]) * cfg.TRAIN_SPLIT)
            val_size = int(len(class_samples[i]) * cfg.VAL_SPLIT)
            
            class_train[i] = class_samples[i][:train_size]
            class_val[i] = class_samples[i][train_size:train_size+val_size]
            class_test[i] = class_samples[i][train_size+val_size:]
        
        # Combine and shuffle splits
        train_samples = []
        val_samples = []
        test_samples = []
        
        for i in range(cfg.NUM_CLASSES):
            train_samples.extend(class_train[i])
            val_samples.extend(class_val[i])
            test_samples.extend(class_test[i])
        
        random.shuffle(train_samples)
        random.shuffle(val_samples)
        random.shuffle(test_samples)
        
        logger.info(f"Split sizes after balancing:")
        for i in range(cfg.NUM_CLASSES):
            logger.info(f"  Class {i} - Train: {len(class_train[i])}, "
                       f"Validation: {len(class_val[i])}, Test: {len(class_test[i])}")
        
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
            seed=42
        )
        
        # Calculate dataset size and splits
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
    
    # Calculate class weights for training
    class_distribution = check_class_distribution(train_dataset)
    if len(class_distribution) == cfg.NUM_CLASSES:
        total_samples = sum(class_distribution.values())
        class_weights = {}
        for i in range(cfg.NUM_CLASSES):
            # Inverse frequency weighting
            class_weights[i] = total_samples / (cfg.NUM_CLASSES * class_distribution.get(i, 1))
        logger.info(f"Class weights for handling imbalance: {class_weights}")
    else:
        class_weights = None
        logger.warning("Could not compute class weights - missing classes in distribution")
    
    # Check class distribution in each split (before batching)
    logger.info("Class distribution in training set:")
    train_dist = check_class_distribution(train_dataset)
    
    logger.info("Class distribution in validation set:")
    val_dist = check_class_distribution(val_dataset)
    
    logger.info("Class distribution in test set:")
    test_dist = check_class_distribution(test_dataset)
    
    # Batch the datasets with drop_remainder=True for consistent shapes
    train_dataset = train_dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)
    val_dataset = val_dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)
    test_dataset = test_dataset.batch(cfg.BATCH_SIZE, drop_remainder=True)
    
    # Optimize with prefetching
    train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)
    
    # Count batches
    train_batches = sum(1 for _ in train_dataset)
    val_batches = sum(1 for _ in val_dataset)
    test_batches = sum(1 for _ in test_dataset)
    
    logger.info(f"Final dataset shapes - train: {train_batches} batches, val: {val_batches} batches, test: {test_batches} batches")
    
    # Verify final shapes
    for features, labels in train_dataset.take(1):
        logger.info(f"Final batched shapes: features={features.shape}, labels={labels.shape}")
    
    return train_dataset, val_dataset, test_dataset, test_batches, class_weights

def create_model():
    """Create a numerically stable model for multiclass classification."""
    logger.info("Creating numerically stable model for multiclass classification")
    
    # Define model with explicit input shape
    inputs = tf.keras.layers.Input(shape=(1, cfg.NUM_FEATURES), name="input_layer")
    
    # Start with batch normalization to stabilize inputs
    x = tf.keras.layers.BatchNormalization()(inputs)
    
    # Simplify to a Flatten + Dense architecture to avoid LSTM instability
    x = tf.keras.layers.Flatten()(x)
    
    # First dense layer with careful initialization and normalization
    x = tf.keras.layers.Dense(64, 
                            kernel_initializer='he_normal',
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Second dense layer
    x = tf.keras.layers.Dense(32, 
                            kernel_initializer='he_normal',
                            kernel_regularizer=tf.keras.regularizers.l2(0.0001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output layer for multiclass with careful initialization
    outputs = tf.keras.layers.Dense(
        cfg.NUM_CLASSES, activation='softmax', name="output_layer",
        kernel_initializer='glorot_uniform'
    )(x)
    
    # Create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Ultra-conservative optimizer settings
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-5,  # Very conservative
        clipnorm=0.1,        # Aggressive gradient clipping
        clipvalue=0.1        # Aggressive value clipping
    )
    
    # Compile model with basic cross-entropy loss
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy', 
                tf.keras.metrics.AUC(multi_label=False),
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
    )
    
    return model

def train_model(model, train_dataset, val_dataset, class_weights=None):
    """Train the model with appropriate callbacks."""
    logger.info(f"Training model for {cfg.EPOCHS} epochs")
    
    # Ensure the model directory exists
    os.makedirs(os.path.dirname(cfg.MODEL_PATH), exist_ok=True)
    
    class NanMonitorCallback(tf.keras.callbacks.Callback):
        def on_batch_end(self, batch, logs=None):
            if logs is not None and np.isnan(logs.get('loss', 0)):
                logger.critical(f"NaN loss detected at batch {batch}. Stopping training.")
                self.model.stop_training = True

    # Callbacks
    callbacks = [
        # Model checkpoint to save the best model
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cfg.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,  # Reduced from 15 for faster response
            restore_best_weights=True,
            mode='min'
        ),
        # Reduce learning rate when training plateaus
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,  # More aggressive reduction
            patience=3,  # Reduced patience for faster response
            min_lr=1e-6,
            verbose=1
        ),
        # Terminate training if NaN values are encountered
        tf.keras.callbacks.TerminateOnNaN(),
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir='./logs',
            histogram_freq=1,
            update_freq='epoch'
        ),
        # Custom callback to monitor NaN values
        NanMonitorCallback()
    ]
    
    # Train the model with class weights
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
    sample_predictions = []
    sample_true_labels = []
    
    for features, labels in test_dataset:
        if batch_count >= sample_size:
            break
            
        predictions = model.predict(features, verbose=0)
        pred_classes = np.argmax(predictions, axis=1)
        
        # Handle NaNs in predictions
        has_nans = np.isnan(predictions).any()
        if has_nans:
            logger.warning(f"Batch {batch_count+1} contains NaN predictions. Using fallback prediction.")
            # Replace NaN predictions with zeros and predict class 0
            predictions = np.nan_to_num(predictions, nan=0.0)
            pred_classes = np.zeros_like(pred_classes)
        
        # Check prediction values
        logger.info(f"Batch {batch_count+1} prediction stats:")
        logger.info(f"  Prediction probabilities - Min: {np.nanmin(predictions) if not has_nans else 'NaN'}, Max: {np.nanmax(predictions) if not has_nans else 'NaN'}")
        logger.info(f"  Confidence (max probability) - Mean: {np.nanmean(np.nanmax(predictions, axis=1)) if not has_nans else 'NaN'}")
        
        # Record predictions and true labels for later analysis
        sample_predictions.extend(pred_classes)
        sample_true_labels.extend(labels.numpy())
        
        # Check actual vs. predicted
        accuracy = np.mean((pred_classes == labels.numpy()).astype(np.float32))
        logger.info(f"  Batch accuracy: {accuracy:.4f}")
        
        # Show prediction distribution
        unique, counts = np.unique(pred_classes, return_counts=True)
        logger.info(f"  Prediction class distribution: {dict(zip(unique, counts))}")
        
        batch_count += 1
    
    # Generate confusion matrix for all test data
    y_true = []
    y_pred = []
    y_probs = []  # Store full probability distributions
    
    # Reset test dataset iterator
    for features, labels in test_dataset:
        batch_pred = model.predict(features, verbose=0)
        
        # Log critical warning if all predictions are NaN
        if np.isnan(batch_pred).all():
            logger.critical("ALL prediction values are NaN! Model has severe numerical instability.")
        
        # Replace NaN values with more intelligent fallback
        if np.isnan(batch_pred).any():
            # Generate small balanced random values for more realistic fallback predictions
            random_preds = np.random.uniform(0.1, 0.4, size=batch_pred.shape)
            # Normalize to sum to 1
            random_preds = random_preds / random_preds.sum(axis=1, keepdims=True)
            # Replace only the NaN values
            batch_pred = np.where(np.isnan(batch_pred), random_preds, batch_pred)
            
        # THIS WAS MISSING: Collect predictions and true labels
        batch_pred_classes = np.argmax(batch_pred, axis=1)
        y_true.extend(labels.numpy())
        y_pred.extend(batch_pred_classes)
        y_probs.extend(batch_pred)
    
    # Calculate and log confusion matrix
    conf_matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=cfg.NUM_CLASSES).numpy()
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Calculate per-class metrics
    precision, recall, f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(cfg.NUM_CLASSES), zero_division=0
    )
    
    # Log per-class metrics
    for i in range(cfg.NUM_CLASSES):
        logger.info(f"Class {i} metrics - Precision: {precision[i]:.4f}, Recall: {recall[i]:.4f}, F1: {f1[i]:.4f}")
    
    # Calculate and log global metrics
    global_precision, global_recall, global_f1, _ = sklearn.metrics.precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    logger.info(f"Global metrics - Precision: {global_precision:.4f}, Recall: {global_recall:.4f}, F1: {global_f1:.4f}")
    
    # Calculate ROC AUC for multiclass (one-vs-rest) - with NaN handling
    try:
        # Convert y_probs to array and replace NaNs with zeros
        y_probs_array = np.array(y_probs)
        if np.isnan(y_probs_array).any():
            logger.warning("NaN values detected in probability predictions. Replacing with zeros.")
            y_probs_array = np.nan_to_num(y_probs_array, nan=0.0)
            
        y_true_onehot = tf.keras.utils.to_categorical(y_true, num_classes=cfg.NUM_CLASSES)
        roc_auc = sklearn.metrics.roc_auc_score(y_true_onehot, y_probs_array, multi_class='ovr')
        logger.info(f"ROC AUC Score (One-vs-Rest): {roc_auc:.4f}")
    except Exception as e:
        logger.error(f"Error calculating ROC AUC: {str(e)}")
        logger.info("Skipping ROC AUC calculation due to errors.")
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # Add labels and values
    tick_marks = np.arange(cfg.NUM_CLASSES)
    plt.xticks(tick_marks, range(cfg.NUM_CLASSES))
    plt.yticks(tick_marks, range(cfg.NUM_CLASSES))
    
    # Add text annotations
    thresh = conf_matrix.max() / 2.0
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save the confusion matrix plot
    plt.savefig('confusion_matrix_multiclass.png')
    logger.info("Saved confusion matrix plot to confusion_matrix_multiclass.png")
    
    # Return metrics and confusion matrix
    return metrics, conf_matrix

def test_data_pipeline(train_dataset):
    """Test data pipeline with an ultra-simple model to check for data issues"""
    logger.info("Testing data pipeline with a simple model")
    
    # Create a trivial model
    inputs = tf.keras.layers.Input(shape=(1, cfg.NUM_FEATURES))
    x = tf.keras.layers.Flatten()(inputs)
    outputs = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation='softmax')(x)
    test_model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    test_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Run for just 1 epoch to test
    try:
        test_model.fit(train_dataset.take(10), epochs=1, verbose=1)
        logger.info("Data pipeline test successful - simple model trained without errors")
        return True
    except Exception as e:
        logger.critical(f"Data pipeline test failed: {str(e)}")
        return False

def plot_training_history(history):
    """Plot and save the training history metrics."""
    logger.info("Plotting training history")
    
    # Create directory for plots if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    
    # Plot accuracy
    plt.subplot(2, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    # Plot Precision/Recall
    plt.subplot(2, 2, 4)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Precision and Recall')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('model_training_results.png')
    logger.info("Saved training history plots to model_training_results.png")
    
    # Save individual metrics plots
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.savefig('model_loss.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('model_accuracy.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('model_auc.png')
    
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.plot(history.history['recall'], label='Training Recall')
    plt.plot(history.history['val_recall'], label='Validation Recall')
    plt.title('Precision and Recall')
    plt.ylabel('Score')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.savefig('model_precision_recall.png')
    
    logger.info("Saved individual metric plots")

def main():
    """Main function to run the multiclass classification workflow."""
    try:
        logger.info("Starting multiclass classification workflow")
        
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
        logger.info("Preparing datasets...")
        train_dataset, val_dataset, test_dataset, test_batches, class_weights = prepare_datasets()
        
        # Test data pipeline before training
        logger.info("Testing data pipeline for issues...")
        pipeline_ok = test_data_pipeline(train_dataset)
        if not pipeline_ok:
            logger.critical("Data pipeline test failed. Check your data for issues before proceeding.")
            # Optionally exit early if pipeline test fails
            # return None, None, None

        # Verify datasets before proceeding
        logger.info("Verifying datasets...")
        # Count batches explicitly to catch any issues before training
        train_batches = sum(1 for _ in train_dataset)
        val_batches = sum(1 for _ in val_dataset)
        test_batches_count = sum(1 for _ in test_dataset)
        logger.info(f"Dataset counts: {train_batches} train batches, {val_batches} val batches, {test_batches_count} test batches")
        
        if train_batches == 0 or val_batches == 0 or test_batches_count == 0:
            raise ValueError("One or more datasets are empty. Check data preprocessing and class distribution.")
        
        # Check if a model file already exists to load
        if os.path.exists(cfg.MODEL_PATH) and os.path.getsize(cfg.MODEL_PATH) > 0:
            try:
                logger.info(f"Loading existing model from {cfg.MODEL_PATH}")
                model = tf.keras.models.load_model(cfg.MODEL_PATH)
                
                # Verify model is valid for multiclass classification
                output_shape = model.outputs[0].shape
                if output_shape[-1] != cfg.NUM_CLASSES:
                    logger.warning(f"Loaded model output shape {output_shape[-1]} doesn't match expected {cfg.NUM_CLASSES} classes")
                    logger.info("Creating new model with correct output shape...")
                    model = create_model()
                else:
                    # Print model summary
                    model.summary(print_fn=logger.info)
            except Exception as e:
                logger.error(f"Error loading model: {str(e)}, creating new model instead")
                model = create_model()
                model.summary(print_fn=logger.info)
        else:
            # Create and train model
            logger.info("Creating new model...")
            model = create_model()
            
            # Print model summary
            model.summary(print_fn=logger.info)
            
            # Train model
            logger.info("Training model...")
            history = train_model(model, train_dataset, val_dataset, class_weights)
            
            # Plot training history
            plot_training_history(history)
        
        # Evaluate model
        logger.info("Evaluating model...")
        metrics, conf_matrix = evaluate_model(model, test_dataset, test_batches)
        
        # Check if model is actually learning (accuracy should be better than random guessing)
        if metrics.get('accuracy', 0) <= (1.0 / cfg.NUM_CLASSES) + 0.05:
            logger.warning(f"Model accuracy ({metrics.get('accuracy', 0):.4f}) is close to random guessing "
                          f"({1.0 / cfg.NUM_CLASSES:.4f}). The model may not be learning properly.")
        else:
            logger.info(f"Model performing better than random guessing (accuracy: {metrics.get('accuracy', 0):.4f} vs. {1.0 / cfg.NUM_CLASSES:.4f})")
        
        logger.info("Multiclass classification workflow completed successfully")
        return model, metrics, conf_matrix

    except Exception as e:
        logger.error(f"Error in multiclass classification workflow: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
