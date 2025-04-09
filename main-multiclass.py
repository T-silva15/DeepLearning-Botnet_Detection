import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
from pathlib import Path
import random
import traceback
from multiclassFalsePositiveRate import MulticlassFalsePositiveRate

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Dataset parameters
    NUM_FEATURES = 39
    NUM_LINES = 500000
    BATCH_SIZE = 32
    DATA_FOLDER = f'proj/datasets/sized_data/multiclass/{NUM_LINES}_lines'
    MODEL_PATH = 'proj/models/best_multiclass_model.keras'
    
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
    
    # Multiclass specific
    NUM_CLASSES = 4  # 3 attack types + 1 benign class

cfg = Config()

def check_class_distribution(dataset):
    """Check the distribution of classes in the dataset"""
    labels = []
    
    # Create a copy of the dataset to avoid using the original
    dataset_copy = dataset.map(lambda x, y: (x, y))
    
    # Unbatch to get individual samples if batched
    if hasattr(dataset_copy, "_batch_size") and dataset_copy._batch_size is not None:
        dataset_copy = dataset_copy.unbatch()
        
    try:
        for _, label in dataset_copy.as_numpy_iterator():
            if isinstance(label, np.ndarray):
                if len(label.shape) > 1 and label.shape[1] > 1:
                    labels.append(int(np.argmax(label)))
                else:
                    labels.append(int(label.item()))
            else:
                labels.append(int(label))
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
    csv_files = [str(f) for f in Path(cfg.DATA_FOLDER).glob("*.csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {cfg.DATA_FOLDER}")
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    try:
        import pandas as pd
        for i, csv_file in enumerate(csv_files[:min(4, len(csv_files))]):
            sample_df = pd.read_csv(csv_file, nrows=10)
            logger.info(f"Sample from {os.path.basename(csv_file)}:")
            logger.info(f"  Columns: {sample_df.columns.tolist()}")
            if 'label' in sample_df.columns:
                logger.info(f"  Labels: {sample_df['label'].values}")
                logger.info(f"  Unique labels: {sample_df['label'].unique()}")
    except ImportError:
        logger.warning("pandas not available, skipping CSV inspection")
    
    dataset = tf.data.experimental.make_csv_dataset(
        csv_files,
        batch_size=cfg.BATCH_SIZE,
        label_name='label',
        num_epochs=1,
        ignore_errors=True
    )
    return dataset

def preprocess_features(features, label):
    """Convert features dictionary to tensor and handle label preprocessing."""
    feature_names = sorted(features.keys())
    features_list = [tf.cast(features[key], tf.float32) for key in feature_names]
    features_tensor = tf.stack(features_list, axis=1)
    features_tensor = tf.where(tf.math.is_nan(features_tensor), tf.zeros_like(features_tensor), features_tensor)
    features_tensor = tf.where(tf.math.is_inf(features_tensor), 1e6 * tf.ones_like(features_tensor), features_tensor)
    features_tensor = tf.clip_by_value(features_tensor, -1e6, 1e6)
    
    # Add time dimension for LSTM
    features_tensor = tf.expand_dims(features_tensor, axis=1)
    features_tensor.set_shape([None, 1, cfg.NUM_FEATURES])
    
    label_tensor = tf.cast(label, tf.int32)
    if len(tf.shape(label_tensor)) > 1 and tf.shape(label_tensor)[1] > 1:
        pass  # Already one-hot encoded
    else:
        label_tensor = tf.one_hot(label_tensor, depth=cfg.NUM_CLASSES)
    
    return features_tensor, label_tensor

def prepare_datasets():
    """Prepare training, validation and test datasets with stratified sampling."""
    raw_dataset = load_dataset()
    preprocessed_dataset = raw_dataset.map(preprocess_features)
    preprocessed_dataset = preprocessed_dataset.unbatch().cache()
    
    # Compute normalization statistics from a sample of the data
    features_sample = []
    for features, _ in preprocessed_dataset.take(5000):
        features_sample.append(features.numpy())
    features_array = np.concatenate(features_sample, axis=0)  # shape (samples, 1, cfg.NUM_FEATURES)
    features_array = features_array.reshape(-1, cfg.NUM_FEATURES)
    mean = np.mean(features_array, axis=0, keepdims=True)
    std = np.std(features_array, axis=0, keepdims=True)
    std = np.maximum(std, 1e-6)
    mean_tensor = tf.constant(mean, dtype=tf.float32)
    std_tensor = tf.constant(std, dtype=tf.float32)
    
    # Normalize the dataset
    normalized_dataset = preprocessed_dataset.map(
        lambda x, y: (tf.clip_by_value((x - mean_tensor) / std_tensor, -cfg.CLIP_VALUE, cfg.CLIP_VALUE), y)
    )
    
    # Stratified splitting: collect samples in numpy arrays
    all_samples = []
    all_labels = []
    for features, labels in normalized_dataset:
        all_samples.append(features.numpy())
        all_labels.append(np.argmax(labels.numpy()))
    X = np.array(all_samples)
    y = np.array(all_labels)
    
    logger.info(f"Total dataset: {len(X)} samples")
    logger.info(f"Class distribution before splitting: {np.bincount(y)}")
    
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, 
        train_size=cfg.TRAIN_SPLIT,
        stratify=y,
        random_state=42
    )
    
    test_size = cfg.TEST_SPLIT / (cfg.VAL_SPLIT + cfg.TEST_SPLIT)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=test_size,
        stratify=y_temp,
        random_state=42
    )
    
    logger.info(f"Train set: {len(X_train)} samples, class dist: {np.bincount(y_train)}")
    logger.info(f"Val set: {len(X_val)} samples, class dist: {np.bincount(y_val)}")
    logger.info(f"Test set: {len(X_test)} samples, class dist: {np.bincount(y_test)}")
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, tf.one_hot(y_train, depth=cfg.NUM_CLASSES)))
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, tf.one_hot(y_val, depth=cfg.NUM_CLASSES)))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, tf.one_hot(y_test, depth=cfg.NUM_CLASSES)))
    
    train_dataset = train_dataset.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(cfg.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    test_batches = len(X_test) // cfg.BATCH_SIZE
    return train_dataset, val_dataset, test_dataset, test_batches, None

def create_model():
    """Create and compile the CNN-LSTM model for multiclass classification."""
    logger.info("Creating CNN-LSTM model for multiclass classification")
    inputs = tf.keras.layers.Input(shape=(1, cfg.NUM_FEATURES), name="input_layer")
    
    # CNN layers
    x = tf.keras.layers.Conv1D(128, 3, padding='same', activation='relu')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(2, padding='same')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # LSTM layers with Recurrent Droupout for regularization
    x = tf.keras.layers.LSTM(64, return_sequences=True, recurrent_dropout=0.2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.LSTM(32, recurrent_dropout=0.2)(x)
    
    # Dense layer with L2 regularization
    x = tf.keras.layers.Dense(16, activation='relu', 
                              kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    
    # Output layer
    outputs = tf.keras.layers.Dense(cfg.NUM_CLASSES, activation='softmax', name="output_layer")(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=cfg.LEARNING_RATE,
        clipnorm=1.0
    )
    
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False, label_smoothing=0.1),
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc', multi_label=True),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.F1Score(
                name='f1_score',
                average='macro',
                threshold=None,
                dtype=tf.float32
            ),
            MulticlassFalsePositiveRate(negative_class=0, name='false_positive_rate')
        ]
    )
    return model

def train_model(model, train_dataset, val_dataset, class_weights=None):
    """Train the model with appropriate callbacks."""
    logger.info(f"Training model for {cfg.EPOCHS} epochs")
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=cfg.MODEL_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            mode='min'
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, 
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TerminateOnNaN(),
        tf.keras.callbacks.TensorBoard(
            log_dir='proj/logs',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]
    
    history = model.fit(
        train_dataset,
        epochs=cfg.EPOCHS,
        validation_data=val_dataset,
        callbacks=callbacks,
        verbose=1
    )
    return history

def evaluate_model(model, test_dataset, test_batches):
    """Evaluate the model on test data with focus on multiclass metrics."""
    logger.info("Evaluating model on test data")
    test_results = model.evaluate(test_dataset, verbose=1)
    metrics = dict(zip(model.metrics_names, test_results))
    
    for metric_name, value in metrics.items():
        logger.info(f"Test {metric_name}: {value:.4f}")
    
    all_true_labels = []
    all_pred_labels = []
    all_pred_probs = []
    for features, labels in test_dataset:
        predictions = model.predict(features, verbose=0)
        pred_classes = tf.argmax(predictions, axis=1).numpy()
        true_classes = tf.argmax(labels, axis=1).numpy()
        all_true_labels.extend(true_classes)
        all_pred_labels.extend(pred_classes)
        all_pred_probs.extend(predictions)
    
    try:
        from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
        unique_classes = np.unique(all_true_labels)
        unique_pred_classes = np.unique(all_pred_labels)
        logger.info(f"Unique classes in true labels: {unique_classes}")
        logger.info(f"Unique classes in predictions: {unique_pred_classes}")
        
        cm = confusion_matrix(all_true_labels, all_pred_labels)
        logger.info(f"Confusion Matrix shape: {cm.shape}")
        logger.info(f"{cm}")
        
        all_possible_classes = np.unique(np.concatenate([unique_classes, unique_pred_classes]))
        class_names = []
        for i in range(len(all_possible_classes)):
            if i == 0:
                class_names.append("Benign")
            if i == 1:
                class_names.append("Mirai GREIP Flood")
            if i == 2:
                class_names.append("Mirai GREETH Flood")
            if i == 3:
                class_names.append("MIRAI UDPPLAIN")
        logger.info(f"Using class names: {class_names}")
        
        report = classification_report(
            all_true_labels, 
            all_pred_labels,
            labels=all_possible_classes,
            target_names=class_names[:len(all_possible_classes)]
        )
        logger.info("Classification Report:")
        logger.info(report)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(all_possible_classes))
        display_class_names = class_names[:len(all_possible_classes)]
        plt.xticks(tick_marks, display_class_names, rotation=45)
        plt.yticks(tick_marks, display_class_names)
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, format(cm[i, j], 'd'),
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig('proj/src/results/multiclass_confusion_matrix.png', dpi=300)
        plt.close()
        
    except Exception as e:
        logger.error(f"Error generating classification report: {str(e)}")
        logger.error(traceback.format_exc())
    
    return metrics

def plot_training_history(history):
    """Plot multiclass training metrics in separate figures and as a combined grid."""
    logger.info("Plotting training history in separate figures (multiclass)")
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title("Multiclass Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('proj/src/results/multiclass_model_loss.png', dpi=300)
    plt.close()
    
    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title("Multiclass Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('proj/src/results/multiclass_model_accuracy.png', dpi=300)
    plt.close()
    
    # Plot AUC (if available)
    if 'auc' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['auc'], label='Training AUC')
        plt.plot(history.history['val_auc'], label='Validation AUC')
        plt.title("Multiclass AUC")
        plt.xlabel("Epoch")
        plt.ylabel("AUC")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj/src/results/multiclass_model_auc.png', dpi=300)
        plt.close()
    
    # Plot F1 Score (if available)
    if 'f1_score' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['f1_score'], label='Training F1 Score')
        plt.plot(history.history['val_f1_score'], label='Validation F1 Score')
        plt.title("Multiclass F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("F1 Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj/src/results/multiclass_model_f1_score.png', dpi=300)
        plt.close()
    
    # Plot Precision and Recall (if available)
    if 'precision' in history.history and 'recall' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['precision'], label="Training Precision")
        plt.plot(history.history['val_precision'], label="Validation Precision")
        plt.plot(history.history['recall'], label="Training Recall")
        plt.plot(history.history['val_recall'], label="Validation Recall")
        plt.title("Multiclass Precision and Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj/src/results/multiclass_model_precision_recall.png', dpi=300)
        plt.close()
    
    # Plot False Positive Rate (if available)
    if 'false_positive_rate' in history.history:
        plt.figure(figsize=(8, 6))
        plt.plot(history.history['false_positive_rate'], label="Training FPR")
        plt.plot(history.history['val_false_positive_rate'], label="Validation FPR")
        plt.title("Multiclass False Positive Rate")
        plt.xlabel("Epoch")
        plt.ylabel("FPR")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig('proj/src/results/multiclass_model_fpr.png', dpi=300)
        plt.close()
    
    # Grid Figure (2 x 3)
    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    
    # Loss graph (top-left)
    axs[0, 0].plot(history.history['loss'], label='Train Loss')
    axs[0, 0].plot(history.history['val_loss'], label='Val Loss')
    axs[0, 0].set_title("Multiclass Loss")
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Accuracy graph (top-middle)
    axs[0, 1].plot(history.history['accuracy'], label="Train Accuracy")
    axs[0, 1].plot(history.history['val_accuracy'], label="Val Accuracy")
    axs[0, 1].set_title("Multiclass Accuracy")
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # AUC graph (top-right)
    if 'auc' in history.history:
        axs[0, 2].plot(history.history['auc'], label="Train AUC")
        axs[0, 2].plot(history.history['val_auc'], label="Val AUC")
        axs[0, 2].set_title("Multiclass AUC")
        axs[0, 2].legend()
        axs[0, 2].grid(True)
    else:
        axs[0, 2].set_visible(False)
        
    # F1 Score graph (bottom-left)
    if 'f1_score' in history.history:
        axs[1, 0].plot(history.history['f1_score'], label="Train F1")
        axs[1, 0].plot(history.history['val_f1_score'], label="Val F1")
        axs[1, 0].set_title("Multiclass F1 Score")
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
        axs[1, 1].set_title("Multiclass Precision & Recall")
        axs[1, 1].legend()
        axs[1, 1].grid(True)
    else:
        axs[1, 1].set_visible(False)
        
    # False Positive Rate graph (bottom-right)
    if 'false_positive_rate' in history.history:
        axs[1, 2].plot(history.history['false_positive_rate'], label="Train FPR")
        axs[1, 2].plot(history.history['val_false_positive_rate'], label="Val FPR")
        axs[1, 2].set_title("Multiclass False Positive Rate")
        axs[1, 2].legend()
        axs[1, 2].grid(True)
    else:
        axs[1, 2].set_visible(False)
    
    plt.tight_layout()
    plt.savefig("proj/src/results/multiclass_model_training_results_combined.png", dpi=300)
    plt.close()

def main():
    """Main function to run the entire training pipeline."""
    try:
        logger.info("Starting multiclass classification training pipeline")
        tf.random.set_seed(42)
        np.random.seed(42)
        random.seed(42)
        
        train_dataset, val_dataset, test_dataset, test_batches, _ = prepare_datasets()
        model = create_model()
        model.summary()
        history = train_model(model, train_dataset, val_dataset)
        metrics = evaluate_model(model, test_dataset, test_batches)
        plot_training_history(history)
        
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
        
        logger.info(f"Test set results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        logger.info("==========================================")
        
        logger.info("Multiclass training pipeline completed successfully")
        return model, history, metrics
        
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}")
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()