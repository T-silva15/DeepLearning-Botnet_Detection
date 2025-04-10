import tensorflow as tf

class MulticlassFalsePositiveRate(tf.keras.metrics.Metric):
    def __init__(self, target_class=1, class_name=None, name='false_positive_rate', **kwargs):
        """
        Calculate False Positive Rate for a specific class in a multiclass setting.
        
        Args:
            target_class: The class to calculate FPR for (when this class is predicted but it's not the true class)
            class_name: Name of the class (for display purposes)
            name: Base name of the metric
        """
        # Use class name in the metric name if provided, otherwise use class index
        if class_name:
            name = f'{name}_{class_name}'
        else:
            name = f'{name}_class_{target_class}'
            
        super(MulticlassFalsePositiveRate, self).__init__(name=name, **kwargs)
        self.target_class = target_class
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot labels to integer labels
        y_true = tf.argmax(y_true, axis=1)
        # For predictions, choose the maximum probability
        y_pred = tf.argmax(y_pred, axis=1)
        
        # For current target class: check predictions vs reality
        # True negative: not the target class in both prediction and truth
        is_target_class_pred = tf.equal(y_pred, self.target_class)
        is_not_target_class_true = tf.not_equal(y_true, self.target_class)
        
        # False positive: predicted as target class but actually not target class
        fp = tf.reduce_sum(
            tf.cast(tf.logical_and(is_target_class_pred, is_not_target_class_true), 
            self.dtype)
        )
        
        # True negative: correctly predicted as not target class when it's not target class
        is_not_target_class_pred = tf.not_equal(y_pred, self.target_class)
        tn = tf.reduce_sum(
            tf.cast(tf.logical_and(is_not_target_class_pred, is_not_target_class_true), 
            self.dtype)
        )
        
        self.false_positives.assign_add(fp)
        self.true_negatives.assign_add(tn)

    def result(self):
        # Calculate FPR: FP / (FP + TN)
        return self.false_positives / (self.false_positives + self.true_negatives + 1e-7)

    def reset_states(self):
        self.false_positives.assign(0.)
        self.true_negatives.assign(0.)


def create_multiclass_fpr_metrics(num_classes, class_names=None):
    """
    Create a list of FPR metrics, one for each class.
    
    Args:
        num_classes: Number of classes in the classification problem
        class_names: List of class names to use in metric names
        
    Returns:
        List of MulticlassFalsePositiveRate metric instances
    """
    if class_names is None or len(class_names) != num_classes:
        # Use default class indices if no names provided or length mismatch
        return [MulticlassFalsePositiveRate(target_class=i) for i in range(num_classes)]
    else:
        # Use provided class names
        return [MulticlassFalsePositiveRate(target_class=i, class_name=name) 
                for i, name in enumerate(class_names)]