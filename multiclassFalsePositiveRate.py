import tensorflow as tf

class MulticlassFalsePositiveRate(tf.keras.metrics.Metric):
    def __init__(self, negative_class=0, name='false_positive_rate', **kwargs):
        super(MulticlassFalsePositiveRate, self).__init__(name=name, **kwargs)
        self.neg_class = negative_class
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert one-hot labels to integer labels
        y_true = tf.argmax(y_true, axis=1)
        # For predictions, choose the maximum probability
        y_pred = tf.argmax(y_pred, axis=1)
        # Create binary indicators: 1 if attack, 0 if benign (negative_class)
        true_binary = tf.cast(tf.not_equal(y_true, self.neg_class), self.dtype)
        pred_binary = tf.cast(tf.not_equal(y_pred, self.neg_class), self.dtype)

        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred_binary, 1),tf.equal(true_binary, 0)), self.dtype))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(pred_binary, 0),tf.equal(true_binary, 0)), self.dtype))
        
        self.false_positives.assign_add(fp)
        self.true_negatives.assign_add(tn)

    def result(self):
        return self.false_positives / (self.false_positives + self.true_negatives + 1e-7)

    def reset_states(self):
        self.false_positives.assign(0.)
        self.true_negatives.assign(0.)