import tensorflow as tf

class BinaryFalsePositiveRate(tf.keras.metrics.Metric):
    def __init__(self, name='false_positive_rate', **kwargs):
        super(BinaryFalsePositiveRate, self).__init__(name=name, **kwargs)
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        # Assuming y_true is either 0 or 1 and y_pred is probability/logits
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), self.dtype)
        y_true = tf.cast(y_true, self.dtype)
        fp = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, 1),
                                                    tf.equal(y_true, 0)), self.dtype))
        tn = tf.reduce_sum(tf.cast(tf.logical_and(tf.equal(y_pred, 0),
                                                    tf.equal(y_true, 0)), self.dtype))
        self.false_positives.assign_add(fp)
        self.true_negatives.assign_add(tn)

    def result(self):
        return self.false_positives / (self.false_positives + self.true_negatives + 1e-7)

    def reset_states(self):
        self.false_positives.assign(0.)
        self.true_negatives.assign(0.)