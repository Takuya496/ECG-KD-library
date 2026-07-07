"""
TensorFlowモデル定義（②③共通）
"""
import tensorflow as tf


class RegressionHead(tf.keras.Model):
    """KD埋め込み(32次元) or ECG特徴量(19次元) → Arousal/Valence"""
    def __init__(self, hidden_units=64, **kwargs):
        super().__init__(**kwargs)
        self.hidden  = tf.keras.layers.Dense(hidden_units, activation='elu', name='mlp_layer')
        self.out_ar  = tf.keras.layers.Dense(1, name='out_ar')
        self.out_val = tf.keras.layers.Dense(1, name='out_val')

    def call(self, z, training=None):
        h = self.hidden(z)
        return self.out_ar(h), self.out_val(h)
