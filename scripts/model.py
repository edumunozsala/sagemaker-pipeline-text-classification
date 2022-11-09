import re
import string
import tensorflow as tf

from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable()
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    
    return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

def create_model(max_features, embedding_dim, drop_out):
    
    model = tf.keras.Sequential([
        layers.Embedding(max_features + 1, embedding_dim),
        layers.Dropout(drop_out), # 0.2
        layers.GlobalAveragePooling1D(),
        layers.Dropout(drop_out),
        layers.Dense(1)])
    
    return model
    