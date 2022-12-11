
import os
import subprocess
import sys
import numpy as np
import pathlib
import tarfile
import re
import string
import json

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == "__main__":
    
    #Install tensorflow library
    install('tensorflow==2.6.3')
    
    model_path = f"/opt/ml/processing/model/model.tar.gz"
    with tarfile.open(model_path, 'r:gz') as tar:
        tar.extractall('./model')

    # Import libraries
    import tensorflow as tf
    #from model import custom_standardization, create_model    
        
    @tf.keras.utils.register_keras_serializable()
    def custom_standardization(input_data):
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    
        return tf.strings.regex_replace(stripped_html,
                                      '[%s]' % re.escape(string.punctuation),
                                      '')        
    # Load the model from the version 1, an inference model
    model = tf.keras.models.load_model('./model/1')
    
    # Load the model from version 0, without vectorize layer
    #base_model = tf.keras.models.load_model('./model/0')
    #loaded_vectorize_layer_model = tf.keras.models.load_model('./model/2')
    #loaded_vectorize_layer = loaded_vectorize_layer_model.layers[0]    
    
    #model = tf.keras.Sequential([
    #    loaded_vectorize_layer,
    #    base_model,
    #    tf.keras.layers.Activation('sigmoid')
    #])

    #model.compile(
    #    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    #)
    
    # print the model
    model.summary()
    # Load the test dataset
    test_path = "/opt/ml/processing/test/"
    raw_test_ds = tf.data.experimental.load(test_path)
    #x_test = np.load(os.path.join(test_path, 'x_test.npy'))
    #y_test = np.load(os.path.join(test_path, 'y_test.npy'))
    loss, accuracy = model.evaluate(raw_test_ds)
    
    print("\nTest ACC :", print(accuracy))
    
    report_dict = {
        "classification_metrics": {
            "accuracy": {"value": accuracy},
        },
    }

    # Save the result dict to a json file
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(report_dict))