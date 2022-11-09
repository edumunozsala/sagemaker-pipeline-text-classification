import argparse
import os
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses

from model import custom_standardization, create_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max-vocab', type=int, default=20000)
    parser.add_argument('--max-length', type=int, default=250)
    parser.add_argument('--embedding-dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.2)

    # data directories
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    parser.add_argument('--validation', type=str, default=os.environ.get('SM_CHANNEL_VALIDATION'))    
    parser.add_argument('--test', type=str, default=os.environ.get('SM_CHANNEL_TEST'))

    # model directory
    parser.add_argument('--sm-model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))

    return parser.parse_known_args()

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

def apply_vectorization_layer(raw_train_ds, raw_val_ds, raw_test_ds):
    # Make a text-only dataset (without labels), then call adapt
    train_text = raw_train_ds.map(lambda x, y: x)
    vectorize_layer.adapt(train_text)
    
    train_ds = raw_train_ds.map(vectorize_text)
    val_ds = raw_val_ds.map(vectorize_text)
    test_ds = raw_test_ds.map(vectorize_text)
    
    AUTOTUNE = tf.data.AUTOTUNE

    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)    
    
    return train_ds, val_ds, test_ds

if __name__ == "__main__":

    args, _ = parse_args()

    print('Training data location: {}'.format(args.train))
    print('Test data location: {}'.format(args.test))
    raw_train_ds = tf.data.experimental.load(args.train)
    raw_val_ds = tf.data.experimental.load(args.validation)
    raw_test_ds = tf.data.experimental.load(args.test)    
    
    global vectorize_layer
    
    max_features = args.max_vocab
    sequence_length = args.max_length
    # Create a TextVectorization layer
    vectorize_layer = layers.TextVectorization(
        standardize=custom_standardization,
        max_tokens=max_features,
        output_mode='int',
        output_sequence_length=sequence_length)

    train_ds, val_ds, test_ds= apply_vectorization_layer(raw_train_ds, raw_val_ds, raw_test_ds)

    #Create the model
    model = create_model(max_features, args.embedding_dim, args.dropout)
    # Define the loss and optimizer
    model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
                  optimizer='adam',
                  metrics=tf.metrics.BinaryAccuracy(threshold=0.0))    
    # Train the model
    #epochs = args.epochs
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs)    
    
    # Evaluate the model
    loss, accuracy = model.evaluate(test_ds)
    print("Loss: ", loss)
    print("Accuracy: ", accuracy)    
    
    # Define a model for inference
    # This model will accept raw strings as input
    inference_model = tf.keras.Sequential([
        vectorize_layer,
        model,
        layers.Activation('sigmoid')
    ])

    inference_model.compile(
        loss=losses.BinaryCrossentropy(from_logits=False), optimizer="adam", metrics=['accuracy']
    )

    # Check its architecture
    inference_model.summary()
    #Save the model
    model.save(args.sm_model_dir + '/0')
    #Save the inference model
    inference_model.save(args.sm_model_dir + '/1')
    
    #Save the Vectorization_layer
    # Create a temporal model
    vectorize_layer_model = tf.keras.Sequential([
        tf.keras.Input(shape=(1,), dtype=tf.string),
        vectorize_layer
    ])

    vectorize_layer_model.save(args.sm_model_dir + '/2')    
    
    
