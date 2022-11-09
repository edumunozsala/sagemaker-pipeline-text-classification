import argparse
import os
import shutil
import subprocess
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def parse_args():

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--url', type=str)
    parser.add_argument('--datapath', type=str)
    # data directories
    parser.add_argument('--train', type=str, default='train')
    parser.add_argument('--test', type=str, default='test')

    return parser.parse_known_args()


def download_datafiles(path, url, train_path, test_path):
    #path = "aclImdb_v1"
    #url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    # Load the files
    dataset = tf.keras.utils.get_file(path, url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')
    # Define the dataset directory
    dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')
    # Files in the directories
    print(os.listdir(dataset_dir))
    # Set the train and test directories
    train_dir = os.path.join(dataset_dir, train_path)
    test_dir = os.path.join(dataset_dir, test_path)
    #REmove the unecessary directories
    remove_dir = os.path.join(train_dir, 'unsup')
    shutil.rmtree(remove_dir)
    
    return train_dir, test_dir


def create_datasets(train_dir, test_dir):
    # train 
    batch_size = 32
    seed = 42

    raw_train_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir, 
        batch_size=batch_size, 
        validation_split=0.2, 
        subset='training', 
        seed=seed)
    
    raw_val_ds = tf.keras.utils.text_dataset_from_directory(
        train_dir, 
        batch_size=batch_size, 
        validation_split=0.2, 
        subset='validation', 
        seed=seed)
    
    raw_test_ds = tf.keras.utils.text_dataset_from_directory(
        test_dir, 
        batch_size=batch_size)
    
    return raw_train_ds, raw_val_ds, raw_test_ds


if __name__ == "__main__":
# Install the tensorflow package
    install(f"tensorflow==2.6.2")
    import tensorflow as tf
    
    args, _ = parse_args()

    #url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    #datapath= "aclImdb_v1"
    train_dir, test_dir=download_datafiles(args.datapath, args.url, args.train, args.test)

    print(train_dir)
    print(test_dir)

    raw_train_ds, raw_val_ds, raw_test_ds= create_datasets(train_dir, test_dir)
    
    train_path='/opt/ml/processing/train'
    val_path='/opt/ml/processing/validation'
    test_path='/opt/ml/processing/test'

    tf.data.experimental.save(raw_train_ds, train_path)
    tf.data.experimental.save(raw_val_ds, val_path)
    tf.data.experimental.save(raw_test_ds, test_path)
    
