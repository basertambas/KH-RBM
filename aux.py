##################################
## Auxiliary Functions
##################################
def binarize_data(X, th=127):
    # th: threshold
    X[X < th] = 0
    X[X >= th] = 1
    return X 

def get_dataset(dataset='mnist'):
    # 'kmnist', 'mnist'
    import tensorflow_datasets as tfds
    from tensorflow.keras.utils import to_categorical
    import numpy as np

    np.random.seed(42)

    # Load the dataset
    kmnist_train, kmnist_test = tfds.load(dataset, split=['train', 'test'], as_supervised=True)

    # Shuffle and preprocess the training data
    kmnist_train = kmnist_train.shuffle(buffer_size=10000, seed = 42)

    # Extract data and labels for training, validation, and test sets
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for image, label in tfds.as_numpy(kmnist_train):
        train_data.append(image.flatten())
        train_labels.append(label)

    for image, label in tfds.as_numpy(kmnist_test):
        test_data.append(image.flatten())
        test_labels.append(label)

    # Convert lists to numpy arrays
    train_data = np.array(train_data).astype(np.float32)
    train_labels = np.array(train_labels)

    indices = np.arange(len(train_data))
    np.random.shuffle(indices)

    # Use the shuffled indices to split the dataset
    split_index = 50000
    train_indices, val_indices = indices[:split_index], indices[split_index:]

    # Split the dataset
    train_data, valid_data = train_data[train_indices], train_data[val_indices]
    train_labels, valid_labels = train_labels[train_indices], train_labels[val_indices]

    test_data = np.array(test_data).astype(np.float32)
    test_labels = np.array(test_labels)

    train_labels = to_categorical(train_labels)
    valid_labels = to_categorical(valid_labels)
    test_labels = to_categorical(test_labels)

    print("Training Images Shape:", train_data.shape)
    print("Training Labels Shape:", train_labels.shape)
    print("Validation Images Shape:", valid_data.shape)
    print("Validation Labels Shape:", valid_labels.shape)
    print("Test Images Shape:", test_data.shape)
    print("Test Labels Shape:", test_labels.shape)

    return train_data, valid_data, test_data, train_labels, valid_labels, test_labels