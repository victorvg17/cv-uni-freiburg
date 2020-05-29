import numpy as np
import pickle
import keras
import tensorflow as tf


def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def extract_labels_filenames(meta_data_dict):
    labels = meta_data_dict[b'labels']
    filenames = meta_data_dict[b'filenames']
    return labels, filenames

def extract_image_batch(meta_data_dict):
    batch_data = []
    imgs = np.array(meta_data_dict[b'data'])
    for i in range(len(imgs)):
        img_k = np.reshape(imgs[i, :], (32, 32, 3))
        batch_data.append(img_k)
    batch_data = np.array(batch_data)
    print(f"batch_data size: {batch_data.shape}")
    return batch_data

def load_cifar_10_data(data_dir, negatives=False):
    """
    Return train_data, train_filenames, train_labels, test_data, test_filenames, test_labels
    """

    # get the meta_data_dict
    # num_cases_per_batch: 1000
    # label_names: ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # num_vis: :3072
    cifar_10_data = {}
    meta_data_dict = unpickle(data_dir + "/batches.meta")
    cifar_label_names = meta_data_dict[b'label_names']
    cifar_label_names = np.array(cifar_label_names)
    # print(f"meta_data_dict: {meta_data_dict}")
    # print(f"cifar_label_names: {cifar_label_names.shape}")

    # training data
    cifar_train_data = np.array([], dtype=np.float64).reshape(0,32, 32, 3)
    cifar_train_filenames = []
    cifar_train_labels = []

    # cifar_train_data_dict
    # 'batch_label': 'training batch 5 of 5'
    # 'data': ndarray
    # 'filenames': list
    # 'labels': list
    for i in range(1, 6):
        meta_data_dict[i] = unpickle(data_dir + "/data_batch_" + str(i))
        labels, filenames = extract_labels_filenames(meta_data_dict[i])
        cifar_train_labels.append(labels)
        cifar_train_filenames.append(filenames)

        batch_image_i = extract_image_batch(meta_data_dict[i])
        cifar_train_data = np.concatenate((batch_image_i, cifar_train_data), axis=0)

    # convert to shape 5000, 1
    cifar_train_labels = np.array(cifar_train_labels)
    cifar_train_labels = cifar_train_labels.reshape((-1))

    # convert to shape 5000, 1
    cifar_train_filenames = np.array(cifar_train_filenames)
    cifar_train_filenames = cifar_train_filenames.reshape((-1))

    cifar_train_data = np.array(cifar_train_data)
    # normalise in range (0, 1)
    cifar_train_data *= 1/cifar_train_data.max()
    print("cifar_train_labels size: {} cifar_train_filenames: {} cifar_train_data: {}"
          .format(cifar_train_labels.shape,
                  cifar_train_filenames.shape,
                 cifar_train_data.shape))

    # load test data
    cifar_test_data = np.array([], dtype=np.float64).reshape(0,32, 32, 3)
    cifar_test_filenames = []
    cifar_test_labels = []
    meta_data_dict_test = unpickle(data_dir + "/test_batch")

    labels, filenames = extract_labels_filenames(meta_data_dict_test)
    cifar_test_labels.append(labels)
    cifar_test_filenames.append(filenames)

    # convert to shape 10000,
    cifar_test_labels = np.array(cifar_test_labels)
    cifar_test_labels = cifar_test_labels.reshape(-1)

    # convert to shape 10000,
    cifar_test_filenames = np.array(cifar_test_filenames)
    cifar_test_filenames = cifar_test_filenames.reshape(-1)

    batch_image_test = extract_image_batch(meta_data_dict_test)
    cifar_test_data = np.concatenate((batch_image_test, cifar_test_data), axis=0)
    cifar_test_data *= 1/cifar_test_data.max()

    print("cifar_test_labels size: {} cifar_test_filenames: {} cifar_test_data: {}"
          .format(cifar_test_labels.shape,
                  cifar_test_filenames.shape,
                 cifar_test_data.shape))
    cifar_10_data['label_names'] = cifar_label_names

    cifar_10_data['train_labels'] = cifar_train_labels
    cifar_10_data['train_file_names'] = cifar_train_filenames
    cifar_10_data['train_data'] = cifar_train_data

    cifar_10_data['test_labels'] = cifar_test_labels
    cifar_10_data['test_file_names'] = cifar_test_filenames
    cifar_10_data['test_data'] = cifar_test_data

    return cifar_10_data

def genr_bin_encodings(num_classes):
    y = np.arange(0, num_classes, 1)
    labels_encoding = tf.keras.utils.to_categorical(y, num_classes = num_classes, dtype = "float32")
    return labels_encoding
