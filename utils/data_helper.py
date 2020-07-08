import os
import errno
import numpy as np
import pickle
import tarfile
import shutil
import sys
import urllib3
from scipy import io

import torchvision
from utils.data_loader import ImageIter, TransformTwice


class Cifar10(object):
    def __init__(self):
        self.root = 'data'
        self.base_folder = 'cifar-10-batches-py'
        self.url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        self.train_list = map(lambda x: 'data_batch_{}'.format(x), range(1, 6))
        self.test_list = ['test_batch']

        self._download()
        self._prepare_train_dataset()
        self._prepare_test_dataset()

    def _download(self):
        if not os.path.exists(self.root):
            try:
                os.makedirs(self.root)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

        file_path = os.path.join(self.root, self.url.split('/')[-1])
        if not os.path.exists(file_path):
            # Download
            print("Downloading {}".format(self.url))
            with urllib3.PoolManager().request('GET', self.url, preload_content=False) as r, \
                    open(file_path, 'wb') as w:
                shutil.copyfileobj(r, w)

            print("Unpacking {}".format(file_path))
            # Unpack data
            tarfile.open(name=file_path, mode="r:gz").extractall(self.root)

        return

    def _prepare_train_dataset(self):
        self.train_data = []
        self.train_labels = []
        for f in self.train_list:
            file = os.path.join(self.root, self.base_folder, f)
            fo = open(file, 'rb')
            if sys.version_info[0] == 2:
                entry = pickle.load(fo)
            else:
                entry = pickle.load(fo, encoding='latin1')
            self.train_data.append(entry['data'])
            if 'labels' in entry:
                self.train_labels += entry['labels']
            else:
                self.train_labels += entry['fine_labels']
            fo.close()

        self.train_data = np.concatenate(self.train_data)
        self.train_data = self.train_data.reshape((50000, 3, 32, 32))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC
        self.train_labels = np.asarray(self.train_labels)

        return

    def _prepare_test_dataset(self):
        f = self.test_list[0]
        file = os.path.join(self.root, self.base_folder, f)
        fo = open(file, 'rb')
        if sys.version_info[0] == 2:
            entry = pickle.load(fo)
        else:
            entry = pickle.load(fo, encoding='latin1')
        self.test_data = entry['data']
        if 'labels' in entry:
            self.test_labels = entry['labels']
        else:
            self.test_labels = entry['fine_labels']
        fo.close()
        self.test_data = self.test_data.reshape((10000, 3, 32, 32))
        self.test_data = self.test_data.transpose((0, 2, 3, 1))  # convert to HWC
        self.test_labels = np.asarray(self.test_labels)

        return

    def get_train_data(self):
        return self.train_data, self.train_labels

    def get_test_data(self):
        return self.test_data, self.test_labels


class SVHN(object):
    def __init__(self):
        self.root = '../data/svhn_dataset'
        self.train_url = "http://ufldl.stanford.edu/housenumbers/train_32x32.mat"
        self.test_url = "http://ufldl.stanford.edu/housenumbers/test_32x32.mat"
        self.train_file_path = os.path.join(self.root, "train_32x32.mat")
        self.test_file_path = os.path.join(self.root, "test_32x32.mat")
        self._download()
        self._prepare_train_dataset()
        self._prepare_test_dataset()

    def _download(self):
        if not os.path.exists(self.root):
            try:
                os.makedirs(self.root)
            except OSError as exception:
                if exception.errno != errno.EEXIST:
                    raise

        train_file_path = self.train_file_path
        if not os.path.exists(train_file_path):
            # Download
            print("Downloading {}".format(self.train_url))
            with urllib3.PoolManager().request('GET', self.train_url, preload_content=False) as r, \
                    open(train_file_path, 'wb') as w:
                shutil.copyfileobj(r, w)

        test_file_path = self.test_file_path
        if not os.path.exists(test_file_path):
            # Download
            print("Downloading {}".format(self.test_url))
            with urllib3.PoolManager().request('GET', self.test_url, preload_content=False) as r, \
                    open(test_file_path, 'wb') as w:
                shutil.copyfileobj(r, w)

        return

    def _prepare_train_dataset(self):
        train_dataset = io.loadmat(self.train_file_path)
        self.train_data = train_dataset['X']
        self.train_labels = train_dataset['y']

        self.train_data = np.transpose(self.train_data, [3, 0, 1, 2])
        self.train_labels = self.train_labels.reshape((-1))
        self.train_labels[self.train_labels==10] = 0

        return

    def _prepare_test_dataset(self):
        test_dataset = io.loadmat(self.test_file_path)
        self.test_data = test_dataset['X']
        self.test_labels = test_dataset['y']

        self.test_data = np.transpose(self.test_data, [3, 0, 1, 2])
        self.test_labels = self.test_labels.reshape((-1))
        self.test_labels[self.test_labels==10] = 0

        return

    def get_train_data(self):
        return self.train_data, self.train_labels

    def get_test_data(self):
        return self.test_data, self.test_labels


def create_train_test_split(train_data_label_percentage, label_random_seed, dataset='cifar10'):
    if dataset == 'cifar10':
        data_provider = Cifar10()
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    elif dataset == 'svhn':
        data_provider = SVHN()
        mean = np.array([0.4377, 0.4438, 0.4728])
        std = np.array([0.1980, 0.2010, 0.1970])
    else:
        raise NotImplementedError('The data loader for this dataset {} has not been supported yet.'.format(dataset))

    if dataset == 'cifar10':
        train_transform = TransformTwice(torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(32, padding=2),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)]
            ))
    elif dataset == 'svhn':
        train_transform = TransformTwice(torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(32, padding=2),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)]
            ))
    else:
        raise NotImplementedError('The transformations for this dataset {} has not been supported yet.'.format(dataset))

    valid_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean, std)]
    )

    train_data, train_labels = data_provider.get_train_data()
    test_data, test_labels = data_provider.get_test_data()

    train_label_data, train_label_labels, train_unlabel_data, train_unlabel_labels = \
        generate_label_unlabel_subsets(train_data, train_labels, label_percentage=train_data_label_percentage,
                                       random_seed=label_random_seed)

    train_label_iter = ImageIter(train_label_data, train_label_labels, transform=train_transform)
    train_unlabel_iter = ImageIter(train_unlabel_data, train_unlabel_labels, transform=train_transform)
    test_iter = ImageIter(test_data, test_labels, transform=valid_transform)

    return train_label_iter, train_unlabel_iter, test_iter


def generate_label_unlabel_subsets(train_data, train_labels, label_percentage=0.02, random_seed=1):
    np.random.seed(random_seed)
    labels = np.unique(train_labels)

    label_samples_indices_list = []
    for label in labels:
        class_indices = np.where(train_labels==label)[0]
        class_num_images = class_indices.size
        class_label_samples_indices = \
            np.random.choice(class_indices, size=int(round(class_num_images * label_percentage)), replace=False)

        label_samples_indices_list.append(class_label_samples_indices)

    label_samples_indices = np.concatenate(label_samples_indices_list)
    label_mask = np.zeros(train_labels.size, dtype=bool)
    for index in label_samples_indices:
        label_mask[index] = True
    unlabel_mask = np.logical_not(label_mask)

    assert np.sum(np.logical_and(label_mask, unlabel_mask)) == 0, \
        'There should be no overlap between label and unlabel masks'
    assert np.sum(np.logical_or(label_mask, unlabel_mask)) == train_labels.size, \
        'The number of label and unlabel samples should be equal to the number of total samples'

    return train_data[label_mask, :], train_labels[label_mask], train_data[unlabel_mask, :], train_labels[unlabel_mask]


def create_train_test_split_by_quantity(train_num_label_examples, label_random_seed, dataset='svhn'):
    if dataset == 'cifar10':
        data_provider = Cifar10()
        mean = np.array([0.4914, 0.4822, 0.4465])
        std = np.array([0.2470, 0.2435, 0.2616])
    elif dataset == 'svhn':
        data_provider = SVHN()
        mean = np.array([0.4377 , 0.4438 , 0.4728])
        std = np.array([0.1980, 0.2010, 0.1970])
    else:
        raise NotImplementedError('The data loader for this dataset {} has not been supported yet.'.format(dataset))

    if dataset == 'cifar10':
        train_transform = TransformTwice(torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(32, padding=2),
             torchvision.transforms.RandomHorizontalFlip(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)]
            ))
    elif dataset == 'svhn':
        train_transform = TransformTwice(torchvision.transforms.Compose(
            [torchvision.transforms.RandomCrop(32, padding=2),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean, std)]
            ))
    else:
        raise NotImplementedError('The transformations for this dataset {} has not been supported yet.'.format(dataset))

    valid_transform = torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(mean, std)]
    )

    train_data, train_labels = data_provider.get_train_data()
    test_data, test_labels = data_provider.get_test_data()

    train_label_data, train_label_labels, train_unlabel_data, train_unlabel_labels = \
        generate_label_unlabel_subsets_by_quantity(train_data, train_labels, 
                                                   num_label_examples=train_num_label_examples,
                                                   random_seed=label_random_seed)

    train_label_iter = ImageIter(train_label_data, train_label_labels, transform=train_transform)
    train_unlabel_iter = ImageIter(train_unlabel_data, train_unlabel_labels, transform=train_transform)
    test_iter = ImageIter(test_data, test_labels, transform=valid_transform)

    return train_label_iter, train_unlabel_iter, test_iter


def generate_label_unlabel_subsets_by_quantity(train_data, train_labels, num_label_examples, random_seed=1):
    np.random.seed(random_seed)
    labels = np.unique(train_labels)
    assert num_label_examples%labels.shape[0] == 0, 'Could not equally divide: {}/{}'.format(num_label_examples, labels.shape[0])
    num_labeled_examples_per_class = int(num_label_examples/labels.shape[0])

    label_samples_indices_list = []
    for label in labels:
        class_indices = np.where(train_labels==label)[0]
        class_label_samples_indices = \
            np.random.choice(class_indices, size=num_labeled_examples_per_class, replace=False)

        label_samples_indices_list.append(class_label_samples_indices)

    label_samples_indices = np.concatenate(label_samples_indices_list)
    label_mask = np.zeros(train_labels.size, dtype=bool)
    for index in label_samples_indices:
        label_mask[index] = True
    unlabel_mask = np.logical_not(label_mask)

    assert np.sum(np.logical_and(label_mask, unlabel_mask)) == 0, \
        'There should be no overlap between label and unlabel masks'
    assert np.sum(np.logical_or(label_mask, unlabel_mask)) == train_labels.size, \
        'The number of label and unlabel samples should be equal to the number of total samples'

    return train_data[label_mask, :], train_labels[label_mask], train_data[unlabel_mask, :], train_labels[unlabel_mask]




