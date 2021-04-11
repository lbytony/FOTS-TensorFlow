import tensorflow as tf
from sklearn import model_selection

from base import BaseDataLoader
from .datasets import ICDAR2019Dataset


class ICDAR2019DataLoaderFactory(BaseDataLoader):

    def __init__(self, config):
        super(ICDAR2019DataLoaderFactory, self).__init__(config)
        dataRoot = config['data_loader']['datasets'][self.activate]['data_dir']
        self.workers = config['data_loader']['workers']
        self.have_test = config['data_loader']['datasets'][self.activate]['have_test']
        dataset = ICDAR2019Dataset(dataRoot)
        self.allDataset = dataset.loadData()

        if self.have_test:
            self.trainDataset, self.testDataset = self.train_val_split(self.allDataset)
            self.trainDataset, self.valDataset = self.train_val_split(self.trainDataset)
        else:
            self.trainDataset, self.valDataset = self.train_val_split(self.allDataset)

    def train(self):
        trainLoader = tf.data.Dataset.from_tensor_slices(self.trainDataset)
        # trainLoader = torchdata.DataLoader(self.trainDataset, num_workers=self.num_workers,
        #                                    batch_size=self.batch_size,
        #                                    shuffle=self.shuffle, collate_fn=collate_fn)
        return trainLoader

    def val(self):
        # valLoader = torchdata.DataLoader(self.valDataset, num_workers=self.num_workers, batch_size=self.batch_size,
        #                                  shuffle=shuffle, collate_fn=collate_fn)
        valLoader = tf.data.Dataset.from_tensor_slices(self.valDataset)
        return valLoader

    def train_val_split(self, dataset):
        """

        :param dataset: dataset
        :return:
        """
        train, val = model_selection.train_test_split(dataset[0], tuple(dataset[1:]), test_size=self.val_rate)
        return train, val

    def train_test_split(self, dataset):
        train, test = model_selection.train_test_split(dataset[0], dataset[1:], test_size=self.test_rate)
        return train, test

    def split_validation(self):
        raise NotImplementedError
