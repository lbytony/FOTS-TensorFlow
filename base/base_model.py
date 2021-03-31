import logging

from tensorflow import keras


class BaseModel(keras.Model):
    """
    Base class for all models
    """

    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

    def forward(self, *input):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError
