import argparse
import json
import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

from data_loader.data_loader import ICDAR2019DataLoaderFactory
from logger import Logger
from trainer.trainer import Trainer
from utils.bbox import Toolbox

logging.basicConfig(level=logging.DEBUG, format='')


def main(config, resume):
    logger = Logger()

    act = config['data_loader']['activate']
    if act == 0:
        # ICDAR 2019 LSVT
        data_loader = ICDAR2019DataLoaderFactory(config)
        train = data_loader.train()
        val = data_loader.val()
    elif act == 1:
        pass

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(i) for i in config['gpus']])
    model = eval(config['arch'])(config)
    # model.summary()

    loss = eval(config['loss'])(config)
    metrics = [eval(metric) for metric in config['metrics']]

    trainer = Trainer(model, loss, metrics,
                      resume=resume,
                      config=config,
                      data_loader=train,
                      valid_data_loader=val,
                      train_logger=logger,
                      toolbox=Toolbox())

    trainer.train()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-c', '--config', default=None, type=str,
                        help='config file path (default: None)')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='path to latest checkpoint (default: None)')

    args = parser.parse_args()

    config = None
    if args.config is not None:
        config = json.load(open(args.config))
        path = os.path.join(config['trainer']['save_dir'], config['name'])
        # assert not os.path.exists(path), "Path {} already exists!".format(path)
    else:
        if args.resume is not None:
            logger.warning('Warning: --config overridden by --resume')
            config = tf.saved_model.load(args.resume, map_location='cpu')['config']

    assert config is not None

    main(config, args.resume)
