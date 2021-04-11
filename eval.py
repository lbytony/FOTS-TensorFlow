import argparse
import logging
import pathlib
import traceback

import tensorflow as tf

from model.model import FOTSModel
from utils.bbox import Toolbox

logging.basicConfig(level=logging.DEBUG, format='')


def load_model(model_path, with_gpu):
    logger.info("Loading checkpoint: {} ...".format(model_path))
    checkpoints = tf.saved_model.load(model_path, map_location='cpu')
    if not checkpoints:
        raise RuntimeError('No checkpoint found.')
    config = checkpoints['config']
    state_dict = checkpoints['state_dict']
    model = FOTSModel(config)
    # if with_gpu:
    #     model.parallelize()
    # model.load_state_dict(state_dict)
    if with_gpu:
        model.to(tf.device('cuda'))
    model.eval()
    return model


def main(args: argparse.Namespace):
    model_path = args.model
    input_dir = args.input_dir
    output_dir = args.output_dir
    with_image = True if output_dir else False
    with_gpu = True if torch.cuda.is_available() else False
    # with_gpu = False

    model = load_model(model_path, with_gpu)

    for image_fn in input_dir.glob('*.jpg'):
        try:
            with torch.no_grad():
                ploy, im = Toolbox.predict(image_fn, model, with_image, output_dir, with_gpu)
                print(len(ploy))
        except Exception as e:
            traceback.print_exc()


if __name__ == '__main__':
    logger = logging.getLogger()

    parser = argparse.ArgumentParser(description='Model eval')
    parser.add_argument('-m', '--model', default=None, type=pathlib.Path, required=True, help='path to model')
    parser.add_argument('-o', '--output_dir', default=None, type=pathlib.Path, help='output dir for drawn images')
    parser.add_argument('-i', '--input_dir', default=None, type=pathlib.Path, required=False,
                        help='dir for input images')
    args = parser.parse_args()
    main(args)
