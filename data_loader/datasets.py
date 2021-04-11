import logging
import pathlib
from itertools import compress

import cv2
import numpy as np
import tensorflow as tf

from .datautils import check_and_validate_polys, crop_area, generate_rbox

logger = logging.getLogger(__name__)


class ICDAR2019Dataset:
    train_images = 'train_full_images'
    label = 'txt'

    def __init__(self, base_dir):
        base_dir = pathlib.Path(base_dir)
        self.imagesRoot = base_dir / self.train_images
        self.labelRoot = base_dir / self.label
        self.images, self.bboxes, self.transcripts = self.loadData()

    def __len__(self):
        return len(self.images)

    def loadData(self):
        all_images = []
        all_bboxes = []
        all_labels = []
        for image in self.imagesRoot.glob('*.jpg'):
            if image.stem == 'gt_1000':
                break
            all_images.append(image)
            gt = self.labelRoot / image.with_name(image.stem).with_suffix('.txt').name
            with gt.open(mode='r', encoding='utf-8') as f:
                bboxes = []
                texts = []
                for line in f:
                    text = line.strip('\ufeff').strip('\xef\xbb\xbf').strip().split(',')
                    x1, y1, x2, y2, x3, y3, x4, y4 = list(map(float, text[:8]))
                    bbox = [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
                    transcript = text[8]
                    bboxes.append(bbox)
                    texts.append(transcript)
                bboxes = np.array(bboxes)
                all_bboxes.append(bboxes)
                all_labels.append(texts)
        return all_images, all_bboxes, all_labels

    def __getitem__(self, index):
        image = self.images[index]
        bbox = self.bboxes[index]  # num_words * 8
        transcript = self.transcripts[index]

        try:
            return self.transform((image, bbox, transcript))
        except Exception as e:
            return self.__getitem__(tf.convert_to_tensor(np.random.randint(0, len(self))))

    def transform(self, gt, input_size=512, random_scale=np.array([0.5, 1, 2.0, 3.0]),
                  background_ratio=3. / 8):
        """
        :param gt: iamge path (str), wordBBoxes (2 * 4 * num_words), transcripts (multiline)
        :return:
        """

        imagePath, wordBBoxes, transcripts = gt
        im = cv2.imread(imagePath.as_posix())
        # wordBBoxes = np.expand_dims(wordBBoxes, axis = 2) if (wordBBoxes.ndim == 2) else wordBBoxes
        # _, _, numOfWords = wordBBoxes.shape
        numOfWords = len(wordBBoxes)
        text_polys = wordBBoxes  # num_words * 4 * 2
        # transcripts = [word for line in transcripts for word in line.split()]
        text_tags = [True if (tag == '*' or tag == '###') else False for tag in transcripts]  # ignore '###'

        if numOfWords == len(transcripts):
            h, w, _ = im.shape
            text_polys, text_tags = check_and_validate_polys(text_polys, text_tags, (h, w))

            rd_scale = np.random.choice(random_scale)
            im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
            text_polys *= rd_scale

            rectangles = []

            # print rd_scale
            # random crop a area from image
            if np.random.rand() < background_ratio:
                # crop background
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=True)
                if text_polys.shape[0] > 0:
                    # cannot find background
                    raise RuntimeError('cannot find background')
                # pad and resize image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = cv2.resize(im_padded, dsize=(input_size, input_size))
                score_map = np.zeros((input_size, input_size), dtype=np.uint8)
                geo_map_channels = 5
                #                     geo_map_channels = 5 if FLAGS.geometry == 'RBOX' else 8
                geo_map = np.zeros((input_size, input_size, geo_map_channels), dtype=np.float32)
                training_mask = np.ones((input_size, input_size), dtype=np.uint8)
            else:
                im, text_polys, text_tags, selected_poly = crop_area(im, text_polys, text_tags, crop_background=False)
                if text_polys.shape[0] == 0:
                    raise RuntimeError('cannot find background')
                h, w, _ = im.shape

                # pad the image to the training input size or the longer side of image
                new_h, new_w, _ = im.shape
                max_h_w_i = np.max([new_h, new_w, input_size])
                im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
                im_padded[:new_h, :new_w, :] = im.copy()
                im = im_padded
                # resize the image to input size
                new_h, new_w, _ = im.shape
                resize_h = input_size
                resize_w = input_size
                im = cv2.resize(im, dsize=(resize_w, resize_h))
                resize_ratio_3_x = resize_w / float(new_w)
                resize_ratio_3_y = resize_h / float(new_h)
                text_polys[:, :, 0] *= resize_ratio_3_x
                text_polys[:, :, 1] *= resize_ratio_3_y
                new_h, new_w, _ = im.shape
                score_map, geo_map, training_mask, rectangles = generate_rbox((new_h, new_w), text_polys, text_tags)

            # predict 出来的feature map 是 128 * 128， 所以 gt 需要取 /4 步长
            images = im[:, :, ::-1].astype(np.float32)  # bgr -> rgb
            score_maps = score_map[::4, ::4, np.newaxis].astype(np.float32)
            geo_maps = geo_map[::4, ::4, :].astype(np.float32)
            training_masks = training_mask[::4, ::4, np.newaxis].astype(np.float32)

            transcripts = [transcripts[i] for i in selected_poly]
            mask = [not (word == '*' or word == '###') for word in transcripts]
            transcripts = list(compress(transcripts, mask))
            rectangles = list(compress(rectangles, mask))  # [ [pt1, pt2, pt3, pt3],  ]

            assert len(transcripts) == len(rectangles)  # make sure length of transcipts equal to length of boxes
            if len(transcripts) == 0:
                raise RuntimeError('No text found.')

            return imagePath, images, score_maps, geo_maps, training_masks, transcripts, rectangles
        else:
            # print(numOfWords, transcripts)
            print(imagePath.stem)
            raise TypeError('Number of bboxes is inconsist with number of transcripts ')


if __name__ == '__main__':
    ICDAR2019Dataset(r"F:\\Code\\HealthHelper\\Dataset\\ICDAR 2019 - LSVT")
