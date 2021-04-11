#!-- encoding: utf-8 --
import numpy as np
import tensorflow as tf

from utils.bbox import Toolbox
from utils.target_keys import N_CLASS
from .modules import SharedConv, Detector, Recognizer, RoIRotate


class FOTSModel:
    def __init__(self, config):
        self.training = config['training']
        self.sharedConv = SharedConv(config)
        self.detector = Detector(config)
        self.recognizer = Recognizer(N_CLASS, config)
        self.roirotate = RoIRotate()

    def summary(self):
        self.sharedConv.summary()
        self.detector.summary()
        self.recognizer.summary()

    def to(self, device):
        self.sharedConv = self.sharedConv.to(device)
        self.detector = self.detector.to(device)
        self.recognizer = self.recognizer.to(device)

    def train(self):
        self.sharedConv.train()
        self.detector.train()
        self.recognizer.train()

    def eval(self):
        self.sharedConv.eval()
        self.detector.eval()
        self.recognizer.eval()

    def call(self, input):
        image, boxes, mapping = input
        feature_map = self.sharedConv(image)
        score_map, geo_map = self.detector(feature_map)
        if self.training:
            rois, lengths, indices = self.roirotate.call(feature_map, boxes[:, :8], mapping)
            pred_mapping = mapping
            pred_boxes = boxes
        else:
            score = score_map.permute(0, 2, 3, 1)
            geometry = geo_map.permute(0, 2, 3, 1)
            score = score.detach().cpu().numpy()
            geometry = geometry.detach().cpu().numpy()

            timer = {'net': 0, 'restore': 0, 'nms': 0}

            pred_boxes = []
            pred_mapping = []
            for i in range(score.shape[0]):
                s = score[i, :, :, 0]
                g = geometry[i, :, :, ]
                bb, _ = Toolbox.detect(score_map=s, geo_map=g, timer=timer)
                bb_size = bb.shape[0]

                if len(bb) > 0:
                    pred_mapping.append(np.array([i] * bb_size))
                    pred_boxes.append(bb)

            if len(pred_mapping) > 0:
                pred_boxes = np.concatenate(pred_boxes)
                pred_mapping = np.concatenate(pred_mapping)
                rois, lengths, indices = self.roirotate.call(feature_map, pred_boxes[:, :8], pred_mapping)
            else:
                return score_map, geo_map, (None, None), pred_boxes, pred_mapping, None

        lengths = tf.convert_to_tensor(lengths)
        preds = self.recognizer(rois, lengths)
        preds = preds.permute(1, 0, 2)  # B, T, C -> T, B, C

        return score_map, geo_map, (preds, lengths), pred_boxes, pred_mapping, indices
