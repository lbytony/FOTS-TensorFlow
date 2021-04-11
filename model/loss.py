import numpy as np
import tensorflow as tf
### 此处默认真实值和预测值的格式均为 bs * W * H * channels
from tensorflow import keras


class DetectionLoss(keras.Model):
    def __init__(self):
        super(DetectionLoss, self).__init__()
        return

    def forward(self, y_true_cls, y_pred_cls,
                y_true_geo, y_pred_geo,
                training_mask):
        classification_loss = self.__dice_coefficient(y_true_cls, y_pred_cls, training_mask)

        # classification_loss = self.__cross_entroy(y_true_cls, y_pred_cls, training_mask)
        # scale classification loss to match the iou loss part
        classification_loss *= 0.01

        # d1 -> top, d2->right, d3->bottom, d4->left
        #     d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(value=y_true_geo, num_or_size_splits=5, axis=3)
        d1_gt, d2_gt, d3_gt, d4_gt, theta_gt = tf.split(y_true_geo, 1, 1)
        #     d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(value=y_pred_geo, num_or_size_splits=5, axis=3)
        d1_pred, d2_pred, d3_pred, d4_pred, theta_pred = tf.split(y_pred_geo, 1, 1)
        area_gt = (d1_gt + d3_gt) * (d2_gt + d4_gt)
        area_pred = (d1_pred + d3_pred) * (d2_pred + d4_pred)
        w_union = np.min(d2_gt, d2_pred) + np.min(d4_gt, d4_pred)
        h_union = np.min(d1_gt, d1_pred) + np.min(d3_gt, d3_pred)
        area_intersect = w_union * h_union
        area_union = area_gt + area_pred - area_intersect
        L_AABB = -np.log((area_intersect + 1.0) / (area_union + 1.0))
        L_theta = 1 - np.cos(theta_pred - theta_gt)
        L_g = L_AABB + 20 * L_theta

        return np.mean(L_g * y_true_cls * training_mask), classification_loss

    @staticmethod
    def __dice_coefficient(y_true_cls, y_pred_cls, training_mask):
        """
        dice loss
        :param y_true_cls:
        :param y_pred_cls:
        :param training_mask:
        :return:
        """
        eps = 1e-5
        intersection = np.sum(y_true_cls * y_pred_cls * training_mask)
        union = np.sum(y_true_cls * training_mask) + np.sum(y_pred_cls * training_mask) + eps
        loss = 1. - (2 * intersection / union)

        return loss

    @staticmethod
    def __cross_entroy(y_true_cls, y_pred_cls, training_mask):
        # import ipdb; ipdb.set_trace()
        return keras.losses.BinaryCrossentropy(y_pred_cls * training_mask, (y_true_cls * training_mask))


class RecognitionLoss(keras.Model):

    def __init__(self):
        super(RecognitionLoss, self).__init__()
        self.ctc_loss = tf.nn.ctc_loss  # labels, pred, labels_len, pred_len

    def call(self, inputs, **kwargs):
        # TODO Waiting to be checked
        gt, pred = inputs[0], inputs[1]
        labels_len, logits_len = gt[0], gt[1]
        labels, logits = pred[0], pred[1]
        loss = self.ctc_loss(labels, logits, labels_len, logits_len)
        return loss


class FOTSLoss(keras.Model):

    def __init__(self, config):
        super(FOTSLoss, self).__init__()
        self.mode = config['model']['mode']
        self.detectionLoss = DetectionLoss()
        self.recogitionLoss = RecognitionLoss()

    def forward(self, y_true_cls, y_pred_cls, y_true_geo, y_pred_geo, y_true_recog, y_pred_recog, training_mask):

        if self.mode == 'recognition':
            recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)
            reg_loss = tf.convert_to_tensor([0.], device=recognition_loss.device)
            cls_loss = tf.convert_to_tensor([0.], device=recognition_loss.device)
        elif self.mode == 'detection':
            reg_loss, cls_loss = self.detectionLoss(y_true_cls, y_pred_cls,
                                                    y_true_geo, y_pred_geo, training_mask)
            recognition_loss = tf.convert_to_tensor([0.], device=reg_loss.device)
        elif self.mode == 'united':
            reg_loss, cls_loss = self.detectionLoss(y_true_cls, y_pred_cls,
                                                    y_true_geo, y_pred_geo, training_mask)
            if y_true_recog:
                recognition_loss = self.recogitionLoss(y_true_recog, y_pred_recog)
                if recognition_loss < 0:
                    import ipdb
                    ipdb.set_trace()

        # recognition_loss = recognition_loss.to(detection_loss.device)
        return reg_loss, cls_loss, recognition_loss
