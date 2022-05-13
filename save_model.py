import tensorflow as tf
from absl import app, flags, logging
from absl.flags import FLAGS
from core.yolov4 import YOLO, decode, filter_boxes
import core.utils as utils
from core.config import cfg
import numpy as np
from core.utils import get_anchors, read_class_names

# flags.DEFINE_string('weights', './data/yolov4.weights', 'path to weights file')
# flags.DEFINE_string('output', './checkpoints/yolov4-416', 'path to output')
# flags.DEFINE_boolean('tiny', False, 'is yolo-tiny or not')
# flags.DEFINE_integer('input_size', 416, 'define input size of export model')
# flags.DEFINE_float('score_thres', 0.2, 'define score threshold')
# flags.DEFINE_string('framework', 'tf', 'define what framework do you want to convert (tf, trt, tflite)')
# flags.DEFINE_string('model', 'yolov4', 'yolov3 or yolov4')

def save_tf(
  weights: str='./data/yolov4.weights',
  output: str='./checkpoints/yolov4-416',
  tiny: bool=False,
  input_size: int=416,
  score_thres: float=0.2,
  framework: str='tf',
  model: str='yolov4'
):
  """_summary_

  Args:
      weights (str, optional): path to weights file. Defaults to './data/yolov4.weights'.
      outputs (str, optional): path to output. Defaults to './checkpoints/yolov4-416'.
      tiny (bool, optional): is yolo-tiny or not. Defaults to False.
      input_size (int, optional): define input size of export model. Defaults to 416.
      score_thres (float, optional): define score threshold. Defaults to 0.2.
      framework (str, optional): define what framework do you want to convert (tf, trt, tflite). Defaults to 'tf'.
      model (str, optional): yolov3 or yolov4. Defaults to 'yolov4'.
  """
  # STRIDES, ANCHORS, NUM_CLASS, XYSCALE = utils.load_config(FLAGS)

  ANCHORS = get_anchors(cfg.YOLO.ANCHORS, tiny)
  XYSCALE = cfg.YOLO.XYSCALE if model == 'yolov4' else [1, 1, 1]
  NUM_CLASS = len(read_class_names(cfg.YOLO.CLASSES))
  STRIDES = np.array(cfg.YOLO.STRIDES)

  input_layer = tf.keras.layers.Input([input_size, input_size, 3])
  feature_maps = YOLO(input_layer, NUM_CLASS, model, tiny)
  bbox_tensors = []
  prob_tensors = []
  if tiny:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
      else:
        output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  else:
    for i, fm in enumerate(feature_maps):
      if i == 0:
        output_tensors = decode(fm, input_size // 8, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
      elif i == 1:
        output_tensors = decode(fm, input_size // 16, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
      else:
        output_tensors = decode(fm, input_size // 32, NUM_CLASS, STRIDES, ANCHORS, i, XYSCALE, framework)
      bbox_tensors.append(output_tensors[0])
      prob_tensors.append(output_tensors[1])
  pred_bbox = tf.concat(bbox_tensors, axis=1)
  pred_prob = tf.concat(prob_tensors, axis=1)
  if framework == 'tflite':
    pred = (pred_bbox, pred_prob)
  else:
    boxes, pred_conf = filter_boxes(pred_bbox, pred_prob, score_threshold=score_thres, input_shape=tf.constant([input_size, input_size]))
    pred = tf.concat([boxes, pred_conf], axis=-1)
  model = tf.keras.Model(input_layer, pred)
  
  utils.load_weights(model, weights, model, tiny)
  model.summary()
  model.save(output)

# def main(_argv):
#   save_tf()

if __name__ == '__main__':
  weights = r"data\visdrone_yolo_v4_original.weights"
  
  save_tf(weights=weights)
  # save_tf()
  #   # try:
    #     app.run(main)
    # except SystemExit:
    #     pass
