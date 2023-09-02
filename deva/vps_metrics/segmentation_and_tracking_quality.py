# coding=utf-8
# Copyright 2021 The Deeplab2 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Numpy Implementation of the Segmentation and Tracking Quality (STQ) metric.

This implementation is designed to work stand-alone. Please feel free to copy
this file and the corresponding unit-test to your project.
"""

import collections
from typing import Mapping, MutableMapping, Sequence, Text, Any
import numpy as np

_EPSILON = 1e-15


def _update_dict_stats(stat_dict: MutableMapping[int, np.ndarray],
                       id_array: np.ndarray):
  """Updates a given dict with corresponding counts."""
  ids, counts = np.unique(id_array, return_counts=True)
  for idx, count in zip(ids, counts):
    if idx in stat_dict:
      stat_dict[idx] += count
    else:
      stat_dict[idx] = count


class STQuality(object):
  """Metric class for the Segmentation and Tracking Quality (STQ).

  Please see the following paper for more details about the metric:

  "STEP: Segmenting and Tracking Every Pixel", Weber et al., arXiv:2102.11859,
  2021.


  The metric computes the geometric mean of two terms.
  - Association Quality: This term measures the quality of the track ID
      assignment for `thing` classes. It is formulated as a weighted IoU
      measure.
  - Segmentation Quality: This term measures the semantic segmentation quality.
      The standard class IoU measure is used for this.

  Example usage:

  stq_obj = segmentation_tracking_quality.STQuality(num_classes, things_list,
    ignore_label, label_bit_shift, offset)
  stq_obj.update_state(y_true_1, y_pred_1)
  stq_obj.update_state(y_true_2, y_pred_2)
  ...
  result = stq_obj.result()
  """

  def __init__(self, num_classes: int, things_list: Sequence[int],
               ignore_label: int, label_bit_shift: int, offset: int):
    """Initialization of the STQ metric.

    Args:
      num_classes: Number of classes in the dataset as an integer.
      things_list: A sequence of class ids that belong to `things`.
      ignore_label: The class id to be ignored in evaluation as an integer or
        integer tensor.
      label_bit_shift: The number of bits the class label is shifted as an
        integer -> (class_label << bits) + trackingID
      offset: The maximum number of unique labels as an integer or integer
        tensor.
    """
    self._num_classes = num_classes
    self._ignore_label = ignore_label
    self._things_list = things_list
    self._label_bit_shift = label_bit_shift
    self._bit_mask = (2**label_bit_shift) - 1

    if ignore_label >= num_classes:
      self._confusion_matrix_size = num_classes + 1
      self._include_indices = np.arange(self._num_classes)
    else:
      self._confusion_matrix_size = num_classes
      self._include_indices = np.array(
          [i for i in range(num_classes) if i != self._ignore_label])

    self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
    self._predictions = collections.OrderedDict()
    self._ground_truth = collections.OrderedDict()
    self._intersections = collections.OrderedDict()
    self._sequence_length = collections.OrderedDict()
    self._offset = offset
    lower_bound = num_classes << self._label_bit_shift
    if offset < lower_bound:
      raise ValueError('The provided offset %d is too small. No guarantess '
                       'about the correctness of the results can be made. '
                       'Please choose an offset that is higher than num_classes'
                       ' * max_instances_per_category = %d' % lower_bound)

  def get_semantic(self, y: np.ndarray) -> np.ndarray:
    """Returns the semantic class from a panoptic label map."""
    return y >> self._label_bit_shift

  def update_state(self, y_true: np.ndarray, y_pred: np.ndarray, sequence_id=0):
    """Accumulates the segmentation and tracking quality statistics.

    IMPORTANT: When encoding the parameters y_true and y_pred, please be aware
    that the `+` operator binds higher than the label shift `<<` operator.

    Args:
      y_true: The ground-truth panoptic label map for a particular video frame
        (defined as (semantic_map << label_bit_shift) + instance_map).
      y_pred: The predicted panoptic label map for a particular video frame
        (defined as (semantic_map << label_bit_shift) + instance_map).
      sequence_id: The optional ID of the sequence the frames belong to. When no
        sequence is given, all frames are considered to belong to the same
        sequence (default: 0).
    """
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)

    semantic_label = self.get_semantic(y_true)
    semantic_prediction = self.get_semantic(y_pred)
    # Check if the ignore value is outside the range [0, num_classes]. If yes,
    # map `_ignore_label` to `_num_classes`, so it can be used to create the
    # confusion matrix.
    if self._ignore_label > self._num_classes:
      semantic_label = np.where(semantic_label != self._ignore_label,
                                semantic_label, self._num_classes)
      semantic_prediction = np.where(semantic_prediction != self._ignore_label,
                                     semantic_prediction, self._num_classes)
    if sequence_id in self._iou_confusion_matrix_per_sequence:
      idxs = (np.reshape(semantic_label, [-1]) <<
              self._label_bit_shift) + np.reshape(semantic_prediction, [-1])
      unique_idxs, counts = np.unique(idxs, return_counts=True)
      self._iou_confusion_matrix_per_sequence[sequence_id][
          unique_idxs >> self._label_bit_shift,
          unique_idxs & self._bit_mask] += counts
      self._sequence_length[sequence_id] += 1
    else:
      self._iou_confusion_matrix_per_sequence[sequence_id] = np.zeros(
          (self._confusion_matrix_size, self._confusion_matrix_size),
          dtype=np.int64)
      idxs = np.stack([
          np.reshape(semantic_label, [-1]),
          np.reshape(semantic_prediction, [-1])
      ],
                      axis=0)
      np.add.at(self._iou_confusion_matrix_per_sequence[sequence_id],
                tuple(idxs), 1)

      self._predictions[sequence_id] = {}
      self._ground_truth[sequence_id] = {}
      self._intersections[sequence_id] = {}
      self._sequence_length[sequence_id] = 1

    instance_label = y_true & self._bit_mask  # 0xFFFF == 2 ^ 16 - 1

    label_mask = np.zeros_like(semantic_label, dtype=bool)
    prediction_mask = np.zeros_like(semantic_prediction, dtype=bool)
    for things_class_id in self._things_list:
      label_mask = np.logical_or(label_mask, semantic_label == things_class_id)
      prediction_mask = np.logical_or(prediction_mask,
                                      semantic_prediction == things_class_id)

    # Select the `crowd` region of the current class. This region is encoded
    # instance id `0`.
    is_crowd = np.logical_and(instance_label == 0, label_mask)
    # Select the non-crowd region of the corresponding class as the `crowd`
    # region is ignored for the tracking term.
    label_mask = np.logical_and(label_mask, np.logical_not(is_crowd))
    # Do not punish id assignment for regions that are annotated as `crowd` in
    # the ground-truth.
    prediction_mask = np.logical_and(prediction_mask, np.logical_not(is_crowd))

    seq_preds = self._predictions[sequence_id]
    seq_gts = self._ground_truth[sequence_id]
    seq_intersects = self._intersections[sequence_id]

    # Compute and update areas of ground-truth, predictions and intersections.
    _update_dict_stats(seq_preds, y_pred[prediction_mask])
    _update_dict_stats(seq_gts, y_true[label_mask])

    non_crowd_intersection = np.logical_and(label_mask, prediction_mask)
    intersection_ids = (
        y_true[non_crowd_intersection] * self._offset +
        y_pred[non_crowd_intersection])
    _update_dict_stats(seq_intersects, intersection_ids)

  def result(self) -> Mapping[Text, Any]:
    """Computes the segmentation and tracking quality.

    Returns:
      A dictionary containing:
        - 'STQ': The total STQ score.
        - 'AQ': The total association quality (AQ) score.
        - 'IoU': The total mean IoU.
        - 'STQ_per_seq': A list of the STQ score per sequence.
        - 'AQ_per_seq': A list of the AQ score per sequence.
        - 'IoU_per_seq': A list of mean IoU per sequence.
        - 'Id_per_seq': A list of string-type sequence Ids to map list index to
            sequence.
        - 'Length_per_seq': A list of the length of each sequence.
    """
    # Compute association quality (AQ)
    num_tubes_per_seq = [0] * len(self._ground_truth)
    aq_per_seq = [0] * len(self._ground_truth)
    iou_per_seq = [0] * len(self._ground_truth)
    id_per_seq = [''] * len(self._ground_truth)

    for index, sequence_id in enumerate(self._ground_truth):
      outer_sum = 0.0
      predictions = self._predictions[sequence_id]
      ground_truth = self._ground_truth[sequence_id]
      intersections = self._intersections[sequence_id]
      num_tubes_per_seq[index] = len(ground_truth)
      id_per_seq[index] = sequence_id

      for gt_id, gt_size in ground_truth.items():
        inner_sum = 0.0
        for pr_id, pr_size in predictions.items():
          tpa_key = self._offset * gt_id + pr_id
          if tpa_key in intersections:
            tpa = intersections[tpa_key]
            fpa = pr_size - tpa
            fna = gt_size - tpa
            inner_sum += tpa * (tpa / (tpa + fpa + fna))

        outer_sum += 1.0 / gt_size * inner_sum
      aq_per_seq[index] = outer_sum

    aq_mean = np.sum(aq_per_seq) / np.maximum(
        np.sum(num_tubes_per_seq), _EPSILON)
    aq_per_seq = aq_per_seq / np.maximum(num_tubes_per_seq, _EPSILON)

    # Compute IoU scores.
    # The rows correspond to ground-truth and the columns to predictions.
    # Remove fp from confusion matrix for the void/ignore class.
    total_confusion = np.zeros(
        (self._confusion_matrix_size, self._confusion_matrix_size),
        dtype=np.int64)
    for index, confusion in enumerate(
        self._iou_confusion_matrix_per_sequence.values()):
      removal_matrix = np.zeros_like(confusion)
      removal_matrix[self._include_indices, :] = 1.0
      confusion *= removal_matrix
      total_confusion += confusion

      # `intersections` corresponds to true positives.
      intersections = confusion.diagonal()
      fps = confusion.sum(axis=0) - intersections
      fns = confusion.sum(axis=1) - intersections
      unions = intersections + fps + fns

      num_classes = np.count_nonzero(unions)
      ious = (
          intersections.astype(np.double) /
          np.maximum(unions, 1e-15).astype(np.double))
      iou_per_seq[index] = np.sum(ious) / num_classes

    # `intersections` corresponds to true positives.
    intersections = total_confusion.diagonal()
    fps = total_confusion.sum(axis=0) - intersections
    fns = total_confusion.sum(axis=1) - intersections
    unions = intersections + fps + fns

    num_classes = np.count_nonzero(unions)
    ious = (
        intersections.astype(np.double) /
        np.maximum(unions, _EPSILON).astype(np.double))
    iou_mean = np.sum(ious) / num_classes

    st_quality = np.sqrt(aq_mean * iou_mean)
    st_quality_per_seq = np.sqrt(aq_per_seq * iou_per_seq)
    return {
        'STQ': st_quality,
        'AQ': aq_mean,
        'IoU': float(iou_mean),
        'STQ_per_seq': st_quality_per_seq,
        'AQ_per_seq': aq_per_seq,
        'IoU_per_seq': iou_per_seq,
        'ID_per_seq': id_per_seq,
        'Length_per_seq': list(self._sequence_length.values()),
    }

  def reset_states(self):
    """Resets all states that accumulated data."""
    self._iou_confusion_matrix_per_sequence = collections.OrderedDict()
    self._predictions = collections.OrderedDict()
    self._ground_truth = collections.OrderedDict()
    self._intersections = collections.OrderedDict()
    self._sequence_length = collections.OrderedDict()
