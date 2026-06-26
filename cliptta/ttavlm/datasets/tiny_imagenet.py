# flake8: noqa
# TODO: Refactor this to follow the same logic as in other datasets
from typing import Dict, Union

from PIL import Image
import numpy as np
import os

from collections import defaultdict
from torch import Tensor
from torch.utils.data import Dataset


def _add_channels(img,):
  while len(img.shape) < 3:  # third axis is the channels
    img = np.expand_dims(img, axis=-1)
  while(img.shape[-1]) < 3:
    img = np.concatenate([img, img[:, :, -1:]], axis=-1)
  return img


class TinyImageNetPaths:
  def __init__(self, root_dir):
    train_path = os.path.join(root_dir, 'train')
    val_path = os.path.join(root_dir, 'val')
    test_path = os.path.join(root_dir, 'test')

    wnids_path = os.path.join(root_dir, 'wnids.txt')
    words_path = os.path.join(root_dir, 'words.txt')

    self._make_paths(train_path, val_path, test_path,
                     wnids_path, words_path)

  def _make_paths(self, train_path, val_path, test_path,
                  wnids_path, words_path):
    self.ids = []
    with open(wnids_path, 'r') as idf:
      for nid in idf:
        nid = nid.strip()
        self.ids.append(nid)
    self.nid_to_words = defaultdict(list)
    with open(words_path, 'r') as wf:
      for line in wf:
        nid, labels = line.split('\t')
        labels = list(map(lambda x: x.strip(), labels.split(',')))
        self.nid_to_words[nid].extend(labels)

    self.paths = {
      'train': [],  # [img_path, id, nid, box]
      'val': [],  # [img_path, id, nid, box]
      'test': []  # img_path
    }

    # Get the test paths
    self.paths['test'] = list(map(lambda x: os.path.join(test_path, x),
                                      os.listdir(test_path)))
    # Get the validation paths and labels
    with open(os.path.join(val_path, 'val_annotations.txt')) as valf:
      for line in valf:
        fname, nid, x0, y0, x1, y1 = line.split()
        fname = os.path.join(val_path, 'images', fname)
        bbox = int(x0), int(y0), int(x1), int(y1)
        label_id = self.ids.index(nid)
        self.paths['val'].append((fname, label_id, nid, bbox))

    # Get the training paths
    train_nids = os.listdir(train_path)
    for nid in train_nids:
      anno_path = os.path.join(train_path, nid, nid+'_boxes.txt')
      imgs_path = os.path.join(train_path, nid, 'images')
      label_id = self.ids.index(nid)
      with open(anno_path, 'r') as annof:
        for line in annof:
          fname, x0, y0, x1, y1 = line.split()
          fname = os.path.join(imgs_path, fname)
          bbox = int(x0), int(y0), int(x1), int(y1)
          self.paths['train'].append((fname, label_id, nid, bbox))


class TinyImageNetDataset(Dataset):
  def __init__(self, root_dir, mode='train', transform=None, max_samples=None):
    tinp = TinyImageNetPaths(root_dir)
    self.shift_type = "original"
    self.mode = mode
    self.label_idx = 1  # from [image, id, nid, box]
    self.transform = transform
    self.transform_results = dict()

    self.IMAGE_SHAPE = (64, 64, 3)

    self.img_data = []
    self.label_data = []

    self.max_samples = max_samples
    self.samples = tinp.paths[mode]
    self.samples_num = len(self.samples)
    self.class_names = [", ".join(tinp.nid_to_words[nid]) for nid in tinp.ids]
    self.idx_to_class = {idx: name for idx, name in enumerate(self.class_names)}

    if self.max_samples is not None:
      self.samples_num = min(self.max_samples, self.samples_num)
      self.samples = np.random.permutation(self.samples)[:self.samples_num]

  def __len__(self):
    return self.samples_num

  def __getitem__(self, idx) -> Dict[str, Union[Tensor, str, int]]:
    s = self.samples[idx]
    img = Image.open(s[0])
    img_array = np.array(img)
    if img_array.shape[-1] < 3 or len(img_array.shape) < 3:
      img_array = _add_channels(img_array)
      img = Image.fromarray(img_array)
    lbl = -1 if self.mode == 'test' else int(s[self.label_idx])

    if self.transform:
      sample = self.transform(img)
    return {
      "image": sample,
      "target": lbl,
      "path": s[0],
      "index": idx,
      "name": "" if lbl < 0 else self.idx_to_class[lbl],
    }
