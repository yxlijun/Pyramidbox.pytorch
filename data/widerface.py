#-*- coding:utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
from PIL import Image, ImageDraw
import torch.utils.data as data
import numpy as np
import random
from utils.augmentations import preprocess


class WIDERDetection(data.Dataset):
    """docstring for WIDERDetection"""

    def __init__(self, list_file, mode='train'):
        super(WIDERDetection, self).__init__()
        self.mode = mode
        self.fnames = []
        self.boxes = []
        self.labels = []

        with open(list_file) as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip().split()
            num_faces = int(line[1])
            box = []
            label = []
            for i in xrange(num_faces):
                x = float(line[2 + 5 * i])
                y = float(line[3 + 5 * i])
                w = float(line[4 + 5 * i])
                h = float(line[5 + 5 * i])
                c = int(line[6 + 5 * i])
                if w <= 0 or h <= 0:
                    continue
                box.append([x, y, x + w, y + h])
                label.append(c)
            if len(box) > 0:
                self.fnames.append(line[0])
                self.boxes.append(box)
                self.labels.append(label)

        self.num_samples = len(self.boxes)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        img, face_target,head_target = self.pull_item(index)
        return img, face_target,head_target

    def pull_item(self, index):
        while True:
            image_path = self.fnames[index]
            img = Image.open(image_path)
            if img.mode == 'L':
                img = img.convert('RGB')

            im_width, im_height = img.size
            boxes = self.annotransform(
                np.array(self.boxes[index]), im_width, im_height)
            label = np.array(self.labels[index])
            bbox_labels = np.hstack((label[:, np.newaxis], boxes)).tolist()
            img, sample_labels = preprocess(
                img, bbox_labels, self.mode, image_path)
            sample_labels = np.array(sample_labels)
            if len(sample_labels) > 0:
                face_target = np.hstack(
                    (sample_labels[:, 1:], sample_labels[:, 0][:, np.newaxis]))

                assert (face_target[:, 2] > face_target[:, 0]).any()
                assert (face_target[:, 3] > face_target[:, 1]).any()

                #img = img.astype(np.float32)
                face_box = face_target[:, :-1]
                head_box = self.expand_bboxes(face_box)
                head_target = np.hstack((head_box, face_target[
                                        :, -1][:, np.newaxis]))
                break
            else:
                index = random.randrange(0, self.num_samples)

        
        #img = Image.fromarray(img)
        '''
        draw = ImageDraw.Draw(img)
        w,h = img.size
        for bbox in sample_labels:
            bbox = (bbox[1:] * np.array([w, h, w, h])).tolist()

            draw.rectangle(bbox,outline='red')
        img.save('image.jpg')
        '''
        return torch.from_numpy(img), face_target, head_target
        

    def annotransform(self, boxes, im_width, im_height):
        boxes[:, 0] /= im_width
        boxes[:, 1] /= im_height
        boxes[:, 2] /= im_width
        boxes[:, 3] /= im_height
        return boxes

    def expand_bboxes(self,
                      bboxes,
                      expand_left=2.,
                      expand_up=2.,
                      expand_right=2.,
                      expand_down=2.):
        expand_bboxes = []
        for bbox in bboxes:
            xmin = bbox[0]
            ymin = bbox[1]
            xmax = bbox[2]
            ymax = bbox[3]
            w = xmax - xmin
            h = ymax - ymin
            ex_xmin = max(xmin - w / expand_left, 0.)
            ex_ymin = max(ymin - h / expand_up, 0.)
            ex_xmax = max(xmax + w / expand_right, 0.)
            ex_ymax = max(ymax + h / expand_down, 0.)
            expand_bboxes.append([ex_xmin, ex_ymin, ex_xmax, ex_ymax])
        expand_bboxes = np.array(expand_bboxes)
        return expand_bboxes

def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on
                                 0 dim
    """
    face_targets = []
    head_targets = []

    imgs = []
    for sample in batch:
        imgs.append(sample[0])
        face_targets.append(torch.FloatTensor(sample[1]))
        head_targets.append(torch.FloatTensor(sample[2]))
    return torch.stack(imgs, 0), face_targets,head_targets

    

if __name__ == '__main__':
    from config import cfg
    dataset = WIDERDetection(cfg.FACE.TRAIN_FILE)
    #for i in range(len(dataset)):
    dataset.pull_item(14)
