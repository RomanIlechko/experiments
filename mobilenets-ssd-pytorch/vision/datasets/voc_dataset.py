import numpy as np
import logging
from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
import os


class VOCDataset:

    def __init__(self, split_dataset, transform=None, target_transform=None, is_test=False, keep_difficult=False,
                 label_file=None):
        # self.root = self.root / "ssd_project/ssd-pytorch-leanh"
        # self.root = './data/VOCdevkit/'
        self.split_dataset = split_dataset
        self.root = split_dataset.path_imgs

        self.transform = transform
        self.target_transform = target_transform

        if is_test:
            # image_sets_file = self.root / "ImageSets/Main/test.txt"

            # image_sets_file = self.root + "test/VOC2007/ImageSets/Main/test.txt"
            image_sets_file = os.path.join(".", "bdd_files", "test.txt")
        else:
            # image_sets_file = self.root / "ImageSets/Main/trainval.txt"

            # image_sets_file = self.root + "VOC2007/ImageSets/Main/trainval.txt"
            image_sets_file = os.path.join(".", "bdd_files", "trainval2.txt")

        self.ids = split_dataset.get_paths()  # VOCDataset._read_image_ids(image_sets_file)
        self.keep_difficult = keep_difficult

        # if the labels file exists, read in the class names
        # label_file_name = self.root + "labels.txt"

        label_file_name = ""

        if os.path.isfile(label_file_name):
            class_string = ""
            with open(label_file_name, 'r') as infile:
                for line in infile:
                    class_string += line.rstrip()

            # classes should be a comma separated list

            classes = class_string.split(',')
            # prepend BACKGROUND as first class
            classes.insert(0, 'BACKGROUND')
            classes = [elem.replace(" ", "") for elem in classes]
            self.class_names = tuple(classes)
            logging.info("VOC Labels read from file: " + str(self.class_names))

        else:
            logging.info("No labels file, using default VOC classes.")
            # self.class_names = ('BACKGROUND',
            # 'aeroplane', 'bicycle', 'bird', 'boat',
            # 'bottle', 'bus', 'car', 'cat', 'chair',
            # 'cow', 'diningtable', 'dog', 'horse',
            # 'motorbike', 'person', 'pottedplant',
            # 'sheep', 'sofa', 'train', 'tvmonitor')

            self.class_names = ('BACKGROUND',
                                'train', 'truck', 'traffic light', 'traffic sign',
                                'rider', 'person', 'bus', 'bike', 'car', 'motor')

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_path, xml_path = self.ids[index]
        assert Path(image_path).stem == Path(xml_path).stem, f"something wrong with {image_path}, {xml_path}"
        boxes, labels, is_difficult = self._get_annotation(xml_path)
        if not self.keep_difficult:
            new_boxes = boxes[is_difficult == 0]
            new_labels = labels[is_difficult == 0]
            if len(new_boxes):
                boxes = new_boxes
                labels = new_labels
        if len(boxes) == 0:
            print(labels, xml_path, image_path)
        image = self._read_image(image_path)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            try:
                boxes, labels = self.target_transform(boxes, labels)
            except Exception as e:
                print(f'{e} {labels, xml_path, image_path}')
        return image, boxes, labels

    def get_image(self, index):
        image_path, xml_path = self.ids[index]
        image = self._read_image(image_path)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_path, xml_path = self.ids[index]
        return Path(xml_path).stem, self._get_annotation(xml_path)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids

    def _get_annotation(self, xml_path):

        # annotation_file = self.root / f"Annotations/{image_id}.xml"

        try:
            # annotation_file = "/home/mju-hpc-01/LATran/MindinTech/ssd_project/bdd100k/bdd100k/xml/" + f"train/{image_id}.xml"
            # annotation_file = os.path.join("..", "bdd100k", "bdd100k", "xml", "train", f"{image_id}.xml")
            # annotation_file = os.path.join(self.root, 'xml', 'train', f"{image_id}.xml")

            objects = ET.parse(xml_path).findall("object")

        # print(1)

        except Exception as e:
            # annotation_file = "/home/mju-hpc-01/LATran/MindinTech/ssd_project/bdd100k/bdd100k/xml/" + f"val/{image_id}.xml"
            # annotation_file = os.path.join("..", "bdd100k", "bdd100k", "xml", "val", f"{image_id}.xml")
            # annotation_file = os.path.join(self.root, 'xml', 'val', f"{image_id}.xml")
            print(e)

            #objects = ET.parse(annotation_file).findall("object")

        # print(2)

        boxes = []
        labels = []
        is_difficult = []
        try:
            some_object_iterator = iter(objects)
        except TypeError as te:
            print(f'is not iterable {objects}')
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_dict:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1

                if x1 == 0.0:
                    x1 = 1.0
                if y1 == 0.0:
                    y1 = 1.0
                if x2 == 0.0:
                    x2 = 1.0
                if y2 == 0.0:
                    y2 = 1.0
                boxes.append([x1, y1, x2, y2])

                labels.append(self.class_dict[class_name])

                try:
                    is_difficult_str = object.find('difficult').text
                except:
                    # is_difficult.append(int(is_difficult_str) if is_difficult_str else 0)
                    is_difficult.append(0)

        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_path):

        # image_file = self.root / f"JPEGImages/{image_id}.jpg"

        try:
            # image_file = "./data/VOCdevkit/VOC2007/" + f"JPEGImages/{image_id}.jpg"
            # image_file = "/home/mju-hpc-01/LATran/MindinTech/ssd_project/bdd100k/bdd100k/images/100k/" + f"train/{image_id}.jpg"
            # image_file = os.path.join("..", "bdd100k", "bdd100k", "images", "100k", "train", f"{image_id}.jpg")
            # image_file = os.path.join(self.root, 'images', 'train', f"{image_id}.jpg")

            image = cv2.imread(image_path)
            if image is None:
                image = cv2.imread(image_path.replace('jpg', 'png'))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            pass
            # image_file = "./data/VOCdevkit/test/VOC2007/" + f"JPEGImages/{image_id}.jpg"
            # image_file = "/home/mju-hpc-01/LATran/MindinTech/ssd_project/bdd100k/bdd100k/images/100k/" + f"val/{image_id}.jpg"
            # image_file = os.path.join("..", "bdd100k", "bdd100k", "images", "100k", "val", f"{image_id}.jpg")
            # image_file = os.path.join(self.root, 'images', 'val', f"{image_id}.jpg")
            #
            # image = cv2.imread(image_file)
            # if image is None:
            #     image = cv2.imread(image_file.replace('jpg', 'png'))
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image
