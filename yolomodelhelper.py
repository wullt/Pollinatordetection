import torch
import time
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import logging


class YoloModel:
    def __init__(
        self,
        model_path,
        yolov5_path=None,
        confidence_threshold=0.25,
        iou_threshold=0.45,
        margin=40,
        multi_label=False,
        multi_label_iou_threshold=0.5,
        class_names=None,
        augment=False,
        amp=False,
        agnostic=False,
        max_det=10,
    ):
        if yolov5_path is None:
            self.model = torch.hub.load("ultralytics/yolov5", "custom", model_path)
        else:
            self.model = torch.hub.load(
                yolov5_path, "custom", model_path, source="local"
            )
        self.model_name = model_path.split("/")[-1]
        self.model.conf = confidence_threshold
        self.model.iou = iou_threshold
        self.model.agnostic = agnostic  # NMS class-agnostic
        self.model.multi_label = multi_label  # NMS multiple labels per box
        self.model.max_det = max_det  # maximum number of detections per image
        self.model.amp = amp  # Automatic Mixed Precision (AMP) inference
        self.model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
        self.margin = margin
        self.augment = augment
        self.class_names = class_names
        self.multi_label_iou_threshold = multi_label_iou_threshold
        self.results = None
        self.total_inference_time = 0
        self.number_of_inferences = 0

    def get_metadata(self):
        metadata = {}
        metadata["confidence_threshold"] = self.model.conf
        metadata["iou_threshold"] = self.model.iou
        metadata["margin"] = self.margin
        metadata["multi_label"] = self.model.multi_label
        metadata["multi_label_iou_threshold"] = self.multi_label_iou_threshold
        metadata["model_name"] = self.model_name
        metadata["max_det"] = self.model.max_det
        metadata["augment"] = self.augment
        total_inference_time, average_inference_time = self.get_inference_times()
        if total_inference_time is not None:
            metadata["inference_times"] = [round(total_inference_time, 3)]
            if self.number_of_inferences > 1:
                metadata["inference_times"].append(round(average_inference_time, 3))

        return metadata

    def reset_inference_times(self):
        self.total_inference_time = 0
        self.number_of_inferences = 0

    def get_inference_times(self):
        """
        Returns the total inference time and the average inference time
        """
        if self.number_of_inferences == 0:
            return None, None
        return (
            self.total_inference_time,
            self.total_inference_time / self.number_of_inferences,
        )

    def predict(self, input, model_img_size=640):
        t0 = time.time()
        self.results = self.model.forward(
            input, augment=self.augment, size=model_img_size
        )
        self.total_inference_time += time.time() - t0
        self.number_of_inferences += 1
        return self.results

    def get_classes(self):
        res = self.results
        classes = res.pandas().xyxy[0]["class"].tolist()
        return classes

    def get_names(self):
        if self.class_names is None:
            res = self.results
            names = res.pandas().xyxy[0]["name"].tolist()
            return names
        else:
            classes = self.get_classes()
            names = []
            for i in range(len(classes)):
                names.append(self.class_names[classes[i]])
            return names

    def get_scores(self):
        res = self.results
        scores = res.pandas().xyxy[0]["confidence"].tolist()
        return scores

    def get_boxes(self):
        res = self.results
        boxes = []
        for i in range(len(res.pandas().xyxy[0])):
            box = []
            box.append(res.pandas().xyxy[0].get("xmin")[i])
            box.append(res.pandas().xyxy[0].get("ymin")[i])
            box.append(res.pandas().xyxy[0].get("xmax")[i])
            box.append(res.pandas().xyxy[0].get("ymax")[i])
            boxes.append(box)

        return boxes

    def get_indexes(self):
        boxes = self.get_boxes()
        if self.model.multi_label:
            overlapping = []
            for bb1 in range(len(boxes)):
                overlapping_bb1 = []
                for bb2 in range(bb1 + 1, len(boxes)):
                    # for bb2 in range( len(boxes)):
                    iou = self._compute_iou(boxes[bb1], boxes[bb2])
                    if iou > self.multi_label_iou_threshold:
                        overlapping_bb1.append(bb2)
                overlapping.append(overlapping_bb1)
            return self._get_overlapping_objects(overlapping)
        else:
            return [i for i in range(len(boxes))]

    def get_crops(self):
        res = self.results
        crops = []
        for i in range(len(res.imgs)):
            img_array = res.imgs[i]
            image_width = img_array.shape[1]
            image_height = img_array.shape[0]

            for coordlist in res.xyxy[i].tolist():
                x_start = int(coordlist[0])
                if x_start - self.margin < 0:
                    x_start = 0
                else:
                    x_start = x_start - self.margin
                y_start = int(coordlist[1])
                if y_start - self.margin < 0:
                    y_start = 0
                else:
                    y_start = y_start - self.margin
                x_end = int(coordlist[2])
                if x_end + self.margin > image_width:
                    x_end = image_width
                else:
                    x_end = x_end + self.margin
                y_end = int(coordlist[3])
                if y_end + self.margin > image_height:
                    y_end = image_height
                else:
                    y_end = y_end + self.margin
                crop = img_array[y_start:y_end, x_start:x_end]
                crops.append(crop)
        return crops

    # https://stackoverflow.com/a/42874377
    def _compute_iou(self, bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : list
            box format: [xmin, ymin, xmax, ymax]

        bb2 : list
            box format: [xmin, ymin, xmax, ymax]

        Returns
        -------
        float
            in [0, 1]

        """

        assert bb1[0] < bb1[2]
        assert bb1[1] < bb1[3]
        assert bb2[0] < bb2[2]
        assert bb2[1] < bb2[3]

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1[0], bb2[0])
        y_top = max(bb1[1], bb2[1])
        x_right = min(bb1[2], bb2[2])
        y_bottom = min(bb1[3], bb2[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1[2] - bb1[0]) * (bb1[3] - bb1[1])
        bb2_area = (bb2[2] - bb2[0]) * (bb2[3] - bb2[1])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def _get_overlapping_objects(self, overlapping):
        """
        Get the overlapping objects from a list of overlapping objects.
        """
        indexes = []
        known_ids = []
        new_indexes = [0 for i in range(len(overlapping))]
        for i in range(len(overlapping)):
            if i not in known_ids:
                idx = self._get_related_elements(overlapping, i, [i])
                known_ids += idx
                indexes.append(idx)

        for i in range(len(indexes)):
            for j in range(len(indexes[i])):
                new_indexes[indexes[i][j]] = i

        return new_indexes

    def _get_related_elements(self, list, index, known_elements=[]):
        elements = known_elements
        # if not index in elements:
        #    elements.append(index)

        for item in list[index]:
            if item not in elements:
                elements.append(item)
                # print("append ",item,"to",elements)
                additional_elements = self._get_related_elements(list, item, elements)
                for element in additional_elements:
                    if element not in elements:
                        elements.append(element)
                        # print("append ",element,"to",elements)

        for i in range(len(list)):
            for item in list[i]:
                if item in elements:
                    if i not in elements:
                        # print("append ",i,"to",elements)
                        elements.append(i)

        return sorted(elements)
