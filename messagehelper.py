import json
import datetime
from PIL import Image
from io import BytesIO
import base64
from dataclasses import dataclass
import os
import sys

import logging
import ssl
import requests

log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
log.addHandler(handler)

DECIMALS_TO_ROUND = 3


@dataclass
class Flower:
    index: int
    class_name: str
    score: float
    width: int
    height: int

    def to_dict(self):
        return {
            "index": self.index,
            "class_name": self.class_name,
            "score": round(float(self.score), DECIMALS_TO_ROUND),
            "size": [self.width, self.height],
        }


@dataclass
class Pollinator:
    index: int
    flower_index: int
    class_name: str
    score: float
    width: int
    height: int
    crop: Image

    def to_dict(self, save_crop=True):

        pollintor_dict = {
            "index": self.index,
            "flower_index": self.flower_index,
            "class_name": self.class_name,
            "score": round(self.score, DECIMALS_TO_ROUND),
            "crop": None,
        }
        if save_crop:
            bio = BytesIO()
            self.crop.save(bio, format="JPEG")
            bio.seek(0)
            encoded_image = base64.b64encode(bio.read()).decode("utf-8")
            pollintor_dict["crop"] = encoded_image
        return pollintor_dict


class MessageGenerator:
    def __init__(self):
        self.node_id = None
        self.timestamp = None
        self.flowers = []
        self.pollinators = []
        self.metadata = {}
        self.filename = None

    def set_filename(self, filename):
        self.filename = filename.split("/")[-1].split(".")[0]
        node_id, timestamp = self.get_nodeid_timestamp_from_filename(self.filename)
        self.set_node_id(node_id)
        self.set_timestamp(datetime.datetime.strptime(timestamp, "%Y-%m-%dT%H-%M-%SZ"))

    def set_timestamp(self, timestamp):
        self.timestamp = timestamp

    def set_node_id(self, node_id):
        self.node_id = node_id

    def add_metadata(self, metadata: dict, key: str):
        self.metadata[key] = metadata

    def set_metadata(self, metadata: dict):
        self.metadata = metadata

    def add_flower(self, flower: Flower):
        self.flowers.append(flower)

    def add_pollinator(self, pollinator: Pollinator):
        self.pollinators.append(pollinator)

    def generate_message(self, save_crop=True):
        flowers = []
        pollinators = []
        for flower in self.flowers:
            flowers.append(flower.to_dict())
        for pollinator in self.pollinators:
            pollinators.append(pollinator.to_dict())
        flowers.sort(key=lambda x: x["index"])
        pollinators.sort(key=lambda x: x["index"])

        message = {
            "detections": {"flowers": flowers, "pollinators": pollinators},
            "metadata": self.metadata,
        }
        return message

    def generate_filename(self, format=".json"):
        filename = (
            self.node_id + "_" + self.timestamp.strftime("%Y-%m-%dT%H-%M-%SZ") + format
        )
        return filename

    def get_nodeid_timestamp_from_filename(self, filename):
        filename_parts = filename.split("_")
        return filename_parts[0], filename_parts[1]

    def _generate_save_path(self):
        date_dir = self.timestamp.strftime("%Y-%m-%d")
        time_dir = self.timestamp.strftime("%H")
        return self.node_id + "/" + date_dir + "/" + time_dir + "/"

    def store_message(self, base_dir, save_crop=True):
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        if not base_dir.endswith("/"):
            base_dir += "/"
        filepath = base_dir + self._generate_save_path()
        if not os.path.exists(filepath):
            os.makedirs(filepath)
            log.info("Created directory: {}".format(filepath))
        with open(filepath + self.generate_filename(), "w") as f:
            json.dump(self.generate_message(save_crop=save_crop), f)
        log.info("Saved message to: {}".format(filepath + self.generate_filename()))
        return True
