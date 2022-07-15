import logging
import sys

log = logging.getLogger(__name__)
log.propagate = False
log.setLevel(logging.INFO)
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(
    logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")
)
log.addHandler(handler)

logging.basicConfig(level=logging.INFO)
import zmq
import time
import os
from PIL import Image
import base64
import yaml
import json

import argparse
from yolomodelhelper import YoloModel
from messagehelper import MessageGenerator, Flower, Pollinator
import socket
from tqdm import tqdm

argparser = argparse.ArgumentParser(description="Pollinator Inference")
argparser.add_argument("--config", type=str, default="config.yaml", help="config file")
args = argparser.parse_args()
# parse yaml configuration file
with open(args.config, "r") as stream:
    try:
        cfg = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        log.error(exc)
        exit(1)

HOSTNAME = socket.gethostname()


# Flower Model configuration
model_flower_config = cfg.get("models").get("flower")
MODEL_FLOWER_WEIGHTS = model_flower_config.get("weights_path")
MODEL_FLOWER_CLASS_NAMES = model_flower_config.get("class_names")
MODEL_FLOWER_CONFIDENCE_THRESHOLD = model_flower_config.get("confidence_threshold")
MODEL_FLOWER_IOU_THRESHOLD = model_flower_config.get("iou_threshold")
MODEL_FLOWER_MARGIN = model_flower_config.get("margin")
MODEL_FLOWER_MULTI_LABEL = model_flower_config.get("multi_label")
MODEL_FLOWER_MULTI_LABEL_IOU_THRESHOLD = model_flower_config.get(
    "multi_label_iou_threshold"
)
MODEL_FLOWER_MAX_DETECTIONS = model_flower_config.get("max_detections")
MODEL_FLOWER_AUGMENT = model_flower_config.get("augment", False)
MODEL_FLOWER_IMG_SIZE = model_flower_config.get("image_size")


model_pollinator_config = cfg.get("models").get("pollinator")
MODEL_POLLINATOR_WEIGHTS = model_pollinator_config.get("weights_path")
MODEL_POLLINATOR_CLASS_NAMES = model_pollinator_config.get("class_names")
MODEL_POLLINATOR_IMG_SIZE = model_pollinator_config.get("image_size")
MODEL_POLLINATOR_CONFIDENCE_THRESHOLD = model_pollinator_config.get(
    "confidence_threshold"
)
MODEL_POLLINATOR_IOU_THRESHOLD = model_pollinator_config.get("iou_threshold")
MODEL_POLLINATOR_MARGIN = model_pollinator_config.get("margin")
MODEL_POLLINATOR_AUGMENT = model_pollinator_config.get("augment", False)
MODEL_POLLINATOR_MAX_DETECTIONS = model_pollinator_config.get("max_detections")
MODEL_POLLINATOR_MULTI_LABEL = model_pollinator_config.get("multi_label")
MODEL_POLLINATOR_MULTI_LABEL_IOU_THRESHOLD = model_pollinator_config.get(
    "multi_label_iou_threshold"
)

# Input configuration (zmq)
zmq_config = cfg.get("message_queue")
ZMQ_HOST = zmq_config.get("zmq_host")
ZMQ_PORT = zmq_config.get("zmq_port")
ZMQ_REQ_TIMEOUT = zmq_config.get("request_timeout", 3000)
ZMQ_REQ_RETRIES = zmq_config.get("request_retries", 10)


context = zmq.Context().instance()
log.info("Connecting to ZMQ server on tcp://{}:{}".format(ZMQ_HOST, ZMQ_PORT))
client = context.socket(zmq.REQ)
client.connect("tcp://{}:{}".format(ZMQ_HOST, ZMQ_PORT))


def request_message(code, client):
    """
    request codes:
        0: get first message
        1: get first message and remove it from queue
        2: remove first message from queue
    response:
        dict with message or
        response codes:
            0: no data available
            1: first message removed from queue
    """
    log.info("Sending request with code {}".format(code))
    client.send_json(code)
    retries_left = ZMQ_REQ_RETRIES
    while True:
        if (client.poll(ZMQ_REQ_TIMEOUT) & zmq.POLLIN) != 0:
            reply = client.recv_json()

            # print("Server replied (%s)", type(reply))
            return reply
        retries_left -= 1
        log.warning("No response from server")
        client.setsockopt(zmq.LINGER, 0)
        client.close()

        if retries_left == 0:
            log.error("Server seems to be offline, abandoning")
            exit(1)
        log.info("Reconnecting to server…")
        # Create new connection
        client = context.socket(zmq.REQ)
        client.connect("tcp://{}:{}".format(ZMQ_HOST, ZMQ_PORT))

        log.info("Resending code {}".format(code))
        client.send_json(code)


# Init Flower Model
flower_model = YoloModel(
    MODEL_FLOWER_WEIGHTS,
    confidence_threshold=MODEL_FLOWER_CONFIDENCE_THRESHOLD,
    iou_threshold=MODEL_FLOWER_IOU_THRESHOLD,
    margin=MODEL_FLOWER_MARGIN,
    class_names=MODEL_FLOWER_CLASS_NAMES,
    multi_label=MODEL_FLOWER_MULTI_LABEL,
    multi_label_iou_threshold=MODEL_FLOWER_MULTI_LABEL_IOU_THRESHOLD,
    augment=MODEL_FLOWER_AUGMENT,
    max_det=MODEL_FLOWER_MAX_DETECTIONS,
)

# Init Pollinator Model
pollinator_model = YoloModel(
    MODEL_POLLINATOR_WEIGHTS,
    confidence_threshold=MODEL_POLLINATOR_CONFIDENCE_THRESHOLD,
    iou_threshold=MODEL_POLLINATOR_IOU_THRESHOLD,
    margin=MODEL_POLLINATOR_MARGIN,
    class_names=MODEL_POLLINATOR_CLASS_NAMES,
    multi_label=MODEL_POLLINATOR_MULTI_LABEL,
    multi_label_iou_threshold=MODEL_POLLINATOR_MULTI_LABEL_IOU_THRESHOLD,
    augment=MODEL_POLLINATOR_AUGMENT,
    max_det=MODEL_POLLINATOR_MAX_DETECTIONS,
)

while True:
    # Get first message
    msg = request_message(1, client)  # get first message, remove it from queue
    if type(msg) == dict:
        # get filename
        filename = msg.get("filename")
        img = Image.open(filename)
        generator = MessageGenerator()
        generator.set_filename(filename)

        flower_model.reset_inference_times()
        pollinator_model.reset_inference_times()
        pollinator_index = 0
        # predict flower
        flower_model.predict(img)
        flower_crops = flower_model.get_crops()
        flower_boxes = flower_model.get_boxes()
        flower_classes = flower_model.get_classes()
        flower_scores = flower_model.get_scores()
        flower_names = flower_model.get_names()
        for flower_index in tqdm(range(len(flower_crops))):
            # add flower to message
            # TODO: add flower to message
            width, height = (
                flower_crops[flower_index].shape[1],
                flower_crops[flower_index].shape[0],
            )
            flower_obj = Flower(
                index=flower_index,
                class_name=flower_names[flower_index],
                score=flower_scores[flower_index],
                width=width,
                height=height,
            )
            generator.add_flower(flower_obj)
            # predict pollinator
            pollinator_model.predict(flower_crops[flower_index])
            pollinator_boxes = pollinator_model.get_boxes()
            pollinator_crops = pollinator_model.get_crops()
            pollinator_classes = pollinator_model.get_classes()
            pollinator_scores = pollinator_model.get_scores()
            pollinator_names = pollinator_model.get_names()
            pollinator_indexes = pollinator_model.get_indexes()
            for detected_pollinator in range(len(pollinator_crops)):
                idx = pollinator_index + pollinator_indexes[detected_pollinator]
                crop_image = Image.fromarray(pollinator_crops[detected_pollinator])
                width_polli, height_polli = crop_image.size
                # add pollinator to message
                pollinator_obj = Pollinator(
                    index=idx,
                    flower_index=flower_index,
                    class_name=pollinator_names[detected_pollinator],
                    score=pollinator_scores[detected_pollinator],
                    width=width_polli,
                    height=height_polli,
                    crop=crop_image,
                )
                generator.add_pollinator(pollinator_obj)
            if len(pollinator_indexes) > 0:
                pollinator_index += max(pollinator_indexes) + 1
        result = generator.generate_message()
        print(json.dumps(result))
    elif type(msg) == int:
        if msg == 0:  # no data available
            log.info("No data available")
            time.sleep(5)
