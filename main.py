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
import time
import os
from PIL import Image
import base64
import yaml
import json

import argparse
from yolomodelhelper import YoloModel
from messagehelper import MessageGenerator, Flower, Pollinator, MQTTClient, HTTPClient
from inputs import ZMQClient, DirectoryInput
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


# Input Configuration
input_config = cfg.get("input")
INPUT_TYPE = input_config.get("type")
if INPUT_TYPE is None:
    log.error("Input type not specified")
    exit(1)
zmq_client = None
dir_input = None
if INPUT_TYPE == "message_queue":
    # ZMQ Input Configuration
    zmq_config = input_config.get("message_queue")

    ZMQ_HOST = zmq_config.get("zmq_host")
    ZMQ_PORT = zmq_config.get("zmq_port")
    ZMQ_REQ_TIMEOUT = zmq_config.get("request_timeout", 3000)
    ZMQ_REQ_RETRIES = zmq_config.get("request_retries", 10)
    zmq_client = ZMQClient(ZMQ_HOST, ZMQ_PORT, ZMQ_REQ_TIMEOUT, ZMQ_REQ_RETRIES)
else:
    # Directory Input Configuration
    directory_config = input_config.get("directory")
    if directory_config is None:
        log.error("Directory input configuration is missing")
        exit(1)
    INPUT_DIRECTORY_BASE_DIR = directory_config.get("base_dir")
    INPUT_DIRECTORY_EXTENSION = directory_config.get("extension")
    dir_input = DirectoryInput(INPUT_DIRECTORY_BASE_DIR, INPUT_DIRECTORY_EXTENSION)
    dir_input.scan()

REMOVE_FILES_AFTER_PROCESSING = input_config.get("remove_after_processing", False)
if REMOVE_FILES_AFTER_PROCESSING:
    log.warning("Removing files after processing")

# Output Configuration
output_config = cfg.get("output")
IGNORE_EMPTY_RESULTS = output_config.get("ignore_empty_results", False)
# Output Configuration (File)
STORE_FILE = False
BASE_DIR = "output"
SAVE_CROPS = True
if output_config.get("file") is not None:
    output_config_file = output_config.get("file")
    if output_config_file.get("store_file", False):
        STORE_FILE = True
        BASE_DIR = output_config_file.get("base_dir", "output")
        SAVE_CROPS = output_config_file.get("save_crops", True)
        log.info("store_file is enabled, base_dir: {}".format(BASE_DIR))


# Output configuration (MQTT)
TRANSMIT_MQTT = False
mclient = None
if output_config.get("mqtt") is not None:
    output_config_mqtt = output_config.get("mqtt")
    if output_config_mqtt.get("transmit_mqtt", False):
        TRANSMIT_MQTT = True
        log.info("Transmitting to MQTT")
        mqtt_host = output_config_mqtt.get("host")
        mqtt_port = output_config_mqtt.get("port")
        mqtt_topic = output_config_mqtt.get("topic")
        mqtt_topic = mqtt_topic.replace("${hostname}", HOSTNAME)
        mqtt_username = output_config_mqtt.get("username")
        mqtt_password = output_config_mqtt.get("password")
        mqtt_use_tls = output_config_mqtt.get("use_tls", mqtt_port == 8883)
        log.info(
            "MQTT host: {}, port: {}, topic: {}, username {} use_tls: {}".format(
                mqtt_host, mqtt_port, mqtt_topic, mqtt_username, mqtt_use_tls
            )
        )
        mclient = MQTTClient(
            mqtt_host, mqtt_port, mqtt_topic, mqtt_username, mqtt_password, mqtt_use_tls
        )

# Output configuration (HTTP)
TRANSMIT_HTTP = False
hclient = None
if output_config.get("http") is not None:
    output_config_http = output_config.get("http")
    if output_config_http.get("transmit_http", False):
        TRANSMIT_HTTP = True
        log.info("Transmitting to HTTP")
        http_url = output_config_http.get("url")
        http_url = http_url.replace("${hostname}", HOSTNAME)
        http_username = output_config_http.get("username")
        http_password = output_config_http.get("password")
        http_method = output_config_http.get("method", "POST")
        log.info(
            "HTTP url: {}, method: {}, username: {}".format(
                http_url, http_method, http_username
            )
        )
        hclient = HTTPClient(http_url, http_username, http_password, http_method)


def get_filename():
    if INPUT_TYPE == "message_queue":
        msg = zmq_client.request_message(1)
        if type(msg) == dict:
            filename = msg.get("filename")
            if filename is None:
                log.error("No filename found in message")
                return None
            return filename
        elif type(msg) == int:
            if msg == 0:  # no data available
                return None
            else:
                log.info("Got message with code %d", msg)
                return None
    else:
        return dir_input.get_next()


# Init Flower Model
flower_model = YoloModel(
    MODEL_FLOWER_WEIGHTS,
    image_size=MODEL_FLOWER_IMG_SIZE,
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
    image_size=MODEL_POLLINATOR_IMG_SIZE,
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
    filename = get_filename()
    if filename is not None:
        generator = MessageGenerator()
        log.info("Processing image: %s", os.path.basename(filename))
        generator.set_filename(os.path.basename(filename))

        flower_model.reset_inference_times()
        pollinator_model.reset_inference_times()
        pollinator_index = 0
        # predict flower
        try:
            img = Image.open(filename)
            original_width, original_height = img.size
            flower_model.predict(img)
        except Exception as e:
            log.error("Error predicting flowers on file %s: %s", filename, e)
            continue
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
        log.info("Found {} flowers in {} ms".format(len(flower_crops), int(flower_model.get_inference_times()[0]*1000)))
        log.info("Found {} pollinators in {} ms".format(pollinator_index, int(pollinator_model.get_inference_times()[0]*1000)))
        # add metadata to message
        generator.add_metadata(flower_model.get_metadata(), "flower_inference")
        generator.add_metadata(pollinator_model.get_metadata(), "pollinator_inference")
        generator.add_metadata(
            {"size": [original_width, original_height]}, "original_image"
        )

        result = generator.generate_message()
        if IGNORE_EMPTY_RESULTS and len(generator.pollinators) == 0:
            log.info("No pollinators detected, skipping")
            continue
        if STORE_FILE:
            generator.store_message(BASE_DIR, SAVE_CROPS)
        if TRANSMIT_HTTP:
            hclient.send_message(
                result,
                filename=generator.generate_filename(),
                node_id=generator.node_id,
                hostname=HOSTNAME,
            )
        if TRANSMIT_MQTT:
            mclient.publish(
                result,
                filename=generator.generate_filename(),
                node_id=generator.node_id,
                hostname=HOSTNAME,
            )

        # print(json.dumps(result))
        if REMOVE_FILES_AFTER_PROCESSING:
            log.info("Removing file %s", filename)
            os.remove(filename)

    else:
        log.info("No data available")
        time.sleep(5)
