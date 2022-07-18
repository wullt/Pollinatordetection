import zmq
import os
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


class ZMQClient:
    def __init__(self, host, port, timeout=3000, retries=20):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.retries = retries
        self.context = zmq.Context().instance()
        self.client = self.context.socket(zmq.REQ)
        self.client.connect("tcp://{}:{}".format(self.host, self.port))
        log.info("Connecting to tcp://{}:{}".format(self.host, self.port))
        self.retries_left = self.retries

    def request_message(self, code):
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
        log.info("Sending request code {}".format(code))

        self.client.send_json(code)
        retries_left = self.retries
        while True:
            if (self.client.poll(self.timeout) & zmq.POLLIN) != 0:
                reply = self.client.recv_json()

                # print("Server replied (%s)", type(reply))
                return reply
            retries_left -= 1
            log.warning("No response from server")

            self.client.setsockopt(zmq.LINGER, 0)
            self.client.close()

            if retries_left == 0:
                log.error(
                    "ZMQ server could not be reached, abandoning\nmake sure the server is running and the port is correct"
                )

                exit(1)
            # Create new connection
            log.info("Reconnecting to server… {} retries left".format(retries_left))

            self.client = self.context.socket(zmq.REQ)
            self.client.connect("tcp://{}:{}".format(self.host, self.port))

            self.client.send_json(code)

    def close(self):
        self.socket.close()
        self.context.term()


class DirectoryInput:
    """
    Load images from a local directory
    """

    def __init__(self, path, format="jpg"):
        self.path = path
        self.format = format
        self.files = []
        self.index = 0

    def scan(self):
        """
        Scan the directory for new images
        """
        print("scanning directory")
        unseen_files = []
        for dir_path, dir_names, file_names in os.walk(self.path):
            for f in file_names:
                if f.endswith(self.format):
                    # yield os.path.join(dir_path, f)
                    abs_fpath = os.path.join(dir_path, f)
                    if abs_fpath not in self.files:
                        unseen_files.append(abs_fpath)
        unseen_files.sort(key=lambda x: os.path.getmtime(x))
        print("adding {} new files".format(len(unseen_files)))
        self.files += unseen_files

    def get_next(self):
        """
        Get the next image in the directory
        """
        if self.index >= len(self.files):
            self.scan()
        if self.index >= len(self.files):
            return None
        self.index += 1
        return self.files[self.index - 1]

