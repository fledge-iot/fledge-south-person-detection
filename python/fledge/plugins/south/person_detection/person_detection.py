# -*- coding: utf-8 -*-

# FLEDGE_BEGIN
# See: http://fledge-iot.readthedocs.io/
# FLEDGE_END

""" Human Detector Plugin
"""
__author__ = "Amandeep Singh Arora, Deepanshu Yadav"
__copyright__ = "Copyright (c) 2020 Dianomic Systems Inc."
__license__ = "Apache 2.0"
__version__ = "${VERSION}"

import asyncio
import copy
import subprocess
import logging
import os
import time
import threading
from threading import Thread
import cv2
import numpy as np

from fledge.common import logger
from fledge.plugins.common import utils
import async_ingest

from fledge.plugins.south.person_detection.videostream import VideoStream
from fledge.plugins.south.person_detection.inference import Inference
from fledge.plugins.south.person_detection.web_stream import WebStream

_LOGGER = logger.setup(__name__, level=logging.INFO)

_DEFAULT_CONFIG = {
    'plugin': {
        'description': 'Person Detection On Fledge',
        'type': 'string',
        'default': 'person_detection',
        'readonly': 'true'
    },
    'model_file': {
        'description': 'TFlite model file to use for inference',
        'type': 'string',
        'default': 'detect_edgetpu.tflite',
        'order': '1',
        'displayName': 'TFlite Model File'
    },
    'labels_file': {
        'description': 'Labels file used during inference',
        'type': 'string',
        'default': 'coco_labels.txt',
        'order': '2',
        'displayName': 'Labels File'
    },
    'asset_name': {
        'description': 'Asset name',
        'type': 'string',
        'default': 'person_detection',
        'order': '3',
        'displayName': 'Asset Name'
    },
    'enable_edge_tpu': {
        'description': 'Connect Coral Edge TPU and select this option to use TPU for inference',
        'type': 'boolean',
        'default': 'true',
        'order': '4',
        'displayName': 'Enable Edge TPU'
    },
    'min_conf_threshold': {
        'description': 'Threshold to select the detected objects',
        'type': 'float',
        'default': '0.5',
        'minimum': '0',
        'maximum': '1',
        'order': '5',
        'displayName': 'Minimum Confidence Threshold'
    },
    'source': {
        'description': 'Whether take input feed from camera or network stream',
        'type': 'enumeration',
        'default': 'stream',
        'options': ['stream', 'camera'],
        'order': '6',
        'displayName': 'Source of video feed'
    },
    'stream_url': {
        'description': 'The url of the network stream if network stream is to be used.',
        'type': 'string',
        'default': 'rtsp://ip:port',
        'order': '7',
        'validity': "source == \"stream\"",
        'displayName': 'Stream URL'
    },
    'opencv_backend': {
        'description': 'Backend for processing stream from network',
        'type': 'enumeration',
        'default': 'ffmpeg',
        'options': ['ffmpeg'],
        'order': '8',
        'validity': "source == \"stream\"",
        'displayName': 'OpenCV Backend'
    },
    'stream_protocol': {
        'description': 'The protocol being used by streaming server',
        'type': 'enumeration',
        'default': 'udp',
        'options': ['udp'],
        'order': '9',
        'validity': "source == \"stream\"",
        'displayName': 'Stream Protocol'
    },
    'camera_id': {
        'description': 'The number associated with your video device. See /dev in your '
                       'filesystem. Enter 0 to use /dev/video0, and so on.',
        'type': 'integer',
        'default': '0',
        'order': '10',
        'validity': "source == \"camera\"",
        'displayName': 'Camera ID'
    },
    'enable_window': {
        'description': 'Show detection results in a window',
        'type': 'boolean',
        'default': 'false',
        'order': '11',
        'displayName': 'Enable Detection Window'
    },
    'enable_web_streaming': {
        'description': 'Enable web streaming on specified web streaming port',
        'type': 'boolean',
        'default': 'true',
        'order': '12',
        'displayName': 'Enable Web Streaming'
    },
    'web_streaming_port_no': {
        'description': 'Port number for web streaming',
        'type': 'string',
        'default': '8085',
        'order': '13',
        'displayName': 'Web Streaming Port',
        "validity": "enable_web_streaming == \"true\" "
    },
}

# GLOBAL VARIABLES DECLARATION
c_callback = None
c_ingest_ref = None
frame_processor = None
loop = None
async_thread = None
enable_web_streaming = None
web_stream = None
# Keeping a fixed camera resolution for now. Can give it inside configuration. However changing
# it is quite risky because some devices support changing camera resolution through openCV API
# but others simply don't (Like the coral board).
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640


def plugin_info():
    """ Returns information about the plugin.
    Args:
    Returns:
        dict: plugin information
    Raises:
    """

    return {
        'name': 'Person Detection plugin',
        'version': '2.0.1',
        'mode': 'async',
        'type': 'south',
        'interface': '1.0',
        'config': _DEFAULT_CONFIG
    }


def plugin_init(config):
    """ Initialise the plugin with the merged config
        Args:
           config: JSON configuration document for the South plugin configuration category
        Returns:
           data: JSON object to be used in future calls to the plugin
        Raises:
    """
    data = copy.deepcopy(config)
    return data


def plugin_start(handle):
    """ Start the plugin operation - observe video feed from camera device and classify that using the
        configured ML model
        Args:
            handle: handle returned by the plugin initialisation call
        Returns:
            Sends back the reading object (JSON doc) via C/python async ingest interface when it recognizes
            the video frame content
        Raises:
    """
    global frame_processor, loop, async_thread, enable_web_streaming, web_stream

    loop = asyncio.new_event_loop()
    try:

        # some extra config parameters required for the camera_loop function
        # Since pixel value can be from 0 to 255 , so we are considering mean to
        # be (0+255)/2 = 127.5 .
        handle['input_mean'] = 127.5
        handle['input_std'] = 127.5
        handle['camera_height'] = CAMERA_HEIGHT
        handle['camera_width'] = CAMERA_WIDTH

        web_streaming_port_no = int(handle['web_streaming_port_no']['value'])

        if handle['enable_web_streaming']['value'] == 'true':
            enable_web_streaming = True
        else:
            enable_web_streaming = False

        def run():
            global loop
            loop.run_forever()

        global frame_processor
        frame_processor = FrameProcessor(handle)
        if frame_processor.is_camera_functional and frame_processor.interpreter_loaded:

            if enable_web_streaming:

                # make shutdown flag of web stream server false.
                WebStream.SHUTDOWN_IN_PROGRESS = False

                web_stream = WebStream(port=web_streaming_port_no).start_web_streaming_server(local_loop=loop)
                async_thread = Thread(target=run, name="Async Thread")
                async_thread.daemon = True
                async_thread.start()

            frame_processor.start()
        else:
            raise Exception("Camera is not functional. Please shutdown and try again. ")

    except Exception as ex:
        _LOGGER.exception("Human detector plugin failed to start. Details: %s", str(ex))
        raise
    else:
        _LOGGER.info("Plugin started")


def check_need_to_shutdown(handle, new_config, parameters_to_check):
    """
    Checks whether shutdown is required if we change configuration.
    Args:
        handle: Old configuration
        new_config: New Configuration
        parameters_to_check: Parameters (list) whose value if changed then shutdown is required

    Returns:
        True or False
    """
    need_to_shutdown = False
    for parameter in parameters_to_check:
        old_value = handle[parameter]['value']
        new_value = new_config[parameter]['value']
        if new_value != old_value:
            need_to_shutdown = True

    return need_to_shutdown


def plugin_reconfigure(handle, new_config):
    """ Reconfigures the plugin
        it should be called when the configuration of the plugin is changed
        during the operation of the south service.
    Args:
        handle: handle returned by the plugin initialisation call
        new_config: JSON object representing the new configuration category for the category
    Returns:
        new_handle: new handle to be used in the future calls
    Raises:
    """
    global frame_processor

    parameters_to_check = ['camera_id', 'enable_window', 'enable_web_streaming',
                           'web_streaming_port_no',
                           'source', 'stream_url', 'stream_protocol', 'opencv_backend']
    need_to_shutdown = check_need_to_shutdown(handle, new_config, parameters_to_check)
    if not need_to_shutdown:
        _LOGGER.debug("No need to shutdown")
        frame_processor.handle_new_config(new_config)
        new_handle = plugin_init(new_config)
        return new_handle
    
    else:

        was_native_window_enabled = handle['enable_window']['value'] == 'true'
        is_native_window_enabled_now = new_config['enable_window']['value'] == 'true'

        # case  was_native_window_enabled  is_native_window_enabled_now
        # 1.          True                       True
        # 2.          True                       False
        # 3.          False                      True
        # 4.          False                      False

        # Only case 4. should be allowed and rest of the cases are causing problems.
        # Refer to the following link for more details
        # https://stackoverflow.com/questions/60737852/opencv-imshow-hangs-if-called-two-times-from-a-thread

        if was_native_window_enabled or is_native_window_enabled_now:
            _LOGGER.warning("Enabling the native window during reconfigure is known to cause problems. "
                            "Restart the plugin manually. ")
            new_handle = plugin_init(new_config)
            return new_handle

        else:
            plugin_shutdown(handle)
            new_handle = plugin_init(new_config)
            plugin_start(new_handle)
            return new_handle


def plugin_shutdown(handle):
    """ Shutdowns the plugin doing required cleanup
        To be called prior to the south service being shut down.

    Args:
        handle: handle returned by the plugin initialisation call
    Returns:
        None
    Raises:
        None
    """
    global frame_processor, loop, async_thread, enable_web_streaming, web_stream
    try:

        frame_processor.shutdown_in_progress = True
        # allow the stream to stop
        time.sleep(3)
        # stopping every other thread one by one

        # checking if the thread is started or not.
        if frame_processor.is_camera_functional and frame_processor.interpreter_loaded:
            frame_processor.join()

        frame_processor = None
        if enable_web_streaming:
            if web_stream is not None:
                WebStream.SHUTDOWN_IN_PROGRESS = True
                web_stream.stop_server(loop)
        loop.stop()
        # async_thread.join()
        async_thread = None
        loop = None

        _LOGGER.info('Plugin has shutdown')

    except Exception as ex:
        _LOGGER.exception(str(ex))
        raise


def plugin_register_ingest(handle, callback, ingest_ref):
    """ Required plugin interface component to communicate to South C server

    Args:
        handle: handle returned by the plugin initialisation call
        callback: C opaque object required to passed back to C/Python async ingest interface
        ingest_ref: C opaque object required to passed back to C/Python async ingest interface
    """
    _LOGGER.debug("register ingest")
    global c_callback, c_ingest_ref
    c_callback = callback
    c_ingest_ref = ingest_ref


class FrameProcessor(Thread):
    def __init__(self, handle):
        super(FrameProcessor, self).__init__()
        # if floating point model is used we need to subtract the mean and divide
        # by standard deviation
        self.input_mean = handle['input_mean']
        self.input_std = handle['input_std']

        # the height of the detection window on which frames are to be displayed
        self.camera_height = handle['camera_height']

        # the width of the detection window on which frames are to be displayed
        self.camera_width = handle['camera_width']

        model = handle['model_file']['value']
        labels = handle['labels_file']['value']
        self.asset_name = handle['asset_name']['value']
        enable_tpu = handle['enable_edge_tpu']['value']
        self.min_conf_threshold = float(handle['min_conf_threshold']['value'])

        model = os.path.join(os.path.dirname(__file__), "model", model)
        labels = os.path.join(os.path.dirname(__file__), "model", labels)

        with open(labels, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            labels = dict((int(k), v) for k, v in pairs)

        # instance of the self.inference class
        self.interpreter_loaded = False
        self.inference = Inference()

        local_interpreter = self.inference.get_interpreter(model, enable_tpu,
                                                           labels, self.min_conf_threshold)
        if local_interpreter is not None:
            self.interpreter_loaded = True

        if handle['enable_window']['value'] == 'true' and not FrameProcessor.check_background():
            self.enable_window = True
        else:
            self.enable_window = False

        if handle['source']['value'] == 'camera':
            camera_id = int(handle['camera_id']['value'])
            # Initialize the stream object and start the thread that keeps on reading frames
            # This thread is independent of the Camera Processing Thread
            self.videostream, is_camera_functional = VideoStream(resolution=(self.camera_width,
                                                                             self.camera_height),
                                                                 source=camera_id).start()
            self.is_camera_functional = is_camera_functional

        else:
            stream_url = handle['stream_url']['value']
            stream_protocol = handle['stream_protocol']['value']
            opencv_backend = handle['opencv_backend']['value']
            self.videostream, is_camera_functional = VideoStream(resolution=(self.camera_width,
                                                                             self.camera_height),
                                                                 stream_url=stream_url,
                                                                 stream_protocol=stream_protocol,
                                                                 opencv_backend=opencv_backend).start()
            self.is_camera_functional = is_camera_functional
        # For using the videostream with threading use the following :
        # videostream = VideoStream(resolution=(self.camera_width, self.camera_height),
        # source=source, enable_thread=True).start()
        self.shutdown_in_progress = False

    @staticmethod
    def check_background():
        """
        Checks for fledge running in background.
        Returns: True if fledge is running in background.

        """
        out = subprocess.Popen(['systemctl', 'status', 'fledge'],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
        stdout, stderr = out.communicate()
        if str(stdout).find("active (running)") != -1:
            return True
        else:
            return False

    @staticmethod
    def construct_readings(objs):
        """ Takes the detection results from the model and convert into readings suitable to insert into database.
             For Example
                Lets say a  single person is detected then there will be a single element in the array
                whose contents  will be
                               {'label': 'person',
                                'score': 64, # probability of prediction
                                'bounding_box': [xmin, ymin, xmax, ymax] # bounding box coordinates
                                }

                A reading will be constructed in the form given below :

                    reads = {
                            'person_' + '1' + '_' + 'label': 'person'
                            'person_' + '1' + '_' + 'score': 64
                            'person_' + '1' + '_' + 'x1': xmin
                            'person_' + '1' + '_' + 'y1': ymin
                            'person_' + '1' + '_' + 'x2': xmax
                            'person_' + '1' + '_' + 'y2': ymax
                            'count': 1
                            }

                Args:
                       x -> an array of detection results
                Returns: Readings to be inserted into database.
               Raises: None
           """

        reads = {}
        for r_index in range(len(objs)):
            reads['person_' + str(r_index + 1) + '_' + 'label'] = objs[r_index]['label']
            reads['person_' + str(r_index + 1) + '_' + 'score'] = objs[r_index]['score']
            reads['person_' + str(r_index + 1) + '_' + 'x1'] = objs[r_index]['bounding_box'][0]
            reads['person_' + str(r_index + 1) + '_' + 'y1'] = objs[r_index]['bounding_box'][1]
            reads['person_' + str(r_index + 1) + '_' + 'x2'] = objs[r_index]['bounding_box'][2]
            reads['person_' + str(r_index + 1) + '_' + 'y2'] = objs[r_index]['bounding_box'][3]

        reads['count'] = len(objs)

        return reads

    def wait_for_frame(self):
        """ Waits for frame to become available else sleeps for 200 milliseconds.
                Args:
                       self-> a videostream object
                Returns: None
               Raises: None
           """
        while True:
            if self.videostream.frame is not None:
                return
            else:
                time.sleep(0.2)

    def handle_new_config(self, new_config):
        """
        If shutdown is not required then it changes the configuration on the fly.
        Args:
            new_config: The configuration during reconfigure.

        Returns:
              None
        """

        _LOGGER.debug("Handling the reconfigure without shutdown")
        model = new_config['model_file']['value']
        labels = new_config['labels_file']['value']
        self.asset_name = new_config['asset_name']['value']
        enable_tpu = new_config['enable_edge_tpu']['value']
        self.min_conf_threshold = float(new_config['min_conf_threshold']['value'])

        model = os.path.join(os.path.dirname(__file__), "model", model)
        labels = os.path.join(os.path.dirname(__file__), "model", labels)

        with open(labels, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            labels = dict((int(k), v) for k, v in pairs)

        _ = self.inference.get_interpreter(model, enable_tpu,
                                           labels, self.min_conf_threshold)
        _LOGGER.debug("Handled the reconfigure")

    def run(self):

        # these variables are used for calculation of frame per seconds (FPS)
        frame_rate_calc = 1
        freq = cv2.getTickFrequency()
        # The thread is allowed to capture a few frames. See FOGL-4132 for details
        self.wait_for_frame()
        # ws = WebStream()

        while True:
            # Capture frame-by-frame
            t1 = cv2.getTickCount()
            global c_callback, c_ingest_ref

            # we need the height , width to resize the image for feeding into the model
            height_for_model = self.inference.height_for_model
            width_for_model = self.inference.width_for_model

            #  check if floating point model is used or not
            floating_model = self.inference.floating_model

            # The list of labels of the supported objects detected by the plugin
            labels = self.inference.labels

            # Taking the frame the stream
            frame1 = self.videostream.read()
            if frame1 is None:
                _LOGGER.error("Either the stream/camera device stopped working.")
                break

            frame = frame1.copy()
            # BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # resizing it to feed into model
            frame_resized = cv2.resize(frame_rgb, (width_for_model, height_for_model))
            # input_data will now become 4 dimensional
            input_data = np.expand_dims(frame_resized, axis=0)
            # now it will have (batchsize, height, width, channel)

            # Normalize pixel values if using a floating model
            # (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - self.input_mean) / self.input_std

            # Perform the actual detection by running the model with the image as input
            boxes, classes, scores = self.inference.perform_inference(input_data)

            # we could have got  number of objects
            # but it does not work most of the times.

            # num = interpreter.get_tensor(output_details[3]['index'])[0]  #
            # Total number of detected objects (inaccurate and not needed)

            # The readings array to be inserted in the readings table
            objs = []

            # Loop over all detections and draw detection box
            #  if confidence is above minimum then  only
            #  that detected object will  be considered

            # The index of person class is zero.
            for i in range(len(scores)):
                if (scores[i] > self.min_conf_threshold) and (int(classes[i] == 0)):
                    # Get bounding box coordinates and draw box
                    # Interpreter can return coordinates that are outside of image dimensions,
                    # need to force them to be within image using max() and min()

                    ymin_model = round(boxes[i][0], 3)
                    xmin_model = round(boxes[i][1], 3)
                    ymax_model = round(boxes[i][2], 3)
                    xmax_model = round(boxes[i][3], 3)

                    # map the bounding boxes from model to the window
                    ymin = int(max(0, (ymin_model * self.camera_height)))
                    xmin = int(max(0, (xmin_model * self.camera_width)))
                    ymax = int(min(self.camera_height, (ymax_model * self.camera_height)))
                    xmax = int(min(self.camera_width, (xmax_model * self.camera_width)))

                    # draw the rectangle on the frame
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                    # Contructing the label

                    # Look up object name from "labels" array using class index
                    object_name = labels[int(classes[i])]

                    # Example: 'person: 72%'
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))

                    # Get font size
                    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                          0.7, 2)

                    # Make sure not to draw label too close to top of window
                    label_ymin = max(ymin, labelSize[1] + 10)

                    # Draw white box to put label text in
                    cv2.rectangle(frame, (xmin, label_ymin - labelSize[1] - 10),
                                  (xmin + labelSize[0], label_ymin + baseLine - 10),
                                  (255, 255, 255), cv2.FILLED)

                    # Draw the text label
                    cv2.putText(frame, label, (xmin, label_ymin - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

                    # the readings to be inserted into the table
                    objs.append({'label': labels[classes[i]],
                                 'score': 100 * scores[i],
                                 'bounding_box': [xmin, ymin, xmax, ymax]
                                 })

            # Draw framerate in corner of frame
            cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc),
                        (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0)
                        , 2, cv2.LINE_AA)

            # All the results have been drawn on the frame, so it's time to display it.
            if self.shutdown_in_progress:
                _LOGGER.debug("Shut down breaking loop")
                break
            else:
                # Calculate framerate
                t_end = cv2.getTickCount()
                time1 = (t_end - t1) / freq
                frame_rate_calc = 1 / time1

                reads = FrameProcessor.construct_readings(objs)
                data = {
                    'asset': self.asset_name,
                    'timestamp': utils.local_timestamp(),
                    'readings': reads
                }

                async_ingest.ingest_callback(c_callback, c_ingest_ref, data)

                # show the frame on the window
                try:
                    if self.enable_window:
                        cv2.imshow("Human Detector", frame)

                    WebStream.FRAME = frame.copy()

                except Exception as e:
                    _LOGGER.info('exception  {}'.format(e))

                # wait for 1 milli second
                cv2.waitKey(1)

        WebStream.SHUTDOWN_IN_PROGRESS = True
        _LOGGER.debug("Shutdown flag of streaming server set True")
        _LOGGER.debug("Stopping the stream ")
        self.videostream.stop()
        _LOGGER.info("Camera stream has been stopped")
        time.sleep(2)
        cv2.destroyAllWindows()
        _LOGGER.debug("All windows destroyed")
