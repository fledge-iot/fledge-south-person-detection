# -*- coding: utf-8 -*-

# FLEDGE_BEGIN
# See: http://fledge.readthedocs.io/
# FLEDGE_END

""" Human Detector Plugin
"""
__author__ = "Amandeep Singh Arora, Deepanshu Yadav"
__copyright__ = "Copyright (c) 2020 Dianomic Systems Inc."
__license__ = "Apache 2.0"
__version__ = "${VERSION}"

import asyncio
import copy
import uuid
import logging
import os
import time
import subprocess

import threading
from threading import Thread
from aiohttp import web, MultipartWriter


import cv2
import numpy as np

from fledge.common import logger
from fledge.plugins.common import utils
import async_ingest

from fledge.plugins.south.person_detection.web_stream import WebStream
from fledge.plugins.south.person_detection.frame_processing import FrameProcessor


_LOGGER = logger.setup(__name__, level=logging.INFO)
BACKGROUND_TASK = False


def check_background():
    out = subprocess.Popen(['systemctl', 'status', 'fledge'],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    if str(stdout).find("active (running)") != -1:
        return True
    else:
        return False


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
        'default': 'detect.tflite',
        'order': '1',
        'displayName': 'TFlite model file'
    },
    'labels_file': {
        'description': 'Labels file used during inference',
        'type': 'string',
        'default': 'coco_labels.txt',
        'order': '2',
        'displayName': 'Labels file'
    },
    'asset_name': {
        'description': 'Asset Name',
        'type': 'string',
        'default': 'Detection Results',
        'order': '3',
        'displayName': 'Asset Name'
    },
    'enable_edge_tpu': {
        'description': 'Connect the Coral Edge TPU and enable this',
        'type': 'boolean',
        'default': 'false',
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
    'camera_id': {
        'description': 'The number associated  with your video device. See /dev in your '
                       'filesystem you will see video0 or video1',
        'type': 'integer',
        'default': '0',
        'order': '6',
        'displayName': 'Camera ID'
    },
    'enable_window': {
        'description': 'Show detection results in a window default :False',
        'type': 'boolean',
        'default': 'false',
        'order': '7',
        'displayName': 'Enable Detection  Window'
    },
    'enable_web_streaming': {
        'description': 'Enable Web Streaming default :True',
        'type': 'boolean',
        'default': 'true',
        'order': '8',
        'displayName': 'Enable Web Streaming'
    },
    'web_streaming_port_no': {
        'description': 'Port number for web streaming',
        'type': 'string',
        'default': '8085',
        'order': '9',
        'displayName': 'Web streaming Port number',
        "validity": "enable_web_streaming == \"true\" "
    },
}


# GLOBAL VARIABLES DECLARATION
c_callback = None
c_ingest_ref = None
frame_processor =None
loop = None
async_thread = None
enable_web_streaming = None
web_stream = None
BACKGROUND_TASK = False


def plugin_info():
    """ Returns information about the plugin.
    Args:
    Returns:
        dict: plugin information
    Raises:
    """

    return {
        'name': 'Person Detection plugin',
        'version': '1.8.0',
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
    global BACKGROUND_TASK
    BACKGROUND_TASK = check_background()
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
        handle['camera_height'] = 640
        handle['camera_width'] = 480

        web_streaming_port_no = int(handle['web_streaming_port_no']['value'])

        if handle['enable_web_streaming']['value'] == 'true':
            enable_web_streaming = True
        else:
            enable_web_streaming = False

        def run():
            global loop
            loop.run_forever()

        if enable_web_streaming:
            global web_stream
            web_stream = WebStream(port=web_streaming_port_no).start_web_streaming_server(local_loop=loop)
            async_thread = Thread(target=run, name="Async Thread")
            async_thread.daemon = True
            async_thread.start()

        global frame_processor
        frame_processor = FrameProcessor(handle)
        frame_processor.start()

    except Exception as ex:
        _LOGGER.exception("Human detector plugin failed to start. Details: %s", str(ex))
        raise
    else:
        _LOGGER.info("Plugin started")


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
        #
        frame_processor.shutdown_in_progress = True
        # allow the stream to stop
        time.sleep(3)
        # stopping every other thread one by one
        frame_processor.join()
        frame_processor = None
        if enable_web_streaming:
            web_stream.SHUTDOWN_IN_PROGRESS = True
            web_stream.stop_server(loop)
        loop.stop()
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
    global c_callback, c_ingest_ref
    c_callback = callback
    c_ingest_ref = ingest_ref
