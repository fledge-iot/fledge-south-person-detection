# -*- coding: utf-8 -*-

# FLEDGE_BEGIN
# See: http://fledge.readthedocs.io/
# FLEDGE_END

""" Helper module for starting a separate thread for reading frame from the Camera Device
"""

import logging
import subprocess
from threading import Thread
import os

import cv2

from fledge.common import logger


_LOGGER = logger.setup(__name__, level=logging.INFO)


def detectCoralDevBoard():
    try:
        if 'MX8MQ' in open('/sys/firmware/devicetree/base/model').read():
            _LOGGER.info('Detected Coral dev board.')
            return True
    except:
        pass
    return False


def detect_mjpg_camera(source):
    out = subprocess.Popen(['v4l2-ctl', '--list-formats-ext', '--device', '/dev/video'.join(str(source))],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, _ = out.communicate()
    if str(stdout).find("MJPG") != -1:
        return True
    else:
        return False


class VideoStream:
    """ Camera object that controls video streaming from the Camera
    """

    def __init__(self, resolution=(640, 480), framerate=30, source=0, enable_thread=False, stream_url=None,
                 stream_protocol="udp", opencv_backend="ffmpeg"):
        # Initialize the PiCamera and the camera image stream

        if stream_url is not None:
            if opencv_backend == 'ffmpeg':
                if stream_protocol == "udp":
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
                    self.stream = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)

                # not implementing tcp for now for that Need to create stream at tcp.
                else:
                    _LOGGER.error("Only streams using udp protocol are being processed for now. Need a "
                                  "way to send stream via tcp. For more information refer to documentation")
                    raise NotImplementedError

            # not implementing other backend like gstreamer though opencv supports them as well.
            else:
                _LOGGER.error("Only ffmpeg is supported for now.")
                raise NotImplementedError

        else:
            if detect_mjpg_camera(source):
                # only mjpg  pixel format and coral camera are supported.
                self.stream = cv2.VideoCapture(source)
                _ = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                _ = self.stream.set(3, resolution[0])
                _ = self.stream.set(4, resolution[1])
            elif detectCoralDevBoard():
                self.stream = cv2.VideoCapture(source)
                # cannot change camera resolution on coral device through opencv API.
            else:
                # If the device is not supported self.stream would be None.
                # There could be another format such as YUYV which may support changing camera resolution
                # through Open CV API. Omitting these calls for now.
                self.stream = cv2.VideoCapture(source)

        if self.stream is None:
            _LOGGER.exception("Either the ID of video device is wrong or the device is not supported. Please "
                              "shut the plugin and try again with a correct device id.")
        else:
            _LOGGER.debug("Camera stream initialized")

        self.enable_thread = enable_thread
        if self.enable_thread:
            # Read first frame from the stream
            # No need to read the first frame here. Read inside thread.
            # See FOGL-4132 for details.
            self.frame = None
            self.grabbed = False

            # Variable to control when the camera is stopped
            self.stopped = False
        else:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                _LOGGER.exception("The device is not functional!. Please shutdown the plugin and try again"
                                  " with either a different camera ID or reconnect the device again. ")
                return
            else:
                _LOGGER.debug("Able to capture video stream from camera.")

    def start(self):
        if self.enable_thread:
            # Start the thread that reads frames from the video stream
            t = Thread(target=self.update, args=(), name="Reader Thread")
            t.daemon = True
            t.start()
        return self, self.grabbed

    def update(self):
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                # Release camera resources
                self.stream.release()
                return

            # Otherwise, grab the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                _LOGGER.exception("Either the ID of video device is wrong or the device is not functional!")
                return

    def read(self):
        # Return the most recent frame
        if not self.enable_thread:
            _, self.frame = self.stream.read()
        return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        if self.enable_thread:
            self.stopped = True
            # stream will be released in update def (which is looping indefinitely until the thread is stopped)
        else:
            self.stream.release()
