
import cv2
import logging
from threading import Thread
from fledge.common import logger
import subprocess


def detectCoralDevBoard():

    try:
        if 'MX8MQ' in open('/sys/firmware/devicetree/base/model').read():
            print('Detected Edge TPU dev board.')
            return True

    except:
        pass

    return False


_LOGGER = logger.setup(__name__, level=logging.INFO)
"""
Helper class for starting a separate thread for reading frame from the Camera 
Device

"""


def detect_mjpg_camera(source):

    out = subprocess.Popen(['v4l2-ctl', '--list-formats-ext', '--device', '/dev/video'.join(str(source))],
                           stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT)
    stdout, stderr = out.communicate()
    if str(stdout).find("MJPG") != -1:
        return True
    else:
        return False


class VideoStream:
    """ Camera object that controls
        video streaming from the Camera
    """

    def __init__(self, resolution=(640, 480), framerate=30, source=0, enable_thread=False):
        # Initialize the PiCamera and the camera image stream

        if detect_mjpg_camera(source):
            # only mjpg  pixel format and coral camera are supported.
            self.stream = cv2.VideoCapture(source)
            _ = self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            _ = self.stream.set(3, resolution[0])
            _ = self.stream.set(4, resolution[1])

        elif detectCoralDevBoard():
            # Fix for FOGL-4148
            self.stream = cv2.VideoCapture(source)

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
                _LOGGER.exception("Either the id of video device is wrong or the device not functional")
                return

    def start(self):

        if self.enable_thread:
            # Start the thread that reads frames from the video stream
            t = Thread(target=self.update, args=(), name="Reader Thread")
            t.daemon = True
            t.start()

        return self

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
                _LOGGER.exception("Either the id of video device is wrong or the device not functional")
                return

    def read(self):
        # Return the most recent frame
        if self.enable_thread:
            return self.frame
        else:
            _, self.frame = self.stream.read()
            return self.frame

    def stop(self):
        # Indicate that the camera and thread should be stopped
        if self.enable_thread:
            self.stopped = True
        else:
            self.stream.release()
