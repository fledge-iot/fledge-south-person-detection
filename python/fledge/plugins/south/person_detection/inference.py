# -*- coding: utf-8 -*-

# FLEDGE_BEGIN
# See: http://fledge.readthedocs.io/
# FLEDGE_END

""" Helper Class for loading the model and performing the inference on it 
"""

import logging
import os

import numpy as np

from fledge.common import logger

_LOGGER = logger.setup(__name__, level=logging.INFO)

EDGETPU_SHARED_LIB = 'libedgetpu.so.1'


try:
    from tflite_runtime.interpreter import Interpreter
    try:
        from tflite_runtime.interpreter import load_delegate
    except ImportError:
        _LOGGER.warning("Edge TPU support is not found in your tensorflow installation!")
        pass
except ImportError as e:
    _LOGGER.exception("Tensorflow installation not found.")


class Inference:
    def __init__(self):

        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.height_for_model = None
        self.width_for_model = None
        self.floating_model = None
        self.min_conf_threshold = None
        self.labels = None

    def get_interpreter(self, model, enable_tpu, labels, min_conf_threshold=0.5):
        """ Returns interpreter from the model
            Args:
                  model -> full path of the .tflite file ,
                  enable_tpu-> whether you want to use Edge TPU or not
        Returns:
                  interpreter object
        Raises:
                Raises exception if the Runtime Library for Edge TPU
                is not found

        """
        interpreter_loaded = False
        if enable_tpu == 'true':
            try:
                # loading the Edge TPU Runtime
                model, *device = model.split('@')
                if os.path.exists(model):
                    load_delegate(EDGETPU_SHARED_LIB)
                    self.interpreter = Interpreter(model_path=model,
                                                   experimental_delegates=[load_delegate(EDGETPU_SHARED_LIB,
                                                                         {'device': device[0]} if device else {})])
                    interpreter_loaded = True

                else:
                    _LOGGER.exception("Please make sure the model file exists")

            except OSError:
                 _LOGGER.exception("Please install runtime for edge tpu")

            except ValueError:
                _LOGGER.exception("Make sure edge tpu is plugged in")

        else:

            self.interpreter = Interpreter(model_path=model)
            interpreter_loaded = True

        if interpreter_loaded:
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            self.height_for_model = self.input_details[0]['shape'][1]
            self.width_for_model = self.input_details[0]['shape'][2]
            self.floating_model = (self.input_details[0]['dtype'] == np.float32)
            self.min_conf_threshold = min_conf_threshold
            self.labels = labels

        # return None if not loaded
        return self.interpreter

    def perform_inference(self, input_data):
        """ Returns bounding box , class , score
            Args:
                   input_data ->  the input data to be fed into the model
            Returns:
                     boxes -> an array of bounding box of the objects detected
                     classes-> an array of the class of objects detected
                     scores-> an array of scores(0 to 1) of the objects detected
            Raises: None
        """

        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        # Retrieve detection results

        # Bounding box coordinates of detected objects
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        # Class index of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]

        # Confidence of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        return boxes, classes, scores
