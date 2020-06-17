

# FLEDGE_BEGIN
# See: http://fledge.readthedocs.io/
# FLEDGE_END

""" Human Detector  async plugin for any platform """
import copy
import asyncio
import uuid
import logging
import os
import numpy as np
from threading import Thread
from fledge.common import logger
from fledge.plugins.common import utils
import async_ingest
import cv2
import time
import subprocess
from fledge.plugins.south.person_detection.videostream import VideoStream
from fledge.plugins.south.person_detection.inference import Inference
import asyncio
from aiohttp import web, MultipartWriter
import  threading
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


__author__ = "Amandeep Singh Arora, Deepanshu Yadav"
__copyright__ = "Copyright (c) 2020 Dianomic Systems Inc."
__license__ = "Apache 2.0"
__version__ = "${VERSION}"
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
loop = None
async_thread = None
camera_processing_thread = None
shutdown_in_progress = None
inference = None
asset_name = None
enable_window = True
enable_web_streaming = True
FRAME = None
web_streaming_port_no = None


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
    global shutdown_in_progress
    shutdown_in_progress = False
    global BACKGROUND_TASK
    BACKGROUND_TASK = check_background()
    data = copy.deepcopy(config)
    return data


def wait_for_frame(stream):
    """ Waits for frame to become available else sleeps for 200 milliseconds.
            Args:
                   x -> a videostream object
            Returns: None
           Raises: None
       """
    while True:
        if stream.frame is not None:
            return
        else:
            time.sleep(0.2)


def round_to_three_decimal_places(x):
    """ Rounds off a floating number to three decimal places
         Args: 
                x -> a floating number 
         Returns:
                returns a floating point number rounded off to three decimal places
        Raises: None
    """

    return round(x, 3)


async def mjpeg_handler(request):
    """    Keeps a watch on a global variable FRAME , encodes FRAME into jpeg and returns a response suitable
           for viewing in a browser.
            Args:
                   request -> a http request
            Returns:
                Returns a response which contains a jpeg compressed image to view on browser.
           Raises: None
    """
    boundary = "boundarydonotcross"
    response = web.StreamResponse(status=200, reason='OK', headers={
        'Content-Type': 'multipart/x-mixed-replace; '
                        'boundary=--%s' % boundary,
    })
    await response.prepare(request)

    global FRAME, shutdown_in_progress
    encode_param = (int(cv2.IMWRITE_JPEG_QUALITY), 90)

    while True:

        if shutdown_in_progress:
            break
        if FRAME is None:
            continue

        result, encimg = cv2.imencode('.jpg', FRAME, encode_param)
        data = encimg.tostring()
        await response.write(
            '--{}\r\n'.format(boundary).encode('utf-8'))
        await response.write(b'Content-Type: image/jpeg\r\n')
        await response.write('Content-Length: {}\r\n'.format(
            len(data)).encode('utf-8'))
        await response.write(b"\r\n")
        # Write data
        await response.write(data)
        await response.write(b"\r\n")
        await response.drain()

    return response


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

    global asset_name
    reads = {}
    for r_index in range(len(objs)):
        reads['person_' + str(r_index + 1) + '_' + 'label'] = objs[r_index]['label']
        reads['person_' + str(r_index + 1) + '_' + 'score'] = objs[r_index]['score']
        reads['person_' + str(r_index + 1) + '_' + 'x1'] = objs[r_index]['bounding_box'][0]
        reads['person_' + str(r_index + 1) + '_' + 'y1'] = objs[r_index]['bounding_box'][1]
        reads['person_' + str(r_index + 1) + '_' + 'x2'] = objs[r_index]['bounding_box'][2]
        reads['person_' + str(r_index + 1) + '_' + 'y2'] = objs[r_index]['bounding_box'][3]

    reads['count'] = len(objs)
    data = {
        'asset': asset_name,
        'timestamp': utils.local_timestamp(),
        'readings': reads
    }
    return data


def camera_loop(**kwargs):
    """ Main function that keeps on fetching frame , 
        performing inference and drawing the result of
        inference on the detection window.
         Args:
           Keyword Arguements -> Each one is listed below
         Returns:
           None 
         Raises:  
           Raises exception if unable to draw the frame on the the window.
    """

    # if floating point model is used we need to subtract the mean and divide 
    # by standard deviation
    input_mean = kwargs['input_mean']
    input_std = kwargs['input_std']

    # the height of the detection window on which frames are to be displayed
    camera_height = kwargs['camera_height']

    # the width of the detection window on which frames are to be displayed
    camera_width = kwargs['camera_width']

    # Note : The object contents may change if plugin_reconfigure is called 
    # thats why the inference object in the loop is global

    # these variables are used for calculation of frame per seconds (FPS)
    frame_rate_calc = 1
    freq = cv2.getTickFrequency()

    source = kwargs['source']

    enable_window_local = kwargs['enable_window']

    # Initialize the stream object and start the thread that keeps on reading frames
    # This thread is independent of the Camera Processing Thread
    videostream = VideoStream(resolution=(camera_width, camera_height), source=source).start()
    # For using the videostream with threading use the following :
    # videostream = VideoStream(resolution=(camera_width, camera_height), source=source, enable_thread=True).start()

    # The thread is allowed to capture a few frames. See FOGL-4132 for details
    wait_for_frame(videostream)

    # creating a window with a name
    window_name = 'Human detector'
    global BACKGROUND_TASK
    if not BACKGROUND_TASK and enable_window_local:
        foreground_task = True
        cv2.namedWindow(window_name)
    else:
        foreground_task = False
   
    while True:
        # Capture frame-by-frame
        t1 = cv2.getTickCount()
        global inference, asset_name, FRAME, enable_window

        # we need the height , width to resize the image for feeding into the model
        height_for_model = inference.height_for_model
        width_for_model = inference.width_for_model

        #  check if floating point model is used or not
        floating_model = inference.floating_model

        # The minimum confidence to threshold the detections obtained from model
        min_conf_threshold = inference.min_conf_threshold

        # The list of labels of the supported objects detected by the plugin
        labels = inference.labels

        # Taking the frame the stream  
        frame1 = videostream.read()

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
            input_data = (np.float32(input_data) - input_mean) / input_std

        # Perform the actual detection by running the model with the image as input
        boxes, classes, scores = inference.perform_inference(input_data)

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
            if (scores[i] > min_conf_threshold) and (int(classes[i] == 0)):
                
                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, 
                # need to force them to be within image using max() and min()

                ymin_model = round_to_three_decimal_places(boxes[i][0])
                xmin_model = round_to_three_decimal_places(boxes[i][1])
                ymax_model = round_to_three_decimal_places(boxes[i][2])
                xmax_model = round_to_three_decimal_places(boxes[i][3])

                # map the bounding boxes from model to the window 
                ymin = int(max(1, (ymin_model * camera_width)))
                xmin = int(max(1, (xmin_model * camera_height)))
                ymax = int(min(camera_width, (ymax_model * camera_width)))
                xmax = int(min(camera_height, ( xmax_model * camera_height)))

                # draw the rectangle on the frame
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

                # Contructing the label

                # Look up object name from "labels" array using class index
                object_name = labels[int(classes[i])] 

                # Example: 'person: 72%'
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) 

                # Get font size
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                                      0.7, 2)
                
                # Make sure not to draw label too close to top of window
                label_ymin = max(ymin, labelSize[1] + 10) 

                # Draw white box to put label text in
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), 
                              (xmin+labelSize[0], label_ymin+baseLine-10), 
                              (255, 255, 255), cv2.FILLED)

                # Draw the text label 
                cv2.putText(frame, label, (xmin, label_ymin-7), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) 
                
                # the readings to be inserted into the table
                objs.append({'label': labels[classes[i]],
                            'score': 100*scores[i],
                            'bounding_box': [xmin, ymin, xmax, ymax]
                            })

        # Draw framerate in corner of frame
        cv2.putText(frame, 'FPS: {0:.2f}'.format(frame_rate_calc),
                    (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0)
                    , 2, cv2.LINE_AA)
        
        # All the results have been drawn on the frame, so it's time to display it.
        global shutdown_in_progress
        if shutdown_in_progress:
            videostream.stop()
            time.sleep(3)
            cv2.destroyWindow(window_name)
            break
        else:

            # Calculate framerate
            t_end = cv2.getTickCount()
            time1 = (t_end-t1)/freq
            frame_rate_calc = 1/time1   

            data = construct_readings(objs)
            if not shutdown_in_progress:
                async_ingest.ingest_callback(c_callback, c_ingest_ref, data)

            # show the frame on the window
            try:
                if foreground_task and enable_window:
                    cv2.imshow(window_name, frame)
                FRAME = frame.copy()
            except Exception as e:
                _LOGGER.info('exception  {}'.format(e))

            # wait for 1 milli second 
            cv2.waitKey(1)


async def index(request):
    return web.Response(text='<img src="/image"/>', content_type='text/html')


def start_web_streaming_server(local_loop, address, port):
    """ Starts a server to display detection results in a browser.
            Args:
                   local_loop-> An asyncio main loop for server.
                   address -> ip address where server to be started. Only localhost is used , so '0.0.0.0'  is used.
                   port -> The port where the server application should run. It is configurable.
            Returns:
                None
           Raises: None
    """

    app = web.Application(loop=local_loop)
    app.router.add_route('GET', "/", index)
    app.router.add_route('GET', "/image", mjpeg_handler)
    coro_server = local_loop.create_server(app.make_handler(loop=local_loop), address, port)
    asyncio.ensure_future(coro_server, loop=local_loop)


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
    global loop, async_thread, camera_processing_thread, shutdown_in_progress

    loop = asyncio.new_event_loop()
    try:
        model = handle['model_file']['value']
        labels = handle['labels_file']['value']
        global asset_name
        asset_name = handle['asset_name']['value']
        enable_tpu = handle['enable_edge_tpu']['value']
        min_conf_threshold = float(handle['min_conf_threshold']['value'])

        model = os.path.join(os.path.dirname(__file__), "model", model)
        labels = os.path.join(os.path.dirname(__file__), "model", labels)

        with open(labels, 'r') as f:
            pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
            labels = dict((int(k), v) for k, v in pairs)

        # instance of the inference class 
        global inference, enable_window, enable_web_streaming, web_streaming_port_no
        inference = Inference()
        interpreter = inference.get_interpreter(model, enable_tpu, 
                                                labels, min_conf_threshold)
        
        # some extra config parameters required for the camera_loop function

        # Since pixel value can be from 0 to 255 , so we are considering mean to
        # be (0+255)/2 = 127.5 .
        input_mean = 127.5
        input_std = 127.5
        camera_height = 640
        camera_width = 480
        
        web_streaming_port_no = int(handle['web_streaming_port_no']['value'])

        if handle['enable_window']['value'] == 'true':
            enable_window = True
        else:
            enable_window = False

        if handle['enable_web_streaming']['value'] == 'true':
            enable_web_streaming = True
        else:
            enable_web_streaming = False

        source = int(handle['camera_id']['value'])

        config_dict = {
            'input_mean': input_mean,
            'input_std': input_std,
            'camera_height': camera_height,
            'camera_width': camera_width,
            'source': source,
            'enable_window': enable_window
        }

        def run():
            global loop
            loop.run_forever()

        if enable_web_streaming:
            start_web_streaming_server(loop, address='0.0.0.0', port=web_streaming_port_no)
            async_thread = Thread(target=run, name="Async Thread")
            async_thread.daemon = True
            async_thread.start()

        camera_processing_thread = Thread(target=camera_loop, 
                                          name='Camera Processing Thread',
                                          kwargs=config_dict)

        # Resolves segmentation fault  when fledge service shutdown
        camera_processing_thread.daemon = True
        camera_processing_thread.start()

    except Exception as exptn:
        _LOGGER.exception("Human detector plugin  failed to start. Details: %s", str(exptn))
        raise


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
    
    new_handle = plugin_init(new_config)

    model = new_handle['model_file']['value']
    model = os.path.join(os.path.dirname(__file__), "model", model)

    labels = new_handle['labels_file']['value']
    labels = os.path.join(os.path.dirname(__file__), "model", labels)
    with open(labels, 'r') as f:
        pairs = (l.strip().split(maxsplit=1) for l in f.readlines())
        labels = dict((int(k), v) for k, v in pairs)

    enable_tpu = new_handle['enable_edge_tpu']['value']
    min_conf_threshold = float(new_handle['min_conf_threshold']['value'])

    global inference, asset_name
    asset_name = new_handle['asset_name']['value']
    _ = inference.get_interpreter(model, enable_tpu, labels, min_conf_threshold)

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
    global camera_processing_thread, shutdown_in_progress, loop, async_thread
    try:
        shutdown_in_progress = True
        # allow the stream to stop
        time.sleep(2)
        
        # stopping  every other thread one by one
        camera_processing_thread = None
        loop.stop()
        async_thread = None
        loop = None
        _LOGGER.info('Plugin has shutdown')

    except Exception as e:
        _LOGGER.exception(str(e))
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


