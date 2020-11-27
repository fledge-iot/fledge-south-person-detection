import asyncio
import logging
from aiohttp import web, MultipartWriter
import cv2
from fledge.common import logger
_LOGGER = logger.setup(__name__, level=logging.INFO)


class WebStream:
    FRAME = None
    SHUTDOWN_IN_PROGRESS = False

    def __init__(self, port=8085, address='0.0.0.0'):
        self.address = address
        self.port = port
        self.ws_app = None
        self.ws_handler = None
        self.ws_server = None

    @staticmethod
    async def mjpeg_handler(request):
        """    Keeps a watch on a class variable FRAME , encodes FRAME into jpeg and returns a response suitable
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
        encode_param = (int(cv2.IMWRITE_JPEG_QUALITY), 90)

        while True:

            if WebStream.SHUTDOWN_IN_PROGRESS:
                break

            if WebStream.FRAME is None:
                continue

            result, encimg = cv2.imencode('.jpg', WebStream.FRAME, encode_param)

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

    def stop_server(self, loop):
        """
                Stops the web stream server (called on plugin shutdown)
                Args:
                       loop-> a asyncio loop
                Returns: None
               Raises: None
        """
        try:
            if self.ws_server:
                self.ws_server.close()
                asyncio.ensure_future(self.ws_server.wait_closed(), loop=loop)
            asyncio.ensure_future(self.ws_app.shutdown(), loop=loop)
            asyncio.ensure_future(self.ws_handler.shutdown(2.0), loop=loop)
            asyncio.ensure_future(self.ws_app.cleanup(), loop=loop)

        except asyncio.CancelledError:
            pass
        except KeyError:
            pass  # We may want to check if enable web streaming is set?!

    @staticmethod
    async def index(request):
        return web.Response(text='<img src="/image"/>', content_type='text/html')

    def start_web_streaming_server(self, local_loop):
        """ Starts a server to display detection results in a browser.
                Args:
                       local_loop-> An asyncio main loop for server.

                Returns:
                    self
               Raises: None
        """
        app = web.Application(loop=local_loop)
        app.router.add_route('GET', "/", WebStream.index)
        app.router.add_route('GET', "/image", WebStream.mjpeg_handler)

        handler = app.make_handler(loop=local_loop)

        coro_server = local_loop.create_server(handler, self.address, self.port)

        f = asyncio.ensure_future(coro_server, loop=local_loop)

        self.ws_app = app
        self.ws_handler = handler
        self.ws_server = None

        def f_callback(f):
            _LOGGER.info(repr(f.result()))
            self.ws_server = f.result()

        f.add_done_callback(f_callback)

        return self
