import numpy as np
import requests

class ipCamera(object):

    def __init__(self, cam=None):
        self.cam = cv2.VideoCapture(cam)

    def retrieve(self, url=None, user=None, passw=None, imgsize=(640,382)):
        '''
        Grab a picture form IP camera, over http protocol
        :param url: camera url
        :param user: user name
        :param passw: password
        :param imgsize:
        :return:
        '''
        frame = np.zeros((imgsize[0], imgsize[1], 3), dtype=np.uint8)

        try:
            if self.cam:
                _, img = self.cam.read()
                if type(img):
                    frame = img

            else:
                img = requests.get(str(url), auth=HTTPBasicAuth(str(usr), str(passw))).content

                imgNp = np.array(bytearray(img), dtype=np.uint8)
                frame = cv2.imdecode(imgNp, -1)
        except:
            print('Could not connect to camera')

        frame = cv2.resize(frame, imgsize)

        return frame