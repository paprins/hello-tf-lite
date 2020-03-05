import os
import re
import time
from multiprocessing import Process
from multiprocessing import Queue
import click
from PIL import Image
import yaml
import cv2

import edgetpu.detection.engine
from edgetpu.utils import image_processing

# from smartcam.ipcam import ipCamera

class Labelizer(object):

    def __init__(self, path):
        self.labels = [None] * 10
        self.path = path

    def get_labels(self):

        # if not self.labels:
        p = re.compile(r'\s*(\d+)(.+)')

        with open(self.path, 'r', encoding='utf-8') as f:
            lines = (p.match(line).groups() for line in f.readlines())

        self.labels = {int(num): text.strip() for num, text in lines}

        return self.labels

class Classifier(object):
    def __init__(self, model, input, output):
        self.model = model
        self.input = input
        self.output = output

    def enqueue(self, img):
        if self.input.empty():
            self.input.put(img)

    def dequeue(self):
        if not self.output.empty():
            return self.output.get()

        return None
    
    def classify(self, image):
        click.echo('Classify things ...')
        engine = edgetpu.detection.engine.DetectionEngine(self.model)

        while True:
            if not self.input.empty():
                # grab frame from input queue
                img = self.input.get()
                results = engine.detect_with_image(
                    img, 
                    threshold=0.4,
                    keep_aspect_ratio=True, 
                    relative_coord=False, 
                    top_k=10
                )

                data_out = []

                if results:
                    for obj in results:
                        inference = []
                        box = obj.bounding_box.flatten().tolist()
                        xmin = int(box[0])
                        ymin = int(box[1])
                        xmax = int(box[2])
                        ymax = int(box[3])

                        inference.extend((obj.label_id,obj.score,xmin,ymin,xmax,ymax))
                        data_out.append(inference)

                self.output.put(data_out)

@click.group()
@click.option('-c', '--config', type=click.Path(), help='Path to config file.')
@click.pass_context
def main(ctx, config):
    if config:
        try:
            with open(config, 'r', encoding='utf-8') as f:
                _config = yaml.load(f, Loader=yaml.FullLoader)

            ctx.obj = dict(
                c = _config,
                basedir = os.path.dirname(os.path.abspath(config))
            )
        except ImportError:
            click.echo('Error loading config')

@main.command()
@click.pass_context
@click.option('--cam', help='rtsp endpoint for ipcamera', default='0', required=False)
@click.option('--threshold', help='Confidence threshold (default: 0.6)', default=0.6)
@click.option('--rotate', help='Rotate camera image', type=click.Choice(['0','1','2']), required=False)
def detect(ctx, cam, threshold, rotate):
    config = ctx.obj['c']

    cap = cv2.VideoCapture(int(cam) if cam.isdigit() else cam)

    click.echo(f"> Capturing video {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} at {round(cap.get(cv2.CAP_PROP_FPS),2)}fps")

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    labelizr = Labelizer(
        path = os.path.join(ctx.obj['basedir'], config['labels'])
    )

    labels = labelizr.get_labels()

    classifier = Classifier(
        model  = os.path.join(ctx.obj['basedir'], config['model']),
        input  = Queue(maxsize=1),
        output = Queue(maxsize=1)
    )

    frames = 0
    queuepulls = 0
    font = cv2.FONT_HERSHEY_SIMPLEX
    detections = 0
    fps = 0.0
    qfps = 0.0
    img = None
    timer1 = time.time()
    timer2 = 0
    t2secs = 0

    p = Process(target=classifier.classify, args=(img,))
    p.daemon = True
    p.start()

    while (True):
        ret, frame = cap.read()

        if ret:
            if queuepulls ==1:
                timer2 = time.time()

            if rotate:
                #
                # 0 = cv::ROTATE_90_CLOCKWISE
                # 1 = cv::ROTATE_180 = 1,
                # 2 = cv::ROTATE_90_COUNTERCLOCKWISE
                #
                # TODO: would like to see a better solution ...
                #
                frame = cv2.rotate(frame, int(rotate))

            img = Image.fromarray(frame)

            classifier.enqueue(img)

            # image has been classified ... automagically ;)
            out = classifier.dequeue()

            if out:
                for detection in out:
                    objId = detection[0]
                    labeltxt = labels[objId]
                    confidence = detection[1]
                    xmin = detection[2]
                    ymin = detection[3]
                    xmax = detection[4]
                    ymax = detection[5]

                    if confidence > threshold:
                        #bounding box
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color=(0, 255, 255))
                        #label
                        labLen = len(labeltxt)*5+40
                        cv2.rectangle(frame, (xmin-1, ymin-1), (xmin+labLen, ymin-10), (0,255,255), -1)
                        #labeltext
                        cv2.putText(frame,' ' + labeltxt + ' ' + str(round(confidence,2)), (xmin,ymin-2), font, 0.3,(0,0,0),1,cv2.LINE_AA)
                        detections +=1 #positive detections

                queuepulls += 1


            # Display the resulting frame
            cv2.rectangle(frame, (0,0), (frameWidth,20), (0,0,0), -1)

            cv2.rectangle(frame, (0,frameHeight-20), (frameWidth,frameHeight), (0,0,0), -1)
            cv2.putText(frame,'Threshold: '+str(round(threshold,1)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame,'VID FPS: '+str(fps), (frameWidth-80, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame,'TPU FPS: '+str(qfps), (frameWidth-80, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame,'Positive detections: '+str(detections), (10, frameHeight-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

            cv2.putText(frame,'Elapsed time: '+str(round(t2secs,2)), (150, frameHeight-10), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0, 255, 255), 1, cv2.LINE_AA)


            cv2.namedWindow('Coral.ai',cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Coral.ai',frameWidth,frameHeight)
            cv2.imshow('Coral.ai',frame)

            # FPS calculation
            frames += 1
            if frames >= 1:
                end1 = time.time()
                t1secs = end1-timer1
                fps = round(frames/t1secs,2)
            if queuepulls > 1:
                end2 = time.time()
                t2secs = end2-timer2
                qfps = round(queuepulls/t2secs,2)

            # keypress?
            keypress = cv2.waitKey(5)

            if keypress == 113:    # Press Q to exit
                break
            elif keypress == -1:
                continue
            elif keypress == 44:
                threshold = min(round(threshold + 0.1, 2), 1.0)
            elif keypress == 45:
                threshold = max(round(threshold - 0.1, 2), 0.0)
        else:
            break
        
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()