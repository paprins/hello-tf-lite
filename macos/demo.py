import os
import time
import json

import click
import cv2
import yaml
from multiprocessing import Process, Queue
from PIL import Image
import numpy as np

from thnktnk.classifier import Classifier

def load(config):
    with open(config, 'r', encoding='utf-8') as f:
        _config = yaml.load(f, Loader=yaml.FullLoader)

    return _config

@click.group()
def main():
    pass

@main.command()
@click.option('-c', '--config', type=click.Path(), help='Path to config file.', required=True)
@click.option('-t', '--threshold', default=0.6)
def detect(config, threshold):
    c = load(config)
    basedir = os.path.dirname(os.path.abspath(config))
    click.echo(basedir)
    click.echo(c['model'])
    model = os.path.join(basedir, c['model'])

    if not os.path.isfile(model):
        click.echo(f"Model {c['model']} not found")
        os.sys.exit(1)

    cap = cv2.VideoCapture(0)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    click.echo(f"> Capturing video {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} at {round(cap.get(cv2.CAP_PROP_FPS),2)}fps")

    classifier = Classifier(
        model  = model,
        labels = os.path.join(basedir, c['labels']),
        input  = Queue(maxsize=1),
        output = Queue(maxsize=1)
    )

    try:
        frames = 0
        queuepulls = 0
        font = cv2.FONT_HERSHEY_SIMPLEX
        fps = 0.0
        qfps = 0.0
        timer1 = time.time()
        timer2 = 0
        t2secs = 0

        p = Process(target=classifier.classify)
        p.daemon = True
        p.start()

        while True:
            ret, frame = cap.read()

            if ret:
                if queuepulls ==1:
                    timer2 = time.time()

                classifier.enqueue(Image.fromarray(frame))

                out = classifier.dequeue()

                if out:
                    for detection in out:
                        # box, class, score
                        if detection['score'] > threshold:
                            for i in range(detection['box'].shape[0]):
                                box = detection['box']
                                x0 = int(box[1] * frame.shape[1])
                                y0 = int(box[0] * frame.shape[0])
                                x1 = int(box[3] * frame.shape[1])
                                y1 = int(box[2] * frame.shape[0])
                                box = box.astype(np.int)

                                # bounding box
                                cv2.rectangle(frame, (x0, y0), (x1, y1), color=(0, 255, 255))

                                # label
                                labLen = len(detection['label']) * 5 + 40
                                cv2.rectangle(frame, (x0-1, y0-1), (x0+labLen, y0-10), (0,255,255), -1)

                                # labeltext
                                cv2.putText(frame,f" {detection['label']} {str(round(detection['score'],2))}", (x0,y0-2), font, 0.3,(0,0,0),1,cv2.LINE_AA)

                    queuepulls += 1

                cv2.rectangle(frame, (0,0), (width,20), (0,0,0), -1)
                cv2.rectangle(frame, (0,height-20), (width,height), (0,0,0), -1)

                cv2.putText(frame,'Threshold: '+str(round(threshold,1)), (10, 10), font, 0.3,(0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame,'VID FPS: '+str(fps), (width-80, 10), font, 0.3,(0, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(frame,'TPU FPS: '+str(qfps), (width-80, 20), font, 0.3,(0, 255, 255), 1, cv2.LINE_AA)

                cv2.namedWindow('coral.ai', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('coral.ai', width, height)
                cv2.imshow('coral.ai', frame)

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

                keypress = cv2.waitKey(5)

                if keypress == 113: # Press Q to exit
                    break
                elif keypress == -1:
                    continue
                elif keypress == 44:
                    threshold = min(round(threshold + 0.1, 2), 1.0)
                elif keypress == 45:
                    threshold = max(round(threshold - 0.1, 2), 0.0)
            else:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()