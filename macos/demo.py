import os
import click
import cv2
import json
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
def detect(config):
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
        p = Process(target=classifier.classify)
        p.daemon = True
        p.start()

        while True:
            ret, frame = cap.read()

            if ret:
                classifier.enqueue(Image.fromarray(frame))

                out = classifier.dequeue()

                if out:
                    for detection in out:
                        # box, class, score
                        for i in range(detection['box'].shape[0]):
                            # box = detection['box'][0, 1, :]
                            box = detection['box']
                            x0 = int(box[1] * frame.shape[1])
                            y0 = int(box[0] * frame.shape[0])
                            x1 = int(box[3] * frame.shape[1])
                            y1 = int(box[2] * frame.shape[0])
                            box = box.astype(np.int)
                            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
                            cv2.rectangle(frame, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
                            cv2.putText(frame,
                                detection['label'],
                                (x0, y0),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1,
                                (255, 255, 255),
                                2
                            )

                    # for i in range(boxes.shape[1]):
                    #     if scores[0, i] > 0.5:
                    #         box = boxes[0, i, :]
                    #         x0 = int(box[1] * img_org.shape[1])
                    #         y0 = int(box[0] * img_org.shape[0])
                    #         x1 = int(box[3] * img_org.shape[1])
                    #         y1 = int(box[2] * img_org.shape[0])
                    #         box = box.astype(np.int)
                    #         cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)
                    #         cv2.rectangle(img_org, (x0, y0), (x0 + 100, y0 - 30), (255, 0, 0), -1)
                    #         cv2.putText(img_org,
                    #             str(int(labels[0, i])),
                    #             (x0, y0),
                    #             cv2.FONT_HERSHEY_SIMPLEX,
                    #             1,
                    #             (255, 255, 255),
                    #             2)    

                cv2.namedWindow('coral.ai', cv2.WINDOW_NORMAL)
                cv2.resizeWindow('coral.ai', width, height)
                cv2.imshow('coral.ai', frame)

                keypress = cv2.waitKey(5)

                if keypress == 113: # Press Q to exit
                    break
                elif keypress == -1:
                    continue
            else:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()