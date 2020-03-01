import io
import re
import time
import json
import click
import numpy as np
import picamera
from annotation import Annotator
from PIL import Image
from tflite import __version__, Annotator

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

@click.group()
@click.version_option(version=__version__)
@click.pass_context
def main(ctx):
    print('hello')

def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels

def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    return tensor

def classify_image(interpreter, image, top_k=1):
    """Returns a sorted array of classification results."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = np.squeeze(interpreter.get_tensor(output_details['index']))

    # If the model is quantized (uint8 data), then dequantize the results
    if output_details['dtype'] == np.uint8:
        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

    ordered = np.argpartition(-output, top_k)
    return [(i, output[i]) for i in ordered[:top_k]]

def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    classes = get_output_tensor(interpreter, 1)
    scores = get_output_tensor(interpreter, 2)
    count = int(get_output_tensor(interpreter, 3))

    results = []
    for i in range(count):
    if scores[i] >= threshold:
        result = {
            'bounding_box': boxes[i],
            'class_id': classes[i],
            'score': scores[i]
        }
        results.append(result)
    return results

def annotate_objects(annotator, results, label):
    """Draws the bounding box and label for each object in the results."""
    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * CAMERA_WIDTH)
        xmax = int(xmax * CAMERA_WIDTH)
        ymin = int(ymin * CAMERA_HEIGHT)
        ymax = int(ymax * CAMERA_HEIGHT)

        # Overlay the box, label, and score on the camera preview
        annotator.bounding_box([xmin, ymin, xmax, ymax])
        annotator.text([xmin, ymin],
                    '%s\n%.2f' % (labels[obj['class_id']], obj['score']))

@main.command()
@click.option('-m', '--model', type=click.Path(), help='File path of .tflite file.', required=True)
@click.option('-l', '--labels', type=click.Path(), help='File path of labels file.', required=True)
@click.pass_context
def classify(ctx, model, labels):
    labels = load_labels(labels)

    # speed up inferencing by delegating to Edge TPU (~ Coral USB Accelerator)
    interpreter = Interpreter(model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    interpreter.allocate_tensors()

    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
        # flip camera
        camera.vflip = True
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB').resize((width, height), Image.ANTIALIAS)

                start_time = time.time()
                results = classify_image(interpreter, image)
                elapsed_ms = (time.time() - start_time) * 1000
                label_id, prob = results[0]

                stream.seek(0)
                stream.truncate()

                if prob >= 0.85:
                    # only annotate if probability >= 85%
                    camera.annotate_text = f'{labels[label_id]} {prob}\n{elapsed_ms}ms'
        finally:
            camera.stop_preview()

@main.command()
@click.option('-m', '--model', type=click.Path(), help='File path of .tflite file.', required=True)
@click.option('-l', '--labels', type=click.Path(), help='File path of labels file.', required=True)
@click.option('--threshold', type=float, help='Score threshold for detected objects.', default=0.85)
@click.pass_context
def detect(ctx, model, labels, threshold):
    click.echo(f'>> threshold: {threshold}')
    labels = load_labels(labels)

    # speed up inferencing by delegating to Edge TPU (~ Coral USB Accelerator)
    interpreter = Interpreter(model, experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    interpreter.allocate_tensors()

    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    with picamera.PiCamera(resolution=(CAMERA_WIDTH, CAMERA_HEIGHT), framerate=30) as camera:
        # flip camera
        camera.vflip = True
        camera.start_preview()
        try:
            stream = io.BytesIO()
            annotator = Annotator(camera)
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB').resize((width, height), Image.ANTIALIAS)

                start_time = time.monotonic()
                results = detect_objects(interpreter, image, threshold)
                elapsed_ms = (time.monotonic() - start_time) * 1000
                
                annotator.clear()
                annotate_objects(annotator, results, labels)
                annotator.text([5, 0], '%.1fms' % (elapsed_ms))
                annotator.update()

                stream.seek(0)
                stream.truncate()

        finally:
            camera.stop_preview()

if __name__ == '__main__':
  main()