import io
import time
import json
import click
import numpy as np
import picamera
from PIL import Image
from tflite import __version__
from tflite_runtime.interpreter import Interpreter

@click.group()
@click.version_option(version=__version__)
@click.pass_context
def main(ctx):
    print('hello')

def load_labels(path):
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def set_input_tensor(interpreter, image):
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

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

@main.command()
@click.option('-m', '--model', type=click.Path(), help='File path of .tflite file.', required=True)
@click.option('-l', '--labels', type=click.Path(), help='File path of labels file.', required=True)
@click.pass_context
def classify(ctx, model, labels):
    labels = load_labels(labels)

    interpreter = Interpreter(model)
    interpreter.allocate_tensors()

    _, height, width, _ = interpreter.get_input_details()[0]['shape']

    with picamera.PiCamera(resolution=(640, 480), framerate=30) as camera:
        camera.start_preview()
        try:
            stream = io.BytesIO()
            for _ in camera.capture_continuous(stream, format='jpeg', use_video_port=True):
                stream.seek(0)
                image = Image.open(stream).convert('RGB').resize(width, height), Image.ANTIALIAS)

                start_time = time.time()
                results = classify_image(interpreter, image)
                elapsed_ms = (time.time() - start_time) * 1000
                label_id, prob = results[0]

                stream.seek(0)
                stream.truncate()
                camera.annotate_text = f'{labels[label_id]} {prob}\n{elapsed_ms}ms'
        finally:
            camera.stop_preview()

if __name__ == '__main__':
  main()