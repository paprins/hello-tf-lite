import re
import numpy as np
import tflite_runtime.interpreter as tflite
from PIL import Image

class Classifier:

    def __init__(self, model, labels, input, output):
        self.model = model
        self.labels = labels
        self.input = input
        self.output = output

        if self.labels:
            out = dict()
            for row in open(self.labels):
                (classID, label) = row.strip().split(maxsplit=1)
                out[int(classID)] = label.strip()

        self.labels = out

        self.interpreter = tflite.Interpreter(
            model_path = self.model,
            experimental_delegates=[
                tflite.load_delegate('libedgetpu.1.dylib')
            ]
        )
        self.interpreter.allocate_tensors()

    def get_label(self, obj_id):
        return self.labels[obj_id]

    def __is_floating_model(self):
        """Check if model is quantized"""
        input_details = self.interpreter.get_input_details()
        return (input_details[0]['dtype'] == np.float32)

    def __set_input_tensor(self, image):
        tensor_index = self.interpreter.get_input_details()[0]['index']
        input_tensor = self.interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def __get_output_tensor(self, index):
        """Returns the output tensor at the given index."""
        output_details = self.interpreter.get_output_details()[index]
        tensor = np.squeeze(self.interpreter.get_tensor(output_details['index']))
        return tensor

    def enqueue(self, img):
        if self.input.empty():
            self.input.put(img)

    def dequeue(self):
        if not self.output.empty():
            return self.output.get()

        return None

    def __input_size(self):
        """Returns input image size as (width, height) tuple."""
        _, height, width, _ = self.interpreter.get_input_details()[0]['shape']
        return width, height

    def classify(self, threshold=0.0):
        while True:
            if not self.input.empty():
                # img = self.input.get()
                size = self.__input_size()
                img = self.input.get()
                image = img.resize(size, Image.ANTIALIAS)
                self.__set_input_tensor(image)

                self.interpreter.invoke()

                # Get all output details
                boxes = self.__get_output_tensor(0)
                classes = self.__get_output_tensor(1)
                scores = self.__get_output_tensor(2)
                count = int(self.__get_output_tensor(3))

                data_out = []
                for i in range(count):
                    if scores[i] >= threshold:
                        result = {
                            'box': boxes[i],
                            'label': self.get_label(classes[i]),
                            'score': scores[i]
                        }
                        data_out.append(result)

                self.output.put(data_out)
