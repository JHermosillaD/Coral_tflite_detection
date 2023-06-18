import tflite_runtime.interpreter as tflite
import tensorflow as tf
import numpy as np
import cv2
import os

class pycoral_detector: 

  def __init__(self):
    self.cwd = os.getcwd()
    self.model = f'{self.cwd}/model/my_face.tflite'
    self.interpreter = tflite.Interpreter(self.model, experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
    self.interpreter.allocate_tensors()

  def pre_process(self, img_rgb, input_size):
    img_rgb = cv2.resize(img_rgb, (640, 640), interpolation= cv2.INTER_LINEAR)
    img_tensor = tf.convert_to_tensor(img_rgb, dtype=tf.uint8)
    original_image = img_tensor
    resized_img = tf.image.resize(img_tensor, input_size)
    resized_img = resized_img[tf.newaxis, :]
    resized_img = tf.cast(resized_img, dtype=tf.uint8)
    return resized_img, original_image 

  def plot_result(self, results, original_image):
    original_image_np = original_image.numpy().astype(np.uint8)
    for obj in results:
       ymin, xmin, ymax, xmax = obj['bounding_box']
       xmin = int(xmin * original_image_np.shape[1])
       xmax = int(xmax * original_image_np.shape[1])
       ymin = int(ymin * original_image_np.shape[0])
       ymax = int(ymax * original_image_np.shape[0])
       class_id = int(obj['class_id'])
       cv2.rectangle(original_image_np, (xmin, ymin), (xmax, ymax), (0,0,255), 2)
       label = "{}: {:.0f}%".format(class_id, obj['score']*100)
       cv2.putText(original_image_np, label, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    return original_image_np.astype(np.uint8)
  
  def run_detection(self, img_rgb, threshold=0.5):
    _, input_height, input_width, _ = self.interpreter.get_input_details()[0]['shape']
    preprocessed_image, original_image = self.pre_process(img_rgb, (input_height, input_width))
    signature_fn = self.interpreter.get_signature_runner()
    output = signature_fn(images=preprocessed_image)
    count = int(np.squeeze(output['output_0']))
    scores = np.squeeze(output['output_1'])
    classes = np.squeeze(output['output_2'])
    boxes = np.squeeze(output['output_3'])
    results = []
    for i in range(count):
       if scores[i] >= threshold:
         result = {'bounding_box': boxes[i], 'class_id': classes[i], 'score': scores[i]}
         results.append(result)
    
    return results, original_image
  
def main():
  pydetection = pycoral_detector()
  vid = cv2.VideoCapture(0)
  while(True):
    _, frame = vid.read()
    pydetection.interpreter.invoke()
    results, original_image = pydetection.run_detection(frame)
    detection_result_image = pydetection.plot_result(results, original_image)
    cv2.imshow('frame', detection_result_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
if __name__ == '__main__':
    main()
