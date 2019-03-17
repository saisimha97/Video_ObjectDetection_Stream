

import time
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, render_template, Response

from utils import label_map_util
from utils import visualization_utils_color as vis_util

app = Flask(__name__)


FlowModel = './model/frozen_inference_graph_face.pb'


LabelPath = './protos/face_label_map.pbtxt'


#As it is binary classification
NUM_CLASSES = 2

label_map = label_map_util.load_labelmap(LabelPath)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

class TensoflowFaceDector(object):
    def __init__(self, FlowModel):
        

        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(FlowModel, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')


        with self.detection_graph.as_default():
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(graph=self.detection_graph, config=config)
         


    def run(self, image):
       

        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

       
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
       
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
       
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('frame lag: {}'.format(elapsed_time))
# the predicted box and score of class
        return (boxes, scores, classes, num_detections)
    
@app.route('/')
def stream():
    return render_template('index.html')
    

def gen(vc):
    while True:
        
        
        frame = vc.capt()
     
        yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


#Streaming video on local webpage
@app.route('/video_feed')
def video_feed():
    return Response(gen(VC()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
       
 
# Whole Detection Starts here
class VC(object):
    
    def __init__(self):
        
        tDetector = TensoflowFaceDector(FlowModel)
    
    # capturing webcam frames
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, self.image = cap.read()
            if ret == 0:
                break
    
            [h, w] = self.image.shape[:2]
            print (h, w)
            self.image = cv2.flip(self.image, 1)
    
    # whole predicted values from a pre-trained model on COCO dataset included in standard TensorFlow  object detection API
            (boxes, scores, classes, num_detections) = tDetector.run(self.image)
    
    
    #appending boxes to image
            vis_util.visualize_boxes_and_labels_on_image_array(
                self.image,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            
            a = self.image
            self.img = a
    
       
    
            cv2.imshow("Presonal WebCam Stream (%d, %d)" % (w, h), self.image)
         
            
            k = cv2.waitKey(1) & 0xff
            if k == ord('q') or k == 27:
                break
            
    
        cap.release()
        
    def capt(self):
        
        ret, jpeg = cv2.imencode('.jpg', self.img)
        return jpeg.tobytes()

if __name__ == '__main__':
    app.run()