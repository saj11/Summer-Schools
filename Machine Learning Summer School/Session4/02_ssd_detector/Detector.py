import numpy as np
import tensorflow as tf
import time
import cv2

from utils import label_map_util
from utils import visualization_utils_color as vis_util

class Detector:
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = './utils/protos/face_label_map.pbtxt'

    NUM_CLASSES = 2

    def __init__(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.config = tf.ConfigProto()
            self.config.gpu_options.allow_growth = True
            self.sess=tf.Session(graph=self.detection_graph, config=self.config)

    def detect(self, img, _debug=False, detect_threshold=0.2, expand=0.1):
        image_np = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #image_np=img
        with self.detection_graph.as_default():
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
            
            # Actual detection.
            start_time = time.time()
            (boxes, scores, classes, num_detections) = self.sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            elapsed_time = time.time() - start_time
            #if _debug: print('inference time cost: {}'.format(elapsed_time))

#             if _debug:
#                 img2=img.copy()
#                 vis_util.visualize_boxes_and_labels_on_image_array(
#                     img2,
#                     np.squeeze(boxes),
#                     np.squeeze(classes).astype(np.int32),
#                     np.squeeze(scores),
#                     self.category_index,
#                     use_normalized_coordinates=True,
#                     line_thickness=4,
#                     min_score_thresh=detect_threshold)
#                 cv2.imshow("Detected Face(s)", cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
                #cv2.waitKey(1)

            imgHeight,imgWidth=img.shape[:2]
            res2=[]

            for s, (ymin, xmin, ymax, xmax) in zip(np.squeeze(scores), np.squeeze(boxes)):
                if s > detect_threshold:
                    w=xmax - xmin
                    h=ymax - ymin
                    xmin=max(xmin-w*expand/2,0)
                    ymin=max(ymin-h*expand/2,0)
                    xmax=min(xmax+w*expand/2,imgWidth)
                    ymax=min(ymax+h*expand/2,imgHeight)
                    res2.append((int(xmin*imgWidth), int(ymin*imgHeight), int((xmax - xmin)*imgWidth), int((ymax - ymin)*imgHeight)))
            return res2
            res = [(int(xmin*imgWidth), int(ymin*imgHeight), int((xmax - xmin)*imgWidth), int((ymax - ymin)*imgHeight)) for s, (ymin, xmin, ymax, xmax) in
                    zip(np.squeeze(scores), np.squeeze(boxes)) if s > detect_threshold]
            return res


if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    d=Detector()
    while 1:
        ret, image = cap.read()
        if (ret == 0):
            break
        d.detect(image,True)


# # Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
#
# # List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = './protos/face_label_map.pbtxt'
#
# NUM_CLASSES = 2
#
# # label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# # categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# # category_index = label_map_util.create_category_index(categories)
#
# cap = cv2.VideoCapture(0)
#
# detection_graph = tf.Graph()
# with detection_graph.as_default():
#     od_graph_def = tf.GraphDef()
#     with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
#         serialized_graph = fid.read()
#         od_graph_def.ParseFromString(serialized_graph)
#         tf.import_graph_def(od_graph_def, name='')
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth=True
#     with tf.Session(graph=detection_graph, config=config) as sess:
#         while 1:
#             ret, image = cap.read()
#             if(ret ==0):
#                 break
#
#             image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
#             # the array based representation of the image will be used later in order to prepare the
#             # result image with boxes and labels on it.
#             # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#             image_np_expanded = np.expand_dims(image_np, axis=0)
#             image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#             # Each box represents a part of the image where a particular object was detected.
#             boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#             # Each score represent how level of confidence for each of the objects.
#             # Score is shown on the result image, together with the class label.
#             scores = detection_graph.get_tensor_by_name('detection_scores:0')
#             classes = detection_graph.get_tensor_by_name('detection_classes:0')
#             num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#             # Actual detection.
#             start_time = time.time()
#             (boxes, scores, classes, num_detections) = sess.run(
#               [boxes, scores, classes, num_detections],
#               feed_dict={image_tensor: image_np_expanded})
#             elapsed_time = time.time() - start_time
#             print('inference time cost: {}'.format(elapsed_time))
#             # vis_util.visualize_boxes_and_labels_on_image_array(
#             #   image,
#             #   np.squeeze(boxes),
#             #   np.squeeze(classes).astype(np.int32),
#             #   np.squeeze(scores),
#             #   category_index,
#             #   use_normalized_coordinates=True,
#             #   line_thickness=4)
#             cv2.imshow("",image)
#             cv2.waitKey(1)
#     cap.release()
#     out.release()
