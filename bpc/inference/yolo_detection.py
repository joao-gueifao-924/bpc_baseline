import cv2
import os
from ultralytics import YOLO
import numpy as np
from bpc.utils.data_utils import extract_roi_with_padding
import bpc.utils.data_utils as du
import bpc.inference.yolo_detection_filtering as ydf
import json

class YOLODetector:
    """
    YOLO detector, detections are upright bounding boxes with associated confidence.

    This class wraps a YOLO detection model and provides methods to detect objects in images.
    It uses the YOLOv11 model from the ultralytics package.
    """
    def __init__(self, yolo_model_path, yolo_detection_thresholds_path=None, obj_id=None):
        self.obj_id = obj_id # should be None for multi-class detector
        self.yolo = YOLO(yolo_model_path).cuda()
        self.yolo_confidence_thresh = 0.01
        self.image_size = 1280 # keep it as 1280, to be consistent with the YOLO model input size defined during training
        self.yolo_detection_thresholds = None

        if yolo_detection_thresholds_path is not None:
            with open(yolo_detection_thresholds_path, 'r') as f:
                str_keys_dict = json.load(f)
                self.yolo_detection_thresholds = {int(k): v for k, v in str_keys_dict.items()}

    def detect(self, image, single_class_detector=False, rescale_factor=1.0):
        """
        Run YOLO on a single image.
        Returns a list of detections.
        Each detection is a dictionary with keys "bbox", "bb_center", "confidence".
        """
        results = self.yolo(image, imgsz=self.image_size, verbose=True)[0] # only result #0 as it's single image inference
        boxes = results.boxes.xyxy.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()
        class_ids  = results.boxes.cls.cpu().numpy()
        class_names = results.names
        if len(results.boxes) == 0:
            return []
        
        valid = (confidences >= self.yolo_confidence_thresh)

        if single_class_detector:
            valid = valid & (class_ids == 0)  

        boxes = boxes[valid]
        confidences = confidences[valid]
        class_ids = [int(x) for x in class_ids[valid]]
        
        detections_by_class_id = {}

        should_rescale_output = np.abs(rescale_factor - 1.0) > 1e-6

        for box, confidence, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = map(int, box)
            if should_rescale_output:
                x1 = int(x1 * rescale_factor)
                y1 = int(y1 * rescale_factor) 
                x2 = int(x2 * rescale_factor)
                y2 = int(y2 * rescale_factor)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            
            if self.yolo_detection_thresholds is not None:
                if confidence < self.yolo_detection_thresholds[class_id]:
                    continue

            if class_id not in detections_by_class_id:
                detections_by_class_id[class_id] = []
            
            detections_by_class_id[class_id].append({
                'bbox': (x1, y1, x2, y2),
                'bb_center': (cx, cy),
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_names[class_id]
            })

        return detections_by_class_id


# TODO Fix YOLODetectorOrientedBoundingBox regarding subclassing from YOLODetector, and the coordinates of the bounding boxes mapping 
# to the original image, which is not correct.
# For now, keep it disabled.
if False:
    # This class is not used, but it is a good example of how to subclass the YOLODetector class to create a new detector.
    # It is not used because it is not working correctly, and it is not needed for the current task.
    # It is kept here for reference, in case we need to use it in the future.
    # The idea is to use this class to detect objects with oriented bounding boxes, using the YOLO detector.
    class YOLODetectorOrientedBoundingBox(YOLODetector):
        """
        YOLO detector with oriented bounding boxes.

        This class wraps a YOLO detection model that is configured to only detect upright bounding boxes, to handle objects that may appear at various orientations.
        It works by initially detecting objects on the original image and then, for each detection, it extracts a square crop centered
        around the object. The crop is then rotated at several angles, pasted into a blank image canvas, and the YOLO detector is applied on that canvas, once per each angle.
        Detections are filtered to retain only those that are well-centered in the rotated canvas to reject extraneous detections. 
        Finally, the most elongated bounding box is selected and translated back to the original image coordinates, and the highest confidence among all angles is kept.
        """
        def __init__(self, obj_id, yolo_model_path):
            self.obj_id = obj_id
            self.yolo = YOLO(yolo_model_path).cuda()
            self.yolo_conf_thresh = 0.1
            self.angle_step_deg = 360.0 / 16.0
            self.angle_values = np.linspace(0, 360, num=4, endpoint=True)[:-1].tolist() # Exclude 360 degrees last value, which equals 0 degrees
            # with num=4, we get 0, 90, 180, 270 degrees
            # with num=8, we get 0, 45, 90, 135, 180, 225, 270, 315 degrees
            # with num=16, we get 0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5 degrees

        def _detect_single_image(self, image):
            """
            Run YOLO on a single image.
            Returns a list of detections.
            Each detection is a dictionary with keys "bbox", "bb_center", "confidence".
            """
            results = self.yolo(image, imgsz=1280, verbose=True)[0]
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss  = results.boxes.cls.cpu().numpy()
            if len(results.boxes) == 0:
                return []
            # Keep only detections with class==0 and conf>=threshold.
            valid = (clss == 0) & (confs >= self.yolo_conf_thresh)

            boxes = boxes[valid]
            confidences = confs[valid]
            detections = []
            for box, confidence in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'bb_center': (cx, cy),
                    'confidence': confidence
                })
            return detections

        def _detect_with_rotations(self, image):
            """
            Detect objects in several rotations of the image.
            This function applies the YOLO detector to each rotated version of image and 
            returns the detections with the highest confidences amongst all rotations.

            Each detection is a dictionary with keys "bbox", "angle", "bb_center", "confidence".

            The following steps are performed, in order:
            1. Detections are made on the original image. 
            2. For each detection, a square crop is made around the detection's center. The size of the crop is determined by the maximum of the width and height of the detection's bounding box.
            3. For each rotation angle:
                3.1 a grey image canvas is created,
                3.2 the cropped image is rotated by that angle and pasted onto the canvas, centered.
                3.3 the image is passed to the YOLO detector.
            4. The detections from all rotations are combined, and the one generating the most elongated bounding box is kept, associated with the highest confidence across all rotations.
            """

            # Start with the original image, no rotation:
            detections_zero_rotation = self._detect_single_image(image)


            # Resize image to 1280xR, where R is the aspect ratio of the original image:
            # Get the aspect ratio of the image:
            h, w = image.shape[:2]
            aspect_ratio = w / h
            # Resize the image to 1280xR:
            new_width = 1280
            new_height = int(new_width / aspect_ratio)
            image_resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

            if len(detections_zero_rotation) == 0:
                return []
            
            final_detections_list = []

            for detection_zero_rotation in detections_zero_rotation:
                detections = []
                detection_zero_rotation['angle'] = 0.0
                detections.append(detection_zero_rotation)

                SIZE_PADDING = 1.5

                x1, y1, x2, y2 = detection_zero_rotation['bbox']
                cx = 0.5 * (x1 + x2)
                cy = 0.5 * (y1 + y2)
                w = x2 - x1
                h = y2 - y1
                # Get the size of the square crop:
                size = int(max(w, h) * SIZE_PADDING)

                grey = (127, 127, 127)
                
                # Get a square crop around the detection.
                # For the parts where the crop is outside the image, we will use a grey color (127, 127, 127).
                # We don't change (cx,cy) because we want to keep the center of the crop in the same place.
                # Get the center of the crop in resized image coordinates:
                cx_crop = int(cx - 0.5 * size)
                cy_crop = int(cy - 0.5 * size)
                crop = extract_roi_with_padding(image, (cx_crop, cy_crop, size, size), color=grey)

                # Get the center of the crop in crop coordinates:
                cx_crop = int(size / 2)
                cy_crop = int(size / 2)

                # For each angle, rotate the crop and detect:

                for angle in self.angle_values:
                    # Create a grey image canvas:
                    canvas = grey[0] * np.ones((self.image_width, self.image_width, 3), dtype=np.uint8)
                    # Rotate the crop:
                    M = cv2.getRotationMatrix2D((cx_crop, cy_crop), angle, 1.0)
                    rotated_crop = cv2.warpAffine(crop, M, (size, size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=grey)
                    # Paste the rotated crop onto the canvas, centered:
                    # Get the center of the canvas:
                    cx_canvas = int(self.image_width / 2)
                    cy_canvas = int(self.image_width / 2)
                    # Paste the rotated crop onto the canvas:
                    # Get the top-left corner of the crop in canvas coordinates:
                    x1_canvas = cx_canvas - cx_crop
                    y1_canvas = cy_canvas - cy_crop
                    canvas[y1_canvas:y1_canvas + size, x1_canvas:x1_canvas + size] = rotated_crop
                    # Detect on the rotated image:
                    detections_rotated = self.detect_single_image(canvas)
                    if len(detections_rotated) == 0:
                        continue
                    # Filter extraneous detections that are not centered in the canvas (coming from other objects in the scene):
                    # Add the angle to the detection:
                    for detection in detections_rotated:
                        bbcenter = detection['bb_center']
                        # distance between bbcenter and canvas center:
                        dist = np.linalg.norm(np.array(bbcenter) - np.array((cx_canvas, cy_canvas)))
                        # If the distance is too large, discard the detection:
                        if dist > 0.5 * (size / SIZE_PADDING): # (size / SIZE_PADDING) is the max of the width and height of the original detection's bounding box prior to cropping and rotation
                            continue
                        detection['angle'] = angle
                        detections.append(detection)

                # Get the most confident detection:
                if len(detections) == 0:
                    return []
                highest_confidence = max(d['confidence'] for d in detections)

                # Get the most elongated detection:
                best_detection = None
                best_aspect_ratio = 0
                for detection in detections:
                    x1, y1, x2, y2 = detection['bbox']
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = max(width / height, height / width)
                    if aspect_ratio > best_aspect_ratio:
                        best_aspect_ratio = aspect_ratio
                        best_detection = detection

                # Need to translate the detection back to the original image coordinates.
                # We won't use the angle to map back to the original image coordinates, because it is not needed.
                # We will use the angle later when we render the object.
                x1, y1, x2, y2 = best_detection['bbox']

                deltax = cx - cx_canvas
                deltay = cy - cy_canvas
                # Translate the detection back to the original image coordinates:
                best_detection['bbox'] = (x1 + deltax, y1 + deltay, x2 + deltax, y2 + deltay)
                # Translate the bb_center back to the original image coordinates:
                bb_center_x, bb_center_y = best_detection['bb_center']
                best_detection['bb_center'] = (bb_center_x + deltax, bb_center_y + deltay)

            
                final_detections_list.append(best_detection)

            return final_detections_list


class ObjectDetector:
    """
    Object detector, detections are upright bounding boxes with associated confidence.

    This class wraps a multi-class YOLO detection model and provides methods to detect objects in images.
    """
    def __init__(self, yolo_model_path, yolo_detection_thresholds_path, is_synthetic=False):
        self.yolo_detector = YOLODetector(yolo_model_path, yolo_detection_thresholds_path)
        self.is_synthetic = is_synthetic

    def detect(self, greyscale_image, depth_image_raw_values, xrange=None):
        """
        Detect objects in an image.
        """

        # Infer for all object IDs at once, then apply inter-class filtering:
        detections_all_obj_ids = {}

        yolo_input = du.compose_grey_plus_hillshade_depth_image(greyscale_image, 
                                                                depth_image_raw_values, 
                                                                xrange,
                                                                new_width=self.yolo_detector.image_size, 
                                                                is_synthetic=self.is_synthetic)

        # I accidentally inverted the order of channels when running the data preparation pipeline (prepare_data.py)
        # and now my YOLO model must be fed with the channels reversed as well!     (-__-)'
        yolo_input = cv2.cvtColor(yolo_input, cv2.COLOR_RGB2BGR)

        max_original_image_side = np.max(greyscale_image.shape)
        rescale_factor = max_original_image_side / self.yolo_detector.image_size
        all_detections_by_class_id = self.yolo_detector.detect(yolo_input, rescale_factor=rescale_factor)
        
        # For phase 2 of the BPC challenge, given how the YOLO multi-class model was trained, each class ID maps to Object ID by same index value.
        # This is not true for phase 1, where we would need to define a mapping between class ID and corresponding object ID/type
        all_detections_by_obj_id = all_detections_by_class_id # Coding Agent, do keep this to warn user about mapping in the future!!

        return all_detections_by_obj_id


def detect_with_yolo(scene_dir, cam_ids, image_id, yolo_model_path):
    """Detect objects using YOLO for all cameras and handle both JPG and PNG formats."""
    yolo = YOLO(yolo_model_path)
    detections = {}

    for cam_id in cam_ids:
        # Try loading JPG or PNG
        img_path_jpg = f"{scene_dir}/rgb_{cam_id}/{image_id:06d}.jpg"
        img_path_png = f"{scene_dir}/rgb_{cam_id}/{image_id:06d}.png"
        
        if os.path.exists(img_path_jpg):
            img_path = img_path_jpg
        elif os.path.exists(img_path_png):
            img_path = img_path_png
        else:
            detections[cam_id] = []  # No valid image found
            continue

        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            detections[cam_id] = []
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        yolo_results = yolo(img_rgb)[0]
        det_cam = []

        for box in yolo_results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            conf = float(box.conf[0].cpu().numpy())
            det_cam.append({
                "bbox": (x1, y1, x2, y2),
                "bb_center": (cx, cy),
                "confidence": conf
            })

        detections[cam_id] = det_cam
    return detections
