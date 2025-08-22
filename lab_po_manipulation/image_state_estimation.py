from dataclasses import dataclass
import numpy as np
import cv2
from typing import Dict, Any, Optional, Sequence

from lab_po_manipulation.env_configurations.objects_and_positions import ItemPosition
from lab_ur_stack.camera.configurations_and_params import color_camera_intrinsic_matrix
from lab_ur_stack.camera.realsense_camera import project_color_pixel_to_depth_pixel
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.vision.object_detection import ObjectDetection


class ObjectPositionEstimator:
    def __init__(self, classes: Sequence[str], positions: Sequence[ItemPosition],
                 gt: GeometryAndTransforms, robot_name='ur5e_1',
                 intrinsic_camera_matrix=color_camera_intrinsic_matrix):
        self.positions = positions
        self.gt = gt
        self.detector = ObjectDetection(classes=classes)
        self.robot_name = robot_name
        self.intrinsic_camera_matrix = intrinsic_camera_matrix

    def get_best_detections_per_class(self, bboxes, confidences, results):
        """Get highest confidence detection for each class"""
        best_detections = {}
        for bbox, conf, result in zip(bboxes[0], confidences[0], results[0].boxes.cls):
            class_name = results[0].names[int(result)]
            if class_name not in best_detections or conf.cpu().numpy() > best_detections[class_name]['confidence']:
                best_detections[class_name] = {
                    'bbox': bbox.cpu().numpy(),
                    'confidence': conf.cpu().numpy()
                }
        return best_detections

    def get_z_depth_mean(self, bbox, depth_image):
        """
        Calculate mean depth in window around bbox center ignore pixels where
         distance is 0 because those are without depth
         """
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        # Project point from color to depth image
        center_in_depth = project_color_pixel_to_depth_pixel([center_x, center_y], depth_image)

        if center_in_depth[0] < 0 or center_in_depth[1] < 0 or \
                center_in_depth[0] >= depth_image.shape[1] or center_in_depth[1] >= depth_image.shape[0]:
            return -1, None

        # Window size: quarter of bbox size, clipped between 3 and 15
        win_size_x = np.clip(int((bbox[2] - bbox[0]) / 4), 3, 15)
        win_size_y = np.clip(int((bbox[3] - bbox[1]) / 4), 3, 15)

        x_min = max(0, int(center_in_depth[0] - win_size_x))
        x_max = min(depth_image.shape[1], int(center_in_depth[0] + win_size_x))
        y_min = max(0, int(center_in_depth[1] - win_size_y))
        y_max = min(depth_image.shape[0], int(center_in_depth[1] + win_size_y))

        window = depth_image[y_min:y_max, x_min:x_max]
        if np.all(window <= 0):
            return -1, None

        return np.mean(window[window > 0]), (x_min, x_max, y_min, y_max)

    def points_image_to_camera_frame(self, points_image_xy, z_depth):
        """Transform points from image coordinates to camera frame"""
        fx = self.intrinsic_camera_matrix[0, 0]
        fy = self.intrinsic_camera_matrix[1, 1]
        cx = self.intrinsic_camera_matrix[0, 2]
        cy = self.intrinsic_camera_matrix[1, 2]

        x_camera = (points_image_xy[0] - cx) * z_depth / fx
        y_camera = (points_image_xy[1] - cy) * z_depth / fy
        return np.array([x_camera, y_camera, z_depth])

    def find_nearest_valid_position(self, detected_position, max_distance=0.4):
        """Find nearest predefined position within max_distance"""
        nearest_position = None
        min_distance = float('inf')

        for position in self.positions:
            distance = np.linalg.norm(np.array(detected_position) - np.array(position.position))
            if distance < min_distance and distance < max_distance:
                min_distance = distance
                nearest_position = position

        return nearest_position, min_distance if nearest_position else None

    def estimate_positions(self, image, depth, robot_configuration, return_annotations=True):
        """Main method to detect objects and estimate their positions"""
        bboxes, confidences, results = self.detector.detect_objects([image])
        best_detections = self.get_best_detections_per_class(bboxes, confidences, results)

        raw_detections = {}
        mapped_positions = {}
        depth_windows = []

        for class_name, detection in best_detections.items():
            bbox = detection['bbox']
            center_xy = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

            # Get depth and window coordinates
            z_depth, window_xyxy = self.get_z_depth_mean(bbox, depth)
            if z_depth <= 0:
                continue

            if window_xyxy:
                depth_windows.append(window_xyxy)

            position_camera = self.points_image_to_camera_frame(center_xy, z_depth)
            position_world = self.gt.point_camera_to_world(position_camera, self.robot_name, robot_configuration)

            # Store raw detection
            raw_detections[class_name] = {
                'position': position_world,
                'confidence': float(detection['confidence']),
                'bbox': bbox
            }

            # Find nearest valid position
            nearest_position, distance = self.find_nearest_valid_position(position_world)
            if nearest_position:
                mapped_positions[class_name] = {
                    'position': nearest_position,
                    'distance': float(distance)
                }

        results_dict = {
            'raw_detections': raw_detections,
            'mapped_positions': mapped_positions
        }

        if return_annotations:
            # Get annotated image from YOLO results
            annotated_image = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

            # Create annotated depth image
            max_depth_for_plot = 3
            depth_vis = depth.copy()
            depth_vis = np.clip(depth_vis, 0, max_depth_for_plot)
            depth_vis = ((depth_vis / max_depth_for_plot) * 255).astype(np.uint8)
            depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_GRAY2RGB)
            # Draw windows used for depth calculation
            for win_xyxy in depth_windows:
                depth_vis = cv2.rectangle(
                    depth_vis,
                    (win_xyxy[0], win_xyxy[2]),
                    (win_xyxy[1], win_xyxy[3]),
                    (0, 255, 0),
                    1
                )

            return results_dict, (annotated_image, depth_vis)

        return results_dict

