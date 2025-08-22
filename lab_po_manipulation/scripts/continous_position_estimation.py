import cv2
import numpy as np
from lab_po_manipulation.poman_motion_planner import POManMotionPlanner
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.manipulation.manipulation_controller_2fg import ManipulationController2FG
from lab_ur_stack.robot_inteface.robot_interface import RobotInterface
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1
from lab_ur_stack.motion_planning.geometry_and_transforms import GeometryAndTransforms
from lab_ur_stack.camera.configurations_and_params import color_camera_intrinsic_matrix
from lab_po_manipulation.image_state_estimation import ObjectPositionEstimator
from lab_po_manipulation.env_configurations.objects_and_positions import positions

classes = ('green soda can', 'red cup with black strap', 'blue cup')

# Initialize all components
camera = RealsenseCamera()
mp = POManMotionPlanner()
mp.add_mobile_obstacle_by_number(2)
gt = GeometryAndTransforms(mp)
estimator = ObjectPositionEstimator(classes, positions, gt)
robot = ManipulationController2FG(ur5e_1["ip"], ur5e_1["name"], mp, gt)
robot.freedriveMode()


def create_info_display(width, height, results_dict):
    """Create an image displaying detection results with horizontal layout"""
    info_display = np.ones((height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(info_display, "Detected Objects:", (10, 30), font, 0.8, (0, 0, 0), 2)

    # Calculate spacing for horizontal layout
    x_offset = 10
    object_width = width // len(classes)  # Divide space evenly between objects

    for class_name in classes:
        y_offset = 70  # Start below title

        if class_name in results_dict['raw_detections']:
            raw_pos = results_dict['raw_detections'][class_name]['position']
            mapped_pos = results_dict['mapped_positions'].get(class_name)

            # Class name
            cv2.putText(info_display, f"{class_name}:", (x_offset, y_offset), font, 1., (0, 0, 0), 2)
            y_offset += 30

            if mapped_pos:
                # Mapped position in large text
                cv2.putText(info_display, mapped_pos['position'].name,
                            (x_offset, y_offset), font, 1.0, (0, 150, 0), 2)
                y_offset += 25
                # Distance and raw position in smaller text
                text = f"d={mapped_pos['distance']:.3f}m"
                cv2.putText(info_display, text, (x_offset, y_offset), font, 0.5, (100, 100, 100), 1)
                y_offset += 20
                text = f"raw: ({raw_pos[0]:.2f}, {raw_pos[1]:.2f}, {raw_pos[2]:.2f})"
                cv2.putText(info_display, text, (x_offset, y_offset), font, 0.4, (100, 100, 100), 1)
            else:
                text = "No valid position match"
                cv2.putText(info_display, text, (x_offset, y_offset), font, 0.8, (200, 0, 0), 1)
        else:
            text = f"{class_name}: Not Seen"
            cv2.putText(info_display, text, (x_offset, y_offset), font, 0.8, (100, 100, 100), 1)

        x_offset += object_width

    return info_display


while True:
    rgb, depth = camera.get_frame_rgb()
    if rgb is None:
        continue

    # Get current robot configuration
    robot_config = robot.getActualQ()

    # Get estimations and annotations
    results_dict, (annotated_img, depth_vis) = estimator.estimate_positions(rgb, depth, robot_config)

    # Convert RGB to BGR for display
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR)

    # Put RGB and depth side by side
    top_row = np.hstack([annotated_img, depth_vis])

    # Create info display with width of combined images and increased height
    info_display = create_info_display(top_row.shape[1], 250, results_dict)

    # Stack info display below
    combined_display = np.vstack([top_row, info_display])

    # Scale if too large
    max_width = 1800
    if combined_display.shape[1] > max_width:
        scale = max_width / combined_display.shape[1]
        new_height = int(combined_display.shape[0] * scale)
        combined_display = cv2.resize(combined_display, (max_width, new_height))

    # Show result
    cv2.imshow('Object Detection and Position Estimation', combined_display)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()