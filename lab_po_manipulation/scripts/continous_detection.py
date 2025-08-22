import cv2
from lab_ur_stack.camera.realsense_camera import RealsenseCamera
from lab_ur_stack.robot_inteface.robot_interface import RobotInterface
from lab_ur_stack.robot_inteface.robots_metadata import ur5e_1
from lab_ur_stack.vision.object_detection import ObjectDetection
from lab_po_manipulation.image_state_estimation import ObjectPositionEstimator
from lab_po_manipulation.env_configurations.objects_and_positions import positions


classes = ('red soda can', 'coca cola can', 'red cup with white strap', 'blue cup', )
confidence = 0.1

classes = ('stupid human',)
confidence = 0.003


camera = RealsenseCamera()
od = ObjectDetection(classes=classes, min_confidence=confidence)
robot = RobotInterface(ur5e_1['ip'])
robot.freedriveMode()


while True:
    rgb, depth = camera.get_frame_rgb()
    if rgb is None:
        continue

    _, _, res = od.detect_objects([rgb])
    ann = od.get_annotated_images(res[0])
    ann = cv2.cvtColor(ann, cv2.COLOR_RGB2BGR)
    cv2.imshow('detections', ann)
    cv2.imshow('depth', depth)
    cv2.waitKey(1)


