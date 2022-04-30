"""trt_yolo_mjpeg.py
This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.  Output stream is displayed to a Web browser.

MJPEG version of trt_yolo_gui.py.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from utils.yolo_classes import get_cls_dict
from utils.camera import add_camera_args, Camera
from utils.display import show_fps
from utils.visualization import BBoxVisualization
from utils.mjpeg import MjpegServer
from utils.yolo_with_plugins import TrtYOLO

from wpi_helpers import ConfigParser, PopulateNTData

def parse_args():
    """Parse input arguments."""
    desc = 'MJPEG version of trt_yolo'
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-c', '--category_num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('[yolov3-tiny|yolov3|yolov3-spp|yolov4-tiny|yolov4|'
              'yolov4-csp|yolov4x-mish|yolov4-p5]-[{dimension}], where '
              '{dimension} could be either a single number (e.g. '
              '288, 416, 608) or 2 numbers, WxH (e.g. 416x256)'))
    parser.add_argument(
        '-l', '--letter_box', action='store_true',
        help='inference with letterboxed image [False]')
    parser.add_argument(
        '-p', '--mjpeg_port', type=int, default=8080,
        help='MJPEG server port [8080]')
    args = parser.parse_args()
    return args


def loop_and_detect(cam, trt_yolo, conf_th, vis, nt, mjpeg_server):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      mjpeg_server
    """
    fps = 0.0
    tic = time.time()
    while True:
        img = cam.read()
        if img is None:
            break
        boxes, confidence, label = trt_yolo.detect(img, conf_th)
        img = vis.draw_bboxes(img, boxes, confidence, label)
        img = show_fps(img, fps)

        # Display stream to browser
        mjpeg_server.send_img(img)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

        # Put data to Network Tables
        nt.put_data(boxes, confidence, label, fps)


def main():
    args = parse_args()
    if args.category_num <= 0:
        raise SystemExit('ERROR: bad category_num (%d)!' % args.category_num)
    if not os.path.isfile('FRC-Jetson-Deployment-Models/%s.trt' % args.model):
        raise SystemExit('ERROR: file (FRC-Jetson-Deployment-Models/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # Get the team number for use in the Network Tables
    config_file = "FRC-Jetson-Deployment-Models/frc.json"
    config_parser = ConfigParser(config_file)    

    # Load the model
    cls_dict = get_cls_dict(args.category_num)
    vis = BBoxVisualization(cls_dict)
    trt_yolo = TrtYOLO(args.model, args.category_num, args.letter_box)

    # Connect to WPILib Network Tables
    print("Connecting to Network Tables")
    hardware_type = "USB Camera"
    nt = PopulateNTData(config_parser.team, hardware_type, cls_dict)

    mjpeg_server = MjpegServer(port=args.mjpeg_port)
    print('MJPEG server started...')
    try:
        loop_and_detect(cam, trt_yolo, conf_th=0.3, vis=vis, nt=nt,
                        mjpeg_server=mjpeg_server)
    except Exception as e:
        print(e)
    finally:
        mjpeg_server.shutdown()
        cam.release()


if __name__ == '__main__':
    main()
