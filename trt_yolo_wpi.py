"""trt_yolo_gui.py

This script demonstrates how to do real-time object detection with
TensorRT optimized YOLO engine.
"""


import os
import time
import argparse

import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver

from pathlib import Path
from utils.camera import add_camera_args, Camera
from utils.display import open_window, set_display, show_fps
from utils.visualization import BBoxVisualization
from utils.yolo_with_plugins import TrtYOLO
import cscore as cs

from wpi_helpers import ConfigParser, WPINetworkTables, ModelConfigParser


WINDOW_NAME = 'TrtYOLODemo'


def parse_args():
    """Parse input arguments."""
    desc = ('Capture and display live camera video, while doing '
            'real-time object detection with TensorRT optimized '
            'YOLO model on Jetson')
    parser = argparse.ArgumentParser(description=desc)
    parser = add_camera_args(parser)
    parser.add_argument(
        '-g', '--gui', action='store_true',
        help='use desktop gui for display [False]')
    parser.set_defaults(gui=False)    
    parser.add_argument(
        '-t', '--conf_thresh', type=float, default=0.3,
        help='set the detection confidence threshold')
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


def loop_and_detect(cam, trt_yolo, conf_th, vis, nt, cvSource):
    """Continuously capture images from camera and do object detection.

    # Arguments
      cam: the camera instance (video source).
      trt_yolo: the TRT YOLO object detector instance.
      conf_th: confidence/score threshold for object detection.
      vis: for visualization.
      nt: the WPI Network Tables.
      cvSource: The source going out to the mjpeg server
    """
    full_scrn = False
    fps = 0.0
    tic = time.time()
    while True:
        if cvSource is False:
            # In gui mode
            if cv2.getWindowProperty(WINDOW_NAME, 0) < 0:
                break

        img = cam.read()
        if img is None:
            break

        # Get inference output
        boxes, confidence, label = trt_yolo.detect(img, conf_th)

        # Draw bounding boxes and label frame
        img = vis.draw_bboxes(img, boxes, confidence, label)
        img = show_fps(img, fps)

        if cvSource == False:
            # Display stream to desktop window
            cv2.imshow(WINDOW_NAME, img)
        else:               
            # Display stream to browser
            cvSource.putFrame(img)

        toc = time.time()
        curr_fps = 1.0 / (toc - tic)
        # calculate an exponentially decaying average of fps number
        fps = curr_fps if fps == 0.0 else (fps*0.95 + curr_fps*0.05)
        tic = toc

        # Put data to Network Tables
        nt.put_data(boxes, confidence, label, fps)

        if cvSource is False:
            # key = cv2.waitKey(1)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # elif key == ord('F') or key == ord('f'):  # Toggle fullscreen
            #     full_scrn = not full_scrn
            #     set_display(WINDOW_NAME, full_scrn) 


def main():
    args = parse_args()
    if not os.path.isfile('FRC-Jetson-Deployment-Models/%s.trt' % args.model):
        raise SystemExit('ERROR: file (FRC-Jetson-Deployment-Models/%s.trt) not found!' % args.model)

    cam = Camera(args)
    if not cam.isOpened():
        raise SystemExit('ERROR: failed to open camera!')

    # Get the team number for use in the Network Tables
    config_file = "FRC-Jetson-Deployment-Models/frc.json"
    config_parser = ConfigParser(config_file)    
   
    ## Read the model configuration file
    print("Loading network settings")
    default_config_file = 'FRC-Jetson-Deployment-Models/rapid-react-config.json'
    configPath = str((Path(__file__).parent / Path(default_config_file)).resolve().absolute())    
    model_config = ModelConfigParser(configPath)
    print(model_config.labelMap)
    print("Classes:", model_config.classes)
    print("Confidence Threshold:", model_config.confidence_threshold)

    # Load the model
    vis = BBoxVisualization(model_config.labelMap)
    trt_yolo = TrtYOLO(args.model, model_config.classes, args.letter_box)

    # Connect to WPILib Network Tables
    print("Connecting to Network Tables")
    hardware_type = "USB Camera"
    nt = WPINetworkTables(config_parser.team, hardware_type, model_config.labelMap)

    if args.gui is True:
        print("Gui requested")
        open_window(
        WINDOW_NAME, 'Camera TensorRT YOLO Demo',
        cam.img_width, cam.img_height)
        try:
            loop_and_detect(cam, trt_yolo, model_config.confidence_threshold, 
                            vis, nt, cvSource=False)
        except Exception as e:
            print(e)
        finally:
            print("Releasing camera...")                
            cam.release()
            cv2.destroyAllWindows()
    else:
        # Use the mjpeg server (default)
        cvSource = cs.CvSource("cvsource", cs.VideoMode.PixelFormat.kMJPEG, 320, 240, 30)
        mjpeg_server = cs.MjpegServer("httpserver", args.mjpeg_port)
        mjpeg_server.setSource(cvSource)
        print('MJPEG server started on port', args.mjpeg_port)
        try:
            loop_and_detect(cam, trt_yolo, model_config.confidence_threshold, 
                            vis, nt, cvSource=cvSource)
        except Exception as e:
            print(e)
        finally:
            print("Releasing camera...")
            cam.release() 


if __name__ == '__main__':
    main()
