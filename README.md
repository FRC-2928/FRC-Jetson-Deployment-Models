# FRC-Jetson-Deployment-Models
This repository stores *Deep Learning* models and scripts that can be deployed on the Jetson Nano. There are two deployment options.  One uses an attached [Luxonis OAK](https://shop.luxonis.com/products/1098obcenclosure) camera.  The OAK camera has an imbedded TPU the runs on the [Intel OpenVINO Toolkit](https://www.intel.com/content/www/us/en/developer/tools/openvino-toolkit/overview.html). The OAK camera's TPU runs the inference model, taking the processing off of the Raspberry Pi.  

The second option uses a Raspberry Pi/USB camera.  In this configuration the model inference is done on the Jetson.  Since the Jetson has its own GPU the detection frames per second FPS is sufficient for competition.

Before deploying these models the Jetson Nano must be installed with Jetpack 4.6.  The main files are included in this repository are as follows:

- `oak_yolo_spacial.py`  This is the default script that runs inference on a Yolo model and outputs detected objects with a label, bounding boxes and their X, Y, Z coordinates from the camera.  The script will display its output in a Web browser at `10.0.0.2:8091` and also places all of the data into the *WPILib* Network Tables.

- `oak_yolo.py`  This is a lighter version of the above script that only collects the label and bounding box data.

- `oak_yolo_spacial_gui.py`  This can be used to display camera stream output to the Jetson Nano desktop gui or any other device that has a gui desktop.

- `tft_yolo.py` This script is used to run inference on the Jetson with a Raspberry Pi or USB camera.  It uses the TensorRT deployment format.  The script displays its output in a Web browser at `10.0.0.2:8091` and also places data into the *WPILib* Network Tables.

- `tft_yolo_gui.py` Same as trt_yolo.py exect that is displays output in a window to the desktop gui.

- `rapid-react.blob` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition. The blob file format is designed to run specifically on an *OpenVINO* device.

- `rapid-react-config.json` This is the configuration file needed to load the rapid-react model.  It includes the labels for the objects of interest. 

- `rapid-react.trt` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition. The TensorRT `trt` file format runs natively on the Jetson.
