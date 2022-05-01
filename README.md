# FRC-Jetson-Deployment-Models
This repository stores *Deep Learning* models and scripts that can be deployed on the Jetson Nano. This option uses a Raspberry Pi/USB camera.  In this configuration the model inference is done on the Jetson.  Since the Jetson has its own GPU the detection frames per second FPS is sufficient for competition.

Before deploying these models the Jetson Nano must be installed with Jetpack 4.6.  The main files are included in this repository are as follows:

- `trt_yolo_wpi.py` This script is used to run inference on the Jetson with a Pi or USB camera.  It runs a Yolo model using the TensorRT deployment format (`.trt` file).  The script displays its output in a Web browser at `<server IP address>:8080` by default and also places data into the *WPILib* Network Tables. If you're running this within a desktop environment you can also use the `--gui` option to display the output in a gui window.

- `rapid-react.trt` This model has been trained on the Rapid-React balls from the 2022 FIRST Competition. The TensorRT `trt` file format runs natively on the Jetson.

- `rapid-react-config.json` This is the configuration file needed to load the rapid-react model.  It includes the class labels and confidence level. 

### Running the inference script

To run the inference using an attached Raspberry Pi camera.  

    cd ${HOME}tensorrt_demos
    python3 trt_yolo_mjpeg.py --onboard 0 -m rapid-react

For a USB camera:    

    python3 trt_yolo_wpi.py --usb 1 -m rapid-react

You can display the output stream in a desktop gui window like this:  

    python3 trt_yolo_wpi.py --usb 1 -m rapid-react --gui