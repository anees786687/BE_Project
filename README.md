# BE Project — Stereo Vision Waste-Sorting Robot

A final-year robotics project for **AI-based recyclable waste detection, stereo depth estimation, live web visualization, ESP32-based robot control, and robotic pick-and-place operation**.

The system uses a stereo camera to estimate object distance and a YOLOv8-based detector to identify recyclable objects such as **plastic bottles** and **aluminum cans**. The perception output is displayed locally, streamed through a web interface, and used to support downstream robot-control logic through ROS 2 and ESP32 microcontrollers.

---

## Project Overview

This project aims to develop a perception-assisted robotic waste-sorting system capable of:

1. Capturing stereo camera input.
2. Splitting a combined stereo image into left and right frames.
3. Rectifying stereo images using calibration parameters.
4. Computing disparity using OpenCV StereoSGBM.
5. Improving disparity quality using WLS filtering.
6. Converting disparity into metric depth.
7. Detecting recyclable objects using a trained YOLOv8 model.
8. Estimating object distance from the depth map.
9. Publishing detection and depth information through ROS 2.
10. Streaming stereo and depth visualization through a browser interface.
11. Controlling the mobile base using an ESP32 and PS5 controller.
12. Sending arm pose commands from the base controller to the RDK X5.
13. Supporting dataset preparation, YOLO training, and fine-tuning through notebooks.

---

## Repository Structure

```text
BE_Project/
├── ESP32/
│   ├── Arm Controller/                  # ESP32 arm-control project
│   │   └── src/main.cpp                 # Arm controller code/template
│   └── Base Controller/                 # ESP32 mobile-base controller project
│       └── src/main.cpp                 # PS5 + BTS7960 base-control code
│
├── web_page/                            # Web dashboard/frontend files
├── depth_estim_1.py                     # ROS 2 + OpenCV visualization perception node
├── depth_estim_2.py                     # ROS 2 + FastAPI web-streaming perception node
├── yolov8m_recycle_detection.ipynb      # YOLOv8m training notebook
├── prepare_finetune_dataset(1).ipynb    # Dataset preparation and fine-tuning notebook
├── recycle_detector_v6_best.pt          # Fine-tuned YOLO model weights
├── tin_plastic_dataset.v3i.yolov8.zip   # YOLO-format recyclable-object dataset
├── .gitattributes                       # Git LFS tracking rules
├── .gitignore
└── README.md
```

---

## Main System Components

### 1. Stereo Depth Estimation

The perception pipeline processes a combined stereo image stream and separates it into left and right camera views. The images are rectified using calibration data and processed using OpenCV StereoSGBM.

Depth estimation includes:

* Stereo image rectification
* Left/right disparity computation
* Right-matcher generation
* WLS disparity filtering
* Disparity-to-depth conversion using the stereo reprojection matrix `Q`
* Invalid-depth filtering
* Temporal median filtering
* ROI-based object distance estimation

---

### 2. YOLOv8 Recyclable Object Detection

The project uses a trained YOLOv8 model for recyclable-object detection.

Final model used for inference:

```text
recycle_detector_v6_best.pt
```

Current detection classes:

```text
0 -> plastic_bottle
1 -> aluminum_can
```

The YOLO model runs on the rectified left camera frame and detects target recyclable objects before depth is sampled from the corresponding bounding-box region.

---

### 3. ROI-Based Object Distance Estimation

For each detected object:

1. YOLO predicts the bounding box.
2. The bounding box is mapped onto the stereo depth map.
3. The inner region of the object is sampled.
4. Invalid or missing depth values are removed.
5. A percentile-based depth estimate is calculated.
6. Previous valid depth values are cached to reduce flickering.
7. The closest valid object can be highlighted for pickup triggering.

The system includes different ROI strategies for different object types:

* **Plastic bottles** use an inner bounding-box crop.
* **Aluminum cans** use a wider center-strip strategy because reflective surfaces can produce sparse stereo depth.
* If the inner ROI has too few valid pixels, a border-ring fallback is used.

---

### 4. ROS 2 Integration

The perception node subscribes to the combined stereo image topic:

```text
/image_combine_jpeg
```

It publishes object detection results to:

```text
/detections
```

Example published message:

```text
plastic_bottle:0.235,aluminum_can:0.410
```

If depth is unavailable:

```text
plastic_bottle:none
```

---

## ESP32 Control System

The `ESP32/` folder contains embedded code for the robot hardware-control side of the project.

There are two ESP32 projects:

```text
ESP32/
├── Arm Controller/
└── Base Controller/
```

---

### 1. Base Controller

The `Base Controller` project controls the mobile robot base using:

* ESP32
* PS5 Bluetooth controller
* BTS7960 motor drivers
* Differential-drive motor control
* UART2 communication with the RDK X5

The base controller reads PS5 controller inputs and converts them into motor commands.

#### Motor Control

The base controller uses two BTS7960 motor-driver objects:

```text
leftSide
rightSide
```

The control logic supports:

* Forward motion using R2
* Reverse motion using L2
* Left/right steering using the left analog stick
* On-spot turning when no trigger is pressed and the stick is moved left/right
* Emergency stop when the controller disconnects

#### Base Controller Pin Mapping

```text
LEFT_L_EN   -> GPIO 25
LEFT_R_EN   -> GPIO 26
LEFT_L_PWM  -> GPIO 32
LEFT_R_PWM  -> GPIO 33

RIGHT_L_EN  -> GPIO 4
RIGHT_R_EN  -> GPIO 18
RIGHT_L_PWM -> GPIO 19
RIGHT_R_PWM -> GPIO 5
```

#### UART Connection to RDK X5

The ESP32 communicates with the RDK X5 through UART2.

```text
RDK_RX -> GPIO 16
RDK_TX -> GPIO 17
Baud   -> 115200
```

This UART link is used to send arm pose commands from the PS5 controller to the RDK X5 arm-command node.

#### PS5 Button Mapping

The base controller sends pose commands over UART2 using the PS5 face buttons and D-pad.

```text
Cross     -> pose_1
Square    -> pose_2
Triangle  -> pose_3
Circle    -> pose_4
D-pad Down  -> pose_5
D-pad Left  -> pose_6
D-pad Up    -> pose_7
D-pad Right -> pose_8
```

These commands can be received by the RDK X5 and mapped to predefined robotic-arm poses.

#### Main Tuning Parameters

```cpp
const int DEAD      = 10;
const int MIN_SPEED = 80;
const int MAX_SPEED = 200;
const int TURN_DIFF = 53;
const int LOOP_MS   = 20;
```

Meaning:

* `DEAD`: trigger deadband.
* `MIN_SPEED`: minimum PWM needed to overcome motor stiction.
* `MAX_SPEED`: maximum PWM cap.
* `TURN_DIFF`: speed difference between fast and slow side while turning.
* `LOOP_MS`: main control-loop delay.

---

### 2. Arm Controller

The `Arm Controller` folder is reserved for the ESP32-side arm-control firmware.

This controller can be used for:

* Servo control
* Robotic arm pose execution
* Receiving pose commands from the RDK X5 or base controller
* Mapping commands such as `pose_1`, `pose_2`, ..., `pose_8` to predefined joint positions

At the current stage, the uploaded arm-controller `main.cpp` appears to be a basic Arduino/PlatformIO starter template. The full arm-control logic can be added later.

---

## OpenCV Visualization Node

`depth_estim_1.py` provides local OpenCV visualization.

Run:

```bash
python3 depth_estim_1.py
```

It displays:

* Rectified left camera view
* Rectified right camera view
* YOLO bounding boxes
* Object labels
* Confidence scores
* Estimated depth values
* Colorized depth map
* Closest-object highlighting
* Trigger-range highlighting

Press `q` to exit the OpenCV windows.

---

## FastAPI Web Streaming Node

`depth_estim_2.py` provides browser-based visualization using FastAPI and MJPEG streaming.

Run:

```bash
python3 depth_estim_2.py
```

Then open:

```text
http://localhost:8000
```

Available endpoints:

```text
GET /api/stream/stereo
GET /api/stream/depth
GET /api/status
GET /api/snapshot/stereo
GET /api/snapshot/depth
GET /api/health
```

The web interface can be used to view:

* Rectified stereo stream
* YOLO detections
* Depth map visualization
* System status
* Snapshot images

---

## Training and Dataset Preparation

This repository includes notebooks used for dataset preparation, YOLO training, and fine-tuning.

---

### 1. YOLOv8m Training Notebook

```text
yolov8m_recycle_detection.ipynb
```

Purpose:

* Train a YOLOv8m recyclable-object detector.
* Use a Roboflow YOLO-format dataset.
* Filter the dataset to two target classes.
* Remap class IDs for binary recyclable-object detection.
* Train using pretrained YOLOv8m weights.
* Evaluate the trained model.
* Export the best model weights.

Target classes:

```text
Plastic bottle
Aluminum can
```

Typical training setup:

```python
model = YOLO("yolov8m.pt")
```

Example training parameters:

```python
epochs=50
imgsz=640
batch=16
optimizer="AdamW"
lr0=0.001
mosaic=1.0
mixup=0.1
degrees=10.0
fliplr=0.5
device=0
```

---

### 2. Fine-Tuning Dataset Preparation Notebook

```text
prepare_finetune_dataset(1).ipynb
```

Purpose:

* Prepare a custom fine-tuning dataset.
* Fix class-name mismatches.
* Convert class labels into the final two-class format.
* Resize and pad images to YOLO-compatible dimensions.
* Create train/validation splits.
* Generate a corrected `data.yaml`.
* Fine-tune from a previous trained YOLO model.
* Export the final fine-tuned model.

Common corrections:

```text
plastic_bottle -> Plastic bottle
tin_can        -> Aluminum can
```

Final exported model:

```text
recycle_detector_v6_best.pt
```

---

## Dataset

The repository includes a YOLO-format dataset archive:

```text
tin_plastic_dataset.v3i.yolov8.zip
```

This dataset is used for training or fine-tuning the recyclable-object detector.

Final object classes:

```text
Plastic bottle
Aluminum can
```

---

## Hardware Used

The project is designed for a robotic waste-sorting platform using:

* Stereo camera
* RDK X5 or Linux-based robotics computer
* ESP32 microcontroller for base control
* ESP32 microcontroller for arm control
* BTS7960 motor drivers
* PS5 Bluetooth controller
* Robotic arm or mobile manipulator
* Servo/motor driver hardware
* YOLOv8 recyclable-object detection model

---

## Software Requirements

Recommended environment:

* Ubuntu Linux
* Python 3.10+
* ROS 2 Humble
* OpenCV with contrib modules
* Ultralytics YOLO
* NumPy
* PyYAML
* FastAPI
* Uvicorn
* Git LFS
* PlatformIO or Arduino framework for ESP32 firmware

Install Python dependencies:

```bash
pip install ultralytics numpy pyyaml fastapi uvicorn
```

Install OpenCV with contrib modules:

```bash
pip install opencv-contrib-python
```

For ROS 2 Humble:

```bash
source /opt/ros/humble/setup.bash
```

For training notebooks, use Google Colab with GPU enabled:

```text
Runtime -> Change runtime type -> GPU
```

---

## ESP32 Build Notes

The ESP32 firmware can be built using PlatformIO or the Arduino framework.

Required ESP32-side libraries may include:

```text
Arduino.h
BTS7960
ps5Controller
```

Before uploading the Base Controller firmware, update the PS5 controller Bluetooth MAC address if required:

```cpp
static const char* PS5_MAC = "7c:66:ef:44:90:c9";
```

The base controller starts by connecting to the PS5 controller. If the controller disconnects, the robot automatically stops the motors.

---

## Git LFS Setup

This repository uses **Git LFS** for large model and dataset files.

Install Git LFS:

```bash
sudo apt update
sudo apt install git-lfs
git lfs install
```

Clone the repository:

```bash
git clone https://github.com/anees786687/BE_Project.git
cd BE_Project
git lfs pull
```

Check LFS files:

```bash
git lfs ls-files
```

Expected large files include:

```text
recycle_detector_v6_best.pt
tin_plastic_dataset.v3i.yolov8.zip
```

Do not delete `.gitattributes`, because it tells Git which files should be handled by Git LFS.

---

## Stereo Calibration

The perception scripts require a stereo calibration YAML file.

Default path used in the scripts:

```text
/home/anees/rdk_yolo/stereo_calib_640x352.yaml
```

The calibration file should contain:

* Left camera intrinsic matrix
* Right camera intrinsic matrix
* Left and right distortion coefficients
* Rectification matrices
* Projection matrices
* Reprojection matrix `Q`
* Image width
* Image height

If your calibration file is in a different location, update the `CALIB_FILE` variable inside the Python scripts.

---

## Important Perception Parameters

Some important parameters used in the stereo-depth pipeline:

```python
DEPTH_OFFSET = 0.025
DEPTH_THRESH = 0.25
BOX_PERSIST_S = 0.5
DEPTH_CACHE_MAX = 50
```

Meaning:

* `DEPTH_OFFSET`: depth correction offset in metres.
* `DEPTH_THRESH`: distance threshold for pickup or trigger logic.
* `BOX_PERSIST_S`: keeps bounding boxes briefly to reduce detection flicker.
* `DEPTH_CACHE_MAX`: prevents unlimited growth of cached depth entries.

---

## Typical Workflow

### 1. Clone the repository

```bash
git clone https://github.com/anees786687/BE_Project.git
cd BE_Project
git lfs pull
```

### 2. Install Python dependencies

```bash
pip install ultralytics numpy pyyaml fastapi uvicorn opencv-contrib-python
```

### 3. Source ROS 2

```bash
source /opt/ros/humble/setup.bash
```

### 4. Run the OpenCV perception node

```bash
python3 depth_estim_1.py
```

### 5. Or run the browser-streaming node

```bash
python3 depth_estim_2.py
```

Then open:

```text
http://localhost:8000
```

### 6. Upload ESP32 firmware

Open the relevant ESP32 project:

```text
ESP32/Base Controller/
ESP32/Arm Controller/
```

Build and upload using PlatformIO or the Arduino framework.

---

## Notes

* Large files are stored using Git LFS.
* Do not commit virtual environments such as `rdk_venv/` or `venv/`.
* Use `opencv-contrib-python` if `cv2.ximgproc` is required.
* Stereo calibration quality strongly affects depth accuracy.
* Transparent plastic bottles and reflective aluminum cans may produce sparse depth values.
* The pipeline includes fallback depth sampling for difficult objects.
* The notebooks are intended to be run in Google Colab with GPU acceleration.
* `depth_estim_1.py` is intended for local OpenCV display.
* `depth_estim_2.py` is intended for browser-based streaming.
* The ESP32 Base Controller supports manual robot driving and pose-command transmission.
* The ESP32 Arm Controller folder is reserved for arm-control firmware.

---

## Future Improvements

* Add a ROS 2 launch file.
* Add a sample stereo calibration YAML file.
* Add setup script or `requirements.txt`.
* Add screenshots of stereo and depth outputs.
* Add web-interface screenshots.
* Add ESP32 wiring diagram.
* Add robot-arm control integration.
* Add finalized Arm Controller firmware.
* Add object pickup state machine.
* Add trained-model evaluation results.
* Add confusion matrix and validation plots.
* Add example inference images.
* Add Docker support for easier deployment.

---

## Author

**Anees Alwani, Dharmil Trivedi, Raghav Agarwal**

Final-year Electronics and Telecommunication Engineering project.

GitHub: [anees786687](https://github.com/anees786687) [Dharmil20](https://github.com/Dharmil20) [GMRicks](https://github.com/GMRicks)
