# Mexican Sign Language (LSM) Recognition Pipeline

## Project Overview
This project aims to develop a pipeline for detecting and classifying hand gestures in Mexican Sign Language (Lenguaje de Señas Mexicano, LSM). The ultimate goal is to create an application that facilitates communication between individuals who are deaf or hard of hearing and those who do not know LSM.

The project is currently in the **experimental phase**, focusing on setting up the environment, testing models, and exploring tools like MediaPipe for hand tracking and gesture recognition.

---

## Features
- **Hand Detection and Tracking**: Using MediaPipe's pre-trained models to detect and track hand landmarks in real-time.
- **Gesture Classification**: Experimental phase to classify hand gestures into LSM signs.
- **Image Processing Pipeline**: Loading, processing, and visualizing images for gesture recognition.
- **Future Integration**: Plans to integrate the pipeline into a user-friendly application.

---

## Project Structure
```
Mexican Sign Language/
├── app/                            # Placeholder for future application logic (e.g., frontend/backend)
├── data/                           # Data directory for all resources used in experiments
│   ├── annotations/                # Manual or auto-generated gesture annotations
│   ├── images/                     # Processed images used for gesture recognition
│   ├── metadata/                   # Supplementary metadata (e.g., labels, sources)
│   ├── raw/                        # Unprocessed/raw image and video files
├── models/                         # Pretrained models (.task) and trained weights
├── notebooks/                      # Jupyter notebooks for experimentation and visualization
│   ├── mediapipe_new_api.ipynb     # Notebook using the new MediaPipe Tasks API
│   ├── mediapipe_new_api_simple.ipynb  # Minimal example for quick testing
├── scripts/                        # Utility and demo scripts
│   ├── test_video_hand_detector_old_api.py  # Real-time detection using legacy MediaPipe API
├── .python-version                 # Defines the Python version for reproducibility
├── pyproject.toml                  # Project configuration and dependency definitions (UV)
├── uv.lock                         # Locked dependency versions for deterministic builds
├── README.md                       # Project overview and usage instructions

```

---

## Getting Started

### Prerequisites
- Python 3.10+
- Install required libraries:
 we recommend using a UV as a package manager to ensure reproducibility and manage dependencies effectively.

  ```bash
  pip install mediapipe opencv-python matplotlib python-dotenv
  ```

### Environment Setup
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Mexican Sign Language
   ```
2. Set up environment variables:
   - Create a `.env` file in the root directory.
   - Add the following:
     ```
     LSM_BASE=<path_to_project_directory>
     ```

---

## Usage

### Running the Experimental Notebooks
1. Open `notebooks/mediapipe_new_api.ipynb` or `mediapipe_new_api_simple.ipynb` in Jupyter Notebook.
2. Follow the cells to test hand detection and gesture recognition.

### Running the Old API Script
1. Run the script to test real-time hand tracking:
   ```bash
   python scripts/test_video_hand_detector_old_api.py
   ```

---

## Key Components

### MediaPipe Integration
- **Hand Landmarker**: Detects 21 hand-knuckle coordinates.
- **Connections and Colors**: Visualizes hand landmarks and connections with customizable colors.

### Experimental Features
- Gesture classification using hand landmarks.
- Image-based and video-based hand tracking.

---

## Future Work
- Develop a robust gesture classification model for LSM.
- Build a user-friendly application for real-time communication.
- Expand the dataset for better accuracy and coverage of LSM gestures.

---

## References
- [MediaPipe Hand Landmarker Documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker/index#models)
- [OpenCV Documentation](https://docs.opencv.org/)

---

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or suggestions.

---

## License
This project is licensed under the MIT License.