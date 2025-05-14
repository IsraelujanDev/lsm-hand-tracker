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
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         lsm_hand_tracker and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── lsm_hand_tracker   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes lsm_hand_tracker a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

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