# Weapon Detection Streamlit App

This is a web-based interface for testing a YOLO-based weapon detection model using Streamlit.

## Features

- **Demo Video Section**: Displays a demonstration video showing weapon detection in action
- **Image Upload & Testing**: Users can upload images to test the weapon detection model
- **Adjustable Confidence Threshold**: Users can adjust the detection confidence threshold
- **Download Results**: Users can download the processed images with detections
- **Real-time Processing**: Fast inference using the trained YOLO model

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Model File Exists

Make sure your trained YOLO model is located at:

```
./runs/detect/Normal_Compressed/weights/best.pt
```

### 3. Add Demo Video (Optional)

Place your processed demonstration video at:

```
./videos/test_video.mp4
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default web browser, typically at `http://localhost:8501`

## File Structure

```
weapon_detection_with_others/
├── app.py                          # Main Streamlit application
├── detecting-images.py             # Original detection functions
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── runs/
│   └── detect/
│       ├── Normal_Compressed/
│       │   └── weights/
│       │       └── best.pt         # Trained YOLO model
│       └── predict/                # Sample detection results
├── videos/
│   └── test_video.mp4              # Demo video (add your own)
├── imgs/
│   └── Test/                       # Output directory for processed images
└── results/                        # Output directory for processed videos
```

## Usage

1. **Demo Video**: The top section shows a demonstration video of weapon detection
2. **Image Upload**: Upload an image using the file uploader in the right column
3. **Adjust Threshold**: Use the slider to set the minimum confidence for detections
4. **Process Image**: Click "Detect Weapons" to run the model on your uploaded image
5. **Download Results**: Download the processed image with detection annotations

## Model Information

- **Architecture**: YOLOv8
- **Purpose**: Weapon detection in images and videos
- **Confidence Threshold**: Adjustable (default: 0.5)
- **Output**: Bounding boxes with confidence scores around detected weapons

## Troubleshooting

### Model Loading Issues

- Ensure the model file path is correct: `./runs/detect/Normal_Compressed/weights/best.pt`
- Check that the model file exists and is accessible

### Video Not Displaying

- Add your demo video to `./videos/test_video.mp4`
- Ensure the video format is supported (MP4, AVI)

### Detection Issues

- Try adjusting the confidence threshold
- Ensure uploaded images are clear and weapons are visible
- Check that the image format is supported (JPG, JPEG, PNG)

## Dependencies

- streamlit: Web application framework
- ultralytics: YOLO model implementation
- opencv-python: Image processing
- torch: Deep learning framework
- Pillow: Image handling
- numpy: Numerical operations

## Note

This application is designed for educational and demonstration purposes. The model's accuracy depends on the training data and may not detect all types of weapons or may produce false positives.
# SAD_RSM
