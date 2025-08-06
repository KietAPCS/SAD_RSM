import streamlit as st
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import ultralytics.nn.tasks as tasks
from PIL import Image
import tempfile
import os

# Set page configuration
st.set_page_config(
    page_title="Weapon Detection System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

def unsafe_load(file):
    """Custom unsafe load function for YOLO model"""
    return torch.load(file, map_location='cpu', weights_only=False), file

@st.cache_resource
def load_model():
    """Load the YOLO model once and cache it"""
    tasks.torch_safe_load = unsafe_load
    # Try different possible model paths
    model_paths = [
        './runs/detect/train/weights/best.pt'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            model = YOLO(path)
            return model
    
    # If no model found, raise an error
    raise FileNotFoundError(f"Model not found in any of these paths: {model_paths}")
    return model

def detect_objects_in_image(image, model, confidence_threshold=0.5):
    """
    Detect objects in an image using the loaded YOLO model
    
    Args:
        image: PIL Image or numpy array
        model: Loaded YOLO model
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        Processed image with detections
    """
    # Convert PIL to numpy array if needed
    if isinstance(image, Image.Image):
        image_array = np.array(image)
        # Convert RGB to BGR for OpenCV
        image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
    else:
        image_array = image.copy()
    
    # Run inference
    results = model(image_array)
    
    # Process results
    for result in results:
        if result.boxes is not None:
            classes = result.names
            cls = result.boxes.cls
            conf = result.boxes.conf
            detections = result.boxes.xyxy
            
            for pos, detection in enumerate(detections):
                if conf[pos] >= confidence_threshold:
                    xmin, ymin, xmax, ymax = detection
                    label = f"{classes[int(cls[pos])]} {conf[pos]:.2f}"
                    
                    # Apply the same label modification as in your original code
                    if label.split()[0]:
                        point = label.split()[1]
                        label = 'weapon ' + point
                    
                    # Color based on class
                    color = (0, int(cls[pos]) * 50, 255)
                    
                    # Draw bounding box
                    cv2.rectangle(image_array, 
                                (int(xmin), int(ymin)), 
                                (int(xmax), int(ymax)), 
                                color, 2)
                    
                    # Draw label
                    cv2.putText(image_array, label, 
                              (int(xmin), int(ymin) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.5, color, 1, cv2.LINE_AA)
    
    # Convert back to RGB for display
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    return image_rgb

def main():
    # Title and description
    st.title("🔍 Weapon Detection System")
    st.markdown("""
    This application leverages a YOLO-based deep learning model specifically trained for weapon detection in both images and videos. It is designed to accurately identify various types of weapons in real-time or static input.
    """)
    
    # Load model
    try:
        model = load_model()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {str(e)}")
        st.stop()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📹 Demo Video")
        st.markdown("Here's a demonstration of weapon detection in action:")
        
        # Placeholder for demo video
        # You can replace this path with your actual processed video
        demo_video_path = "results/test_video.mp4"

        if os.path.exists(demo_video_path):
            st.video(demo_video_path)
        else:
            st.info("Demo video will be displayed here. Please add your processed video to the videos folder.")
            # Show a placeholder or sample images from your results
            st.markdown("### Sample Detection Results")
            
            # Display some sample detection results
            sample_images_dir = "./runs/detect/predict"
            if os.path.exists(sample_images_dir):
                sample_images = [f for f in os.listdir(sample_images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if sample_images:
                    # Show first few sample images
                    for i, img_name in enumerate(sample_images[:3]):
                        img_path = os.path.join(sample_images_dir, img_name)
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"Sample Detection: {img_name}", width=300)
    
    with col2:
        st.header("📤 Upload & Test")
        st.markdown("Upload an image to test weapon detection:")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image file (JPG, JPEG, PNG)"
        )
        
        # Confidence threshold slider
        confidence_threshold = st.slider(
            "Detection Confidence Threshold",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum confidence score for detections"
        )
        
        if uploaded_file is not None:
            # Display original image
            st.subheader("Original Image")
            image = Image.open(uploaded_file)
            # Center the image using columns with more space for the image
            img_col1, img_col2, img_col3 = st.columns([0.2, 1.6, 0.2])
            with img_col2:
                st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Center the process button
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 1])
            with btn_col2:
                process_button = st.button("Detect Weapons", type="primary", use_container_width=True)
                
            # Update first column with detection analysis
            with col1:
                st.subheader("Detection Analysis")
                
                # Row 1: Result chart
                result_chart_path = "runs/detect/train/results.png"
                if os.path.exists(result_chart_path):
                    st.subheader("Training Results")
                    st.image(result_chart_path, caption="Model Training Results", use_column_width=True)
                else:
                    st.info("Result chart (result.png) not found. Please ensure the file exists in the project directory.")
                
                # Row 2: Confusion matrix
                confusion_matrix_path = "runs/detect/train/confusion_matrix_normalized.png"
                if os.path.exists(confusion_matrix_path):
                    st.subheader("Confusion Matrix")
                    st.image(confusion_matrix_path, caption="Normalized Confusion Matrix", use_column_width=True)
                else:
                    st.info("Confusion matrix (confusion_matrix_normalized.png) not found. Please ensure the file exists in the project directory.")
            
            if process_button:
                with st.spinner("Processing image..."):
                    try:
                        # Run detection
                        processed_image = detect_objects_in_image(image, model, confidence_threshold)
                        
                        # Display results in second column
                        st.subheader("Detection Results")
                        # Center the result image using columns with more space
                        result_col1, result_col2, result_col3 = st.columns([0.2, 1.6, 0.2])
                        with result_col2:
                            st.image(processed_image, caption="Processed Image with Detections", use_column_width=True)
                        
                        # Option to download result
                        result_pil = Image.fromarray(processed_image)
                        
                        # Save to temporary file for download
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                            result_pil.save(tmp_file.name, 'JPEG')
                            
                            with open(tmp_file.name, 'rb') as file:
                                # Center the download button
                                dl_col1, dl_col2, dl_col3 = st.columns([1, 1, 1])
                                with dl_col2:
                                    st.download_button(
                                        label="📥 Download Result",
                                        data=file.read(),
                                        file_name=f"weapon_detection_result_{uploaded_file.name}",
                                        mime="image/jpeg",
                                        use_container_width=True
                                    )
                        
                        # Clean up temporary file
                        os.unlink(tmp_file.name)
                        
                    except Exception as e:
                        st.error(f"❌ Error processing image: {str(e)}")
    
    # Sidebar with information
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        **Weapon Detection System**
        
        This application uses a YOLOv8-based model trained specifically for weapon detection.
        
        **Features:**
        - Real-time weapon detection
        - Adjustable confidence threshold
        - High accuracy detection
        - Easy-to-use interface
        
        **Supported Formats:**
        - Images: JPG, JPEG, PNG
        - Videos: MP4, AVI (demo section)
        
        **Model Information:**
        - Architecture: YOLOv8
        - Training: Custom weapon dataset
        - Confidence: Adjustable threshold
        """)
        
        st.header("📊 Detection Statistics")
        if 'detection_count' not in st.session_state:
            st.session_state.detection_count = 0
        
        st.metric("Images Processed", st.session_state.detection_count)
        
        # Increment counter when detection is performed
        if uploaded_file is not None and st.button("Reset Counter"):
            st.session_state.detection_count = 0
            st.rerun()

if __name__ == "__main__":
    main()
