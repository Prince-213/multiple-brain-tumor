import streamlit as st
import cv2 
import numpy as np
from ultralytics import YOLO
import supervision as sv

# Load model with caching
@st.cache_resource
def load_model():
    return YOLO("./best.pt")

# Check authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    # Login Page
    st.title("Brain Tumor Detection Login")
    
    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Get Started")
        
        if submitted:
            # Simple authentication (replace with proper authentication in production)
            if email == "user@gmail.com" and password == "123":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Invalid credentials. Please try again.")

else:
    # Main Application Page
    model = load_model()
    st.title("Brain Tumor Detection")
    
    uploaded_file = st.file_uploader("Upload an MRI scan", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Store uploaded image in session state to prevent reloading
        if 'uploaded_image' not in st.session_state or st.session_state.get('file_id') != id(uploaded_file):
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            st.session_state.uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.session_state.file_id = id(uploaded_file)
        
        opencv_image = st.session_state.uploaded_image
        
        # Display original image
        st.image(opencv_image, channels="BGR", caption="Uploaded MRI Scan", use_column_width=True)
        
        if st.button("Analyze Scan"):
            # Perform tumor detection
            results = model(opencv_image)[0]
            detections = sv.Detections.from_ultralytics(results)
            
            # Create annotators
            mask_annotator = sv.MaskAnnotator()
            label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)
            corner_annotator = sv.BoxCornerAnnotator()
            
            # Apply annotations
            annotated_image = mask_annotator.annotate(
                scene=opencv_image.copy(), detections=detections)
            annotated_image = label_annotator.annotate(
                scene=annotated_image, detections=detections)
            annotated_image = corner_annotator.annotate(
                scene=annotated_image, detections=detections)
            
            # Convert BGR to RGB for Streamlit display
            annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            
            # Display results
            st.subheader("Analysis Results")
            col1, col2 = st.columns(2)
            with col1:
                st.image(opencv_image, channels="BGR", 
                        caption="Original Scan", use_column_width=True)
            with col2:
                st.image(annotated_image_rgb, 
                        caption="Detected Tumor Regions", use_column_width=True)
            
            # Show detection summary
            if len(detections) > 0:
                st.success(f"Detected {len(detections)} potential tumor regions")
            else:
                st.info("No tumors detected in this scan")

    # Add logout button
    if st.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.clear()
        st.rerun()
