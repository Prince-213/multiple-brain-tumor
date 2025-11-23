import streamlit as st
import cv2 
import numpy as np
from ultralytics import YOLO
import supervision as sv
import json
import os
from datetime import datetime
import hashlib

# Database file
DB_FILE = "users.json"
HISTORY_FILE = "detection_history.json"

# Initialize database files
def init_database():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, 'w') as f:
            json.dump({}, f)
    
    if not os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'w') as f:
            json.dump({}, f)

# Load database
def load_db():
    try:
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

def load_history():
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except:
        return {}

# Save database
def save_db(db):
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=4)

def save_history(history):
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

# Hash password
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# Initialize databases
init_database()

# Load model with caching
@st.cache_resource
def load_model():
    return YOLO("./best.pt")

# Save detection to history
def save_detection_history(user_id, image_info, results_count, timestamp):
    history = load_history()
    
    if user_id not in history:
        history[user_id] = []
    
    detection_record = {
        "timestamp": timestamp,
        "results_count": results_count,
        "image_info": image_info
    }
    
    history[user_id].append(detection_record)
    save_history(history)

# Get user detection history
def get_user_history(user_id):
    history = load_history()
    return history.get(user_id, [])[::-1]  # Reverse to show latest first

# Authentication functions
def register_user(email, password, name):
    db = load_db()
    
    if email in db:
        return False, "Email already registered"
    
    db[email] = {
        "password_hash": hash_password(password),
        "name": name,
        "user_id": hash_password(email + datetime.now().isoformat())[:16]
    }
    
    save_db(db)
    return True, "Registration successful"

def authenticate_user(email, password):
    db = load_db()
    
    if email not in db:
        return False, "User not found"
    
    if db[email]["password_hash"] != hash_password(password):
        return False, "Invalid password"
    
    return True, db[email]

# Check authentication state
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False
if 'user_info' not in st.session_state:
    st.session_state.user_info = None

if not st.session_state.authenticated:
    # Authentication Page with Tabs
    st.title("Brain Tumor Detection System")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        st.subheader("Login to Your Account")
        
        with st.form("login_form"):
            login_email = st.text_input("Email", key="login_email")
            login_password = st.text_input("Password", type="password", key="login_password")
            login_submitted = st.form_submit_button("Login")
            
            if login_submitted:
                if login_email and login_password:
                    success, result = authenticate_user(login_email, login_password)
                    if success:
                        st.session_state.authenticated = True
                        st.session_state.user_info = result
                        st.success(f"Welcome back, {result['name']}!")
                        st.rerun()
                    else:
                        st.error(result)
                else:
                    st.error("Please fill in all fields")
    
    with tab2:
        st.subheader("Create New Account")
        
        with st.form("register_form"):
            reg_name = st.text_input("Full Name")
            reg_email = st.text_input("Email")
            reg_password = st.text_input("Password", type="password")
            reg_confirm = st.text_input("Confirm Password", type="password")
            reg_submitted = st.form_submit_button("Register")
            
            if reg_submitted:
                if reg_name and reg_email and reg_password:
                    if reg_password == reg_confirm:
                        success, message = register_user(reg_email, reg_password, reg_name)
                        if success:
                            st.success(message)
                            st.info("You can now login with your credentials")
                        else:
                            st.error(message)
                    else:
                        st.error("Passwords do not match")
                else:
                    st.error("Please fill in all fields")

else:
    # Main Application with Dashboard
    user_info = st.session_state.user_info
    model = load_model()
    
    # Sidebar Navigation
    st.sidebar.title(f"Welcome, {user_info['name']}!")
    st.sidebar.subheader("Navigation")
    
    page = st.sidebar.radio("Go to", ["Dashboard", "Upload MRI", "Detection History", "Account Info"])
    
    if page == "Dashboard":
        st.title("Dashboard")
        st.subheader("Brain Tumor Detection System")
        
        col1, col2, col3 = st.columns(3)
        
        # Get user history for stats
        user_history = get_user_history(user_info['user_id'])
        
        with col1:
            st.metric("Total Scans", len(user_history))
        
        with col2:
            positive_detections = len([h for h in user_history if h['results_count'] > 0])
            st.metric("Tumor Detections", positive_detections)
        
        with col3:
            latest_date = user_history[0]['timestamp'] if user_history else "No scans yet"
            st.metric("Latest Scan", latest_date[:10] if latest_date != "No scans yet" else "N/A")
        
        st.markdown("---")
        st.info("Use the navigation menu to upload new MRI scans or view your detection history.")
    
    elif page == "Upload MRI":
        st.title("Upload MRI Scan")
        
        uploaded_file = st.file_uploader("Choose an MRI image file", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Store uploaded image in session state to prevent reloading
            if 'uploaded_image' not in st.session_state or st.session_state.get('file_id') != id(uploaded_file):
                file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                st.session_state.uploaded_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                st.session_state.file_id = id(uploaded_file)
            
            opencv_image = st.session_state.uploaded_image
            
            # Display original image
            st.image(opencv_image, channels="BGR", caption="Uploaded MRI Scan", use_container_width=True)
            
            if st.button("Analyze Scan"):
                with st.spinner("Analyzing MRI scan for tumors..."):
                    # Perform tumor detection
                    results = model(opencv_image, conf=0.75)[0]
                    detections = sv.Detections.from_ultralytics(results)
                    
                    # Get class names from the model
                    class_names = model.names
                    
                    # Check if any tumor detections exist (not "No tumor" class)
                    tumor_detections = []
                    
                    # Filter out "No tumor" detections
                    if len(detections) > 0:
                        for i in range(len(detections)):
                            # Check if the detection is not "No tumor" class
                            class_id = detections.class_id[i]
                            class_name = class_names.get(class_id, f"Class_{class_id}")
                            if class_name.lower() != "no tumor":  # Case-insensitive check
                                tumor_detections.append(i)
                    
                    # Create filtered detections for actual tumors only
                    if tumor_detections:
                        tumor_detections = sv.Detections(
                            xyxy=detections.xyxy[tumor_detections],
                            confidence=detections.confidence[tumor_detections],
                            class_id=detections.class_id[tumor_detections]
                        )
                        num_detections = len(tumor_detections)
                    else:
                        tumor_detections = sv.Detections.empty()
                        num_detections = 0
                    
                    # Apply annotations only if tumors are detected
                    if num_detections > 0:
                        # Create custom labels with class names and confidence
                        labels = []
                        for i in range(len(tumor_detections)):
                            class_id = tumor_detections.class_id[i]
                            class_name = class_names.get(class_id, f"Class_{class_id}")
                            confidence = tumor_detections.confidence[i]
                            labels.append(f"{class_name} ")
                        
                        # Create annotators
                        mask_annotator = sv.MaskAnnotator()
                        # Use custom labels for the label annotator
                        label_annotator = sv.LabelAnnotator(
                            text_position=sv.Position.TOP_LEFT,
                            text_scale=0.7,
                            text_thickness=2
                        )
                        corner_annotator = sv.BoxCornerAnnotator()
                        
                        # Apply annotations only to tumor detections
                        annotated_image = mask_annotator.annotate(
                            scene=opencv_image.copy(), detections=tumor_detections)
                        annotated_image = label_annotator.annotate(
                            scene=annotated_image, detections=tumor_detections, labels=labels)
                        annotated_image = corner_annotator.annotate(
                            scene=annotated_image, detections=tumor_detections)
                        
                        # Convert BGR to RGB for Streamlit display
                        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    else:
                        # No tumors detected, use original image
                        annotated_image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
                    
                    # Save detection to history
                    timestamp = datetime.now().isoformat()
                    image_info = {
                        "filename": uploaded_file.name,
                        "size": f"{opencv_image.shape[1]}x{opencv_image.shape[0]}"
                    }
                    save_detection_history(user_info['user_id'], image_info, num_detections, timestamp)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.image(opencv_image, channels="BGR", 
                                caption="Original Scan", use_container_width=True)
                    with col2:
                        if num_detections > 0:
                            st.image(annotated_image_rgb, 
                                    caption=f"Detected Tumor Regions ({num_detections} found)", use_container_width=True)
                            
                            # Show detailed detection information
                            st.subheader("Detection Details")
                            for i in range(len(tumor_detections)):
                                class_id = tumor_detections.class_id[i]
                                class_name = class_names.get(class_id, f"Class_{class_id}")
                                confidence = tumor_detections.confidence[i]
                                st.write(f"**{class_name}**")
                        else:
                            st.image(annotated_image_rgb, 
                                    caption="No Tumors Detected", use_container_width=True)
                    
                    # Show detection summary
                    if num_detections > 0:
                        st.success(f"✅ Detected {num_detections} potential tumor regions")
                        st.warning("⚠️ Please consult with a medical professional for proper diagnosis.")
                    else:
                        st.success("✅ No tumors detected in this scan")
                        st.info("🎉 This is a good sign, but regular checkups are still recommended.")
    
    elif page == "Detection History":
        st.title("Detection History")
        
        user_history = get_user_history(user_info['user_id'])
        
        if not user_history:
            st.info("No detection history found. Upload your first MRI scan to get started.")
        else:
            st.subheader(f"Your Scan History ({len(user_history)} scans)")
            
            for i, record in enumerate(user_history):
                with st.expander(f"Scan {i+1} - {record['timestamp'][:19]}", expanded=i==0):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**File:** {record['image_info']['filename']}")
                        st.write(f"**Image Size:** {record['image_info']['size']}")
                        st.write(f"**Analysis Date:** {record['timestamp'][:19]}")
                    
                    with col2:
                        if record['results_count'] > 0:
                            st.error(f"**Tumors Detected:** {record['results_count']}")
                        else:
                            st.success("**Result:** No tumors detected")
            
            # Option to clear history
            if st.button("Clear All History"):
                history = load_history()
                if user_info['user_id'] in history:
                    history[user_info['user_id']] = []
                    save_history(history)
                    st.success("History cleared successfully!")
                    st.rerun()
    
    elif page == "Account Info":
        st.title("Account Information")
        
        st.subheader("Personal Details")
        st.write(f"**Name:** {user_info['name']}")
        st.write(f"**Email:** {list(load_db().keys())[list(load_db().values()).index(user_info)]}")
        st.write(f"**User ID:** {user_info['user_id']}")
        
        st.markdown("---")
        st.subheader("Statistics")
        user_history = get_user_history(user_info['user_id'])
        st.write(f"**Total Scans Performed:** {len(user_history)}")
        st.write(f"**Account Created:** Unknown")  # You might want to add registration date to user info
        
        st.markdown("---")
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user_info = None
            st.session_state.clear()
            st.rerun()

# Add footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>"
    "Brain Tumor Detection System • For medical consultation, always consult with healthcare professionals"
    "</div>", 
    unsafe_allow_html=True
)
