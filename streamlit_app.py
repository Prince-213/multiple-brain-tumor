import streamlit as st
import pandas as pd
import cv2 
import numpy as np
from PIL import Image
from ultralytics import YOLO

import supervision as sv





# Load a pretrained YOLO11n model


st.title('Brain tumor detection')

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])


    
if uploaded_file is not None:
    # Read the uploaded image
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    #model.predict(opencv_image, save=True, imgsz=320, conf=0.5)
    def predict():
      model = YOLO("./best.pt")
      #image = cv2.imread(opencv_image)
      results = model(opencv_image)[0]
      detections = sv.Detections.from_ultralytics(results)

      mask_annotator = sv.MaskAnnotator()
      label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT)

      corner_annotator = sv.BoxCornerAnnotator()
      
      annotated_image = mask_annotator.annotate(
          scene=opencv_image, detections=detections)
      annotated_image = label_annotator.annotate(
          scene=annotated_image, detections=detections)
      annotated_image = corner_annotator.annotate(
        scene=annotated_image, detections=detections
      )
      output_path = './annotated_dog.jpeg'
      cv2.imwrite(output_path, annotated_image)

      print(f"Annotated image saved at {output_path}")
      
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR", caption="Original Image", width=400,)
    
    # Save the upscaled image
    st.markdown("### Download Upscaled Image")
    
   
    if st.button("Run Detection", on_click=predict()):
    
      st.write("The Segmentation Result")
      st.image('./annotated_dog.jpeg', channels="BGR", caption="Tumor Segmented Image")
