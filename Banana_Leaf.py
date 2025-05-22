import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

# Load the trained model
MODEL_PATH = "model_lenet_1.h5"  
model = tf.keras.models.load_model(MODEL_PATH)

# Define class labels
class_names = ["Cordana", "Healthy", "Panama", "Pestalotiopsis", "Sigatoka"]

# Define treatment recommendations
treatment_data = {
    "Cordana": {
        "Chemical Treatment": "Mancozeb (0.4%)",
        "Natural Treatment": "Neem Oil Spray (1-2 tbsp/L water, every 7-14 days)",
        "Cultural Control": "Remove infected leaves"
    },
    "Panama": {
        "Chemical Treatment": "Tebuconazole or Propiconazole",
        "Natural Treatment": "Bacillus subtilis application",
        "Cultural Control": "Uproot and destroy infected plants"
    },
    "Pestalotiopsis": {
        "Chemical Treatment": "Chlorothalonil fungicide",
        "Natural Treatment": "Compost tea foliar spray",
        "Cultural Control": "Prune infected leaves & maintain field hygiene"
    },
    "Sigatoka": {
        "Chemical Treatment": "Copper Oxychloride (0.2-0.4%) OR Spray 3 times with Carbendazim 0.1%, Propiconazole 0.1%, or Mancozeb 0.25%",
        "Natural Treatment": "Neem Oil & Bacillus subtilis",
        "Cultural Control": "Remove & burn infected pseudostems"
    }
}

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = ImageOps.fit(image, (150, 150))  
    img_array = np.asarray(image)
    img_array = img_array.astype(np.float32) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# Streamlit UI
st.title("🍌 Banana Disease Detection & Prevention")

uploaded_file = st.file_uploader("📤 Upload a banana leaf image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)
    st.write("🔍 **Processing...**")

    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    disease_name = class_names[predicted_class]

    # Stylish Prediction Result
    if disease_name == "Healthy":
        st.markdown(f"""
            <div style="display: flex; align-items: center; background-color: #34A853; 
                        padding: 10px; border-radius: 8px; color: white; font-size: 22px;
                        font-weight: bold; text-align: center; width: 60%; margin: auto;">
                ✅ Prediction: {disease_name} (No disease detected!)
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
            <div style="display: flex; align-items: center; background-color: #EA4335; 
                        padding: 10px; border-radius: 8px; color: white; font-size: 22px;
                        font-weight: bold; text-align: center; width: 60%; margin: auto;">
                ⚠️ Prediction: {disease_name} (Disease detected!)
            </div>
        """, unsafe_allow_html=True)

    # Display Treatment Recommendations
    if disease_name in treatment_data:
        st.write("### 🩺 Recommended Treatments")

        # Add CSS for styling the Treatment column
        st.markdown("""
            <style>
                table {
                    width: 100%;
                    border-collapse: collapse;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 10px;
                    text-align: left;
                    
                }
                th {
                    background-color: #28a745;
                    color: white;  /* White font for column names */
                }
                
            </style>
        """, unsafe_allow_html=True)

        # Convert treatment data into a DataFrame
        treatment_df = pd.DataFrame(list(treatment_data[disease_name].items()), columns=["Treatment Type", "Treatment"])
        
        # Make the first column (Treatment Type) bold
        treatment_df["Treatment Type"] = treatment_df["Treatment Type"].apply(lambda x: f"**{x}**")

        # Render table in Streamlit
        st.markdown(treatment_df.to_markdown(index=False), unsafe_allow_html=True)

    else:
        st.write("✅ **No disease detected. The leaf appears healthy!**")
