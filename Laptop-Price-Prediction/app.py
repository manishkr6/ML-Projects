import streamlit as st
import pickle
import numpy as np

# Load model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")
st.markdown("## 💻 Laptop Price Prediction Tool")
st.write("Fill in the details below and click **Predict Price** to estimate the market price for your laptop configuration.")

# Brand & Type
with st.container():
    col1, col2 = st.columns(2)
    company = col1.selectbox('Select Laptop Brand', df['Company'].unique())
    type_name = col2.selectbox('Laptop Category', df['TypeName'].unique())

# RAM & Weight
with st.container():
    col1, col2 = st.columns(2)
    ram = col1.selectbox('RAM (GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
    weight = col2.number_input('Weight (kg)', min_value=0.5, max_value=5.0, value=1.5, step=0.1, help="Approximate laptop weight")

# Display Features
with st.container():
    st.markdown("### Display Configuration")
    col1, col2, col3 = st.columns(3)
    touchscreen = col1.radio('Touchscreen', ['No', 'Yes'])
    ips = col2.radio('IPS Panel', ['No', 'Yes'])
    screen_size = col3.slider('Screen Size (inches)', min_value=10.0, max_value=18.0, value=13.0, step=0.1)

    resolution = st.selectbox('Screen Resolution', [
        '1920x1080','1366x768','1600x900','3840x2160','3200x1800',
        '2880x1800','2560x1600','2560x1440','2304x1440'
    ])

# CPU, Storage, GPU
with st.container():
    st.markdown("### Performance Features")
    col1, col2 = st.columns(2)
    cpu = col1.selectbox('CPU Brand', df['Cpu_brand'].unique())
    gpu = col2.selectbox('GPU Brand', df['Gpu brand'].unique())

    col3, col4 = st.columns(2)
    hdd = col3.selectbox('HDD (GB)', [0, 128, 256, 512, 1024, 2048])
    ssd = col4.selectbox('SSD (GB)', [0, 8, 128, 256, 512, 1024])

# OS
os = st.selectbox('Operating System', df['os'].unique())

if st.button('Predict Price'):
    # Convert categorical to numerical where needed
    touchscreen_val = 1 if touchscreen == 'Yes' else 0
    ips_val = 1 if ips == 'Yes' else 0

    # Calculate PPI (pixels per inch)
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2 + Y_res**2)**0.5) / screen_size

    # Create query array
    query = np.array([
        company, type_name, ram, weight, touchscreen_val, ips_val, ppi,
        cpu, hdd, ssd, gpu, os
    ]).reshape(1, 12)

    # Predict and display result
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    st.success(f"💸 **Estimated Price:** ₹ {predicted_price:,}")

    st.caption("Price prediction is based on current data and configuration. Actual market prices may vary.")