import streamlit as st
import joblib
import numpy as np

model = joblib.load('LYKN_model.pkl')

st.set_page_config(page_title="LYKN Views Prediction", page_icon="🐺", layout="centered")

st.markdown("""
    <style>
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    /* ปรับสีปุ่ม */
    div.stButton > button:first-child {
        background-color: #d32f2f;
        color: white;
        border-radius: 10px;
    }
    </style>        
    """, unsafe_allow_html=True)

st.title("LYKN Views Prediction🐺")
st.write("ระบบทำนายยอดวิวเพลงของวง LYKN")

st.subheader("กรอกข้อมูลเพลง")
col1, col2 = st.columns(2)

with col1:
    days = st.number_input("ปล่อยเพลงมาแล้วกี่วัน", min_value=1, value=30)
    length = st.number_input("ความยาวเพลง (วินาที)", min_value=60, value=200)

with col2:
    members = st.selectbox("จำนวนสมาชิกในวง", options=[5, 6, 7])
    trend = st.slider("ระดับกระแสโซเชียล(1-5)", 1, 5, 3)

if st.button("ทำนายยอดวิว"):
    input_data = np.array([[days, length, members, trend]])
    prediction = model.predict(input_data)

    result = max(0, prediction[0])

    st.success(f"ยอดวิวที่คาดการณ์ได้คือ: {result:.2f} ล้านวิว")
    st.info("อ้างอิงข้อมูลชุด Training จาก th.wikipedia.org/wiki/LYKN")