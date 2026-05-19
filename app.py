import streamlit as st
import pickle
import numpy as np

# ১. মডেল এবং অন্যান্য অ্যাসেট লোড করা
try:
    with open('best_crop_recommendation_model.pkl', 'rb') as f:
        saved_assets = pickle.load(f)
    
    # ডিকশনারি থেকে উপাদানগুলো আলাদা করা
    model = saved_assets['model']
    scaler = saved_assets['scaler']
    le = saved_assets['label_encoder']
    feature_names = saved_assets['feature_names']
    
except FileNotFoundError:
    st.error("Error: 'best_crop_recommendation_model.pkl' ফাইলটি খুঁজে পাওয়া যায়নি! ফাইলটি আপনার প্রজেক্ট ফোল্ডারে রাখুন।")

# ওয়েবসাইটের ইন্টারফেস ডিজাইন
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

# --- টিম নাম (Team Name) সেকশন ---
st.markdown("<p style='text-align: center; color: gray; font-size: 14px; margin-bottom: 0px;'>Created By</p>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #4CAF50; margin-top: 0px; margin-bottom: 20px;'>Group 7</h3>", unsafe_allow_html=True)

st.title("🌾 Crop Recommendation System")
st.write("আপনার মাটির ও আবহাওয়ার উপাদানগুলোর মান স্লাইডারের সাহায্যে সিলেক্ট করুন এবং সঠিক ফসল জেনে নিন।")
st.markdown("---")

# ২. স্লাইডার তৈরি করা 
col1, col2 = st.columns(2)

with col1:
    N = st.slider("Nitrogen (N) - নাইট্রোজেন", min_value=0, max_value=150, value=50)
    P = st.slider("Phosphorus (P) - ফসফরাস", min_value=5, max_value=150, value=50)
    K = st.slider("Potassium (K) - পটাশিয়াম", min_value=5, max_value=210, value=50)
    ph = st.slider("pH Level - মাটির অম্লতা", min_value=3.5, max_value=10.0, value=6.5, step=0.1)

with col2:
    temperature = st.slider("Temperature (°C) - তাপমাত্রা", min_value=8.0, max_value=45.0, value=25.0, step=0.1)
    humidity = st.slider("Humidity (%) - আর্দ্রতা", min_value=14.0, max_value=100.0, value=70.0, step=0.1)
    rainfall = st.slider("Rainfall (mm) - বৃষ্টিপাত", min_value=20.0, max_value=300.0, value=100.0, step=0.1)

st.markdown("---")

# ৩. প্রেডিকশন বাটন এবং আউটপুট
if st.button("Recommend the Best Crop 🎯", use_container_width=True):
    try:
        # ইনপুট ডেটাকে সঠিক ফরমেটে (2D Array) নেওয়া
        raw_input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # থিসিস মডেলের নিয়ম অনুযায়ী ইনপুট ডেটাকে স্কেল (Scale) করা
        scaled_input_data = scaler.transform(raw_input_data)
        
        # মডেল থেকে লাইভ প্রেডিকশন ও প্রোবাবিলিটি নেওয়া
        probabilities = model.predict_proba(scaled_input_data)[0]
        best_class_idx = np.argmax(probabilities)
        
        # Label Encoder থেকে সরাসরি আসল ফসলের নাম ডাইনামিকালি বের করা
        recommended_crop = le.inverse_transform([best_class_idx])[0]
        confidence_score = probabilities[best_class_idx] * 100 
        
        # ফলাফল স্ক্রিনে দেখানো 
        st.success(f"🌱 আপনার জমির জন্য সবচেয়ে উপযোগী ফসল হলো: **{recommended_crop.upper()}**")
        
        # লাইভ কনফিডেন্স স্কোর প্রোগ্রেস বার আকারে দেখানো
        st.info(f"🎯 **Prediction Confidence:** {confidence_score:.2f}%")
        st.progress(int(confidence_score)) 
        
    except NameError:
        st.error("মডেল লোড হতে সমস্যা হয়েছে। দয়া করে নিশ্চিত করুন '.pkl' ফাইলটি সঠিক ফোল্ডারে আছে।")
    except Exception as e:
        st.error(f"কোনো সমস্যা হয়েছে: {e}")

# --- ছোট ফুটার (Footer) সেকশন ---
st.markdown("<br><br><br>", unsafe_allow_html=True) # কিছুটা খালি জায়গা তৈরি করার জন্য
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>Empowering Smart Agriculture through Machine Learning.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 17px;'>© 2026 Crop Recommendation System | Department of CSE, European University Of Bangladesh</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>Developed by Group 7 🌐</p>", unsafe_allow_html=True)
