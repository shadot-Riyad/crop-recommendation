import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import shap

# ১. মডেল এবং অন্যান্য অ্যাসেট লোড করা
try:
    with open('best_crop_recommendation_model.pkl', 'rb') as f:
        saved_assets = pickle.load(f)
    
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
st.write("আপনার মাটির ও আবহাওয়ার উপাদানগুলোর মান স্লাইডার দিয়ে বা সরাসরি বক্সে টাইপ করে সিলেক্ট করুন এবং সঠিক ফসল জেনে নিন।")
st.markdown("---")

# ২. স্লাইডার এবং সরাসরি ভ্যালু বসানোর ইনপুট বক্স (লিংকড)
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### **মাটির উপাদান (Soil Attributes)**")
    
    # Nitrogen (N)
    N_input = st.number_input("Nitrogen (N) - সরাসরি লিখুন", min_value=0, max_value=150, value=50, key="N_num")
    N = st.slider("Nitrogen (N) - স্লাইডার", min_value=0, max_value=150, value=int(N_input), key="N_slide")
    
    # Phosphorus (P)
    P_input = st.number_input("Phosphorus (P) - সরাসরি লিখুন", min_value=5, max_value=150, value=50, key="P_num")
    P = st.slider("Phosphorus (P) - স্লাইডার", min_value=5, max_value=150, value=int(P_input), key="P_slide")
    
    # Potassium (K)
    K_input = st.number_input("Potassium (K) - সরাসরি লিখুন", min_value=5, max_value=210, value=50, key="K_num")
    K = st.slider("Potassium (K) - স্লাইডার", min_value=5, max_value=210, value=int(K_input), key="K_slide")
    
    # pH Level
    ph_input = st.number_input("pH Level - সরাসরি লিখুন", min_value=3.5, max_value=10.0, value=6.5, step=0.1, key="ph_num")
    ph = st.slider("pH Level - স্লাইডার", min_value=3.5, max_value=10.0, value=float(ph_input), step=0.1, key="ph_slide")

with col2:
    st.markdown("#### **আবহাওয়ার উপাদান (Environment)**")
    
    # Temperature
    temp_input = st.number_input("Temperature (°C) - সরাসরি লিখুন", min_value=8.0, max_value=45.0, value=25.0, step=0.1, key="temp_num")
    temperature = st.slider("Temperature (°C) - স্লাইডার", min_value=8.0, max_value=45.0, value=float(temp_input), step=0.1, key="temp_slide")
    
    # Humidity
    hum_input = st.number_input("Humidity (%) - সরাসরি লিখুন", min_value=14.0, max_value=100.0, value=70.0, step=0.1, key="hum_num")
    humidity = st.slider("Humidity (%) - স্লাইডার", min_value=14.0, max_value=100.0, value=float(hum_input), step=0.1, key="hum_slide")
    
    # Rainfall
    rain_input = st.number_input("Rainfall (mm) - সরাসরি লিখুন", min_value=20.0, max_value=300.0, value=100.0, step=0.1, key="rain_num")
    rainfall = st.slider("Rainfall (mm) - স্লাইডার", min_value=20.0, max_value=300.0, value=float(rain_input), step=0.1, key="rain_slide")

st.markdown("---")

# বাংলা নাম দেখানোর জন্য ডিকশনারি ম্যাপিং
crop_bangla_dict = {
    'banana': 'Banana (কলা)', 'barley': 'Barley (বার্লি/যব)', 'bitter gourd': 'Bitter Gourd (করলা)', 'brinjal': 'Brinjal (বেগুন)',
    'cabbage': 'Cabbage (বাধাকপি)', 'cauliflower': 'Cauliflower (ফুলকপি)', 'chili': 'Chili (মরিচ)', 'coriander': 'Coriander (ধনেপাতা)',
    'garlic': 'Garlic (রসুন)', 'ginger': 'Ginger (আদা)', 'jute': 'Jute (পাট)', 'lentil': 'Lentil (মসুর ডাল)',
    'maize': 'Maize (ভুট্টা)', 'mustard': 'Mustard (সরিষা)', 'okra': 'Okra (ঢেঁড়স)', 'onion': 'Onion (পেঁয়াজ)',
    'papaya': 'Papaya (পেঁপে)', 'potato': 'Potato (আলু)', 'pumpkin': 'Pumpkin (কুমড়া)', 'radish': 'Radish (মুলা)',
    'rice': 'Rice (ধান)', 'ridge gourd': 'Ridge Gourd (ঝিংগা)', 'spinach': 'Spinach (পালং শাক)', 'sugarcane': 'Sugarcane (আখ)',
    'sweet gourd': 'Sweet Gourd (মিষ্টি কুমড়া)', 'tea': 'Tea (চা)', 'tomato': 'Tomato (টমেটো)', 'turnip': 'Turnip (শালগম)',
    'watermelon': 'Watermelon (তরমুজ)', 'wheat': 'Wheat (গম)'
}

# ৩. প্রেডিকশন বাটন এবং আউটপুট
if st.button("Recommend the Best Crop 🎯", use_container_width=True):
    try:
        # ইনপুট ডেটাকে সঠিক ফরমেটে (2D Array) নেওয়া
        raw_input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        
        # ডাটা স্কেল (Scale) করা
        scaled_input_data = scaler.transform(raw_input_data)
        
        # модель প্রেডিকশন ও প্রোবাবিলিটি
        probabilities = model.predict_proba(scaled_input_data)[0]
        best_class_idx = np.argmax(probabilities)
        
        # ইংরেজী নাম বের করা
        english_crop_name = le.inverse_transform([best_class_idx])[0]
        
        # ডিকশনারি থেকে বাংলা নাম খোঁজা
        recommended_crop = crop_bangla_dict.get(english_crop_name.lower(), english_crop_name)
        confidence_score = probabilities[best_class_idx] * 100 
        
        # ফলাফল স্ক্রিনে দেখানো 
        st.success(f"🌱 আপনার জমির জন্য সবচেয়ে উপযোগী ফসল হলো: **{recommended_crop}**")
        
        # লাইভ কনফিডেন্স স্কোর প্রোগ্রেস বার
        st.info(f"🎯 **Prediction Confidence:** {confidence_score:.2f}%")
        st.progress(int(confidence_score)) 
        
        # ---- ৪. XAI (SHAP) সেকশন ওয়েবসাইটে প্রদর্শনের জন্য ----
        st.markdown("---")
        st.subheader("📊 Explainable AI (XAI) - সিদ্ধান্তের পেছনের কারণ")
        st.write("নিচের গ্রাফটি ব্যাখ্যা করছে কোন কোন উপাদানের কারণে মডেলটি এই ফসলটিকে রিকমেন্ড করেছে:")
        
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(scaled_input_data)
        
        # মাল্টি-ক্লাস এর জন্য নির্দিষ্ট ক্লাসের SHAP ভ্যালু আলাদা করা
        if isinstance(shap_values, list):
            current_shap = shap_values[best_class_idx][0]
        elif len(shap_values.shape) == 3:
            current_shap = shap_values[0, :, best_class_idx]
        else:
            current_shap = shap_values[0]
            
        # একটি সুন্দর মেটপ্লটলিব বার চার্ট তৈরি করা XAI এর জন্য
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#FF4B4B' if x < 0 else '#4CAF50' for x in current_shap]
        
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, current_shap, color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.invert_yaxis()  # টপ ফিচার উপরে রাখার জন্য
        ax.set_xlabel('SHAP Value (Decision Impact)')
        ax.set_title(f"Why the model chose {english_crop_name.upper()}?")
        
        # স্ট্রিমলিটে ম্যাটপ্লটলিব ফিগার শো করা
        st.pyplot(fig)
        st.caption("💡 সবুজ বার (Positive): এই উপাদানগুলো ফসলটি হওয়ার পক্ষে সিদ্ধান্ত দিয়েছে। লাল বার (Negative): এই উপাদানগুলো বাধা সৃষ্টি করার চেষ্টা করেছে।")
        
    except NameError:
        st.error("মডেল লোড হতে সমস্যা হয়েছে। দয়া করে নিশ্চিত করুন '.pkl' ফাইলটি সঠিক ফোল্ডারে আছে।")
    except Exception as e:
        st.error(f"কোনো সমস্যা হয়েছে: {e}")

# --- ছোট ফুটার (Footer) সেকশন ---
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>Empowering Smart Agriculture through Machine Learning.</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 17px;'>© 2026 Crop Recommendation System | Department of CSE, European University Of Bangladesh</p>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>Developed by Group 7 🌐</p>", unsafe_allow_html=True)
