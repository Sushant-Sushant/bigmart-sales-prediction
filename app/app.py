import streamlit as st
import pandas as pd
from utils import preprocess_input
from predictions import load_model, predict_sales
import shap
import matplotlib.pyplot as plt


st.set_page_config(page_title="Big Mart Sales Prediction", layout="centered")
st.title("🛒 Big Mart Sales Predictor")

model = load_model()

with st.form("prediction_form"):
    item_weight = st.number_input("Item Weight", min_value=0.0)
    item_fat_content = st.selectbox("Item Fat Content", ["Low Fat", "Regular"])
    item_visibility = st.slider("Item Visibility", 0.0, 0.2, 0.05)
    item_type = st.selectbox("Item Type", ["Fruits and Vegetables", "Dairy", "Baking Goods", "Soft Drinks", "Household", "Meat"])
    item_mrp = st.number_input("Item MRP", min_value=0.0)
    outlet_size = st.selectbox("Outlet Size", ["Small", "Medium", "High"])
    outlet_location_type = st.selectbox("Outlet Location Type", ["Tier 1", "Tier 2", "Tier 3"])
    outlet_type = st.selectbox("Outlet Type", ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"])
    outlet_est_year = st.number_input("Outlet Establishment Year", min_value=1980, max_value=2025, value=2005)


    submitted = st.form_submit_button("Predict")

if submitted:
    input_dict = {
        "Item_Weight": item_weight,
        "Item_Fat_Content": item_fat_content,
        "Item_Visibility": item_visibility,
        "Item_Type": item_type,
        "Item_MRP": item_mrp,
        "Outlet_Size": outlet_size,
        "Outlet_Location_Type": outlet_location_type,
        "Outlet_Type": outlet_type,
        "Outlet_Establishment_Year": outlet_est_year
    }

    input_df = preprocess_input(input_dict)
    prediction = predict_sales(model, input_df)
    st.success(f"Predicted Sales: ₹ {round(prediction, 2)}")

    # SHAP explanation
    explainer = shap.Explainer(model.named_steps["model"])
    transformed = model.named_steps["preprocessor"].transform(input_df)
    transformed_dense = transformed.toarray() 
    transformed = pd.DataFrame(transformed_dense)
    transformed = transformed.apply(pd.to_numeric, errors='coerce')
    transformed = transformed.fillna(0)  # or transformed.dropna()
    transformed = transformed.astype(float)



    shap_values = explainer(transformed)

   # st.subheader("📊 SHAP Feature Importance")
    #st.set_option('deprecation.showPyplotGlobalUse', False)
    
    
   # shap.plots.waterfall(shap_values[0], show=False)
    #st.pyplot(bbox_inches="tight")
