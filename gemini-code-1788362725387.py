import streamlit as st
import pandas as pd
import joblib

# Thiết lập cấu hình trang
st.set_page_config(
    page_title="Diabetes Risk Prediction System",
    page_icon="🩺",
    layout="wide"
)

# Tải mô hình pipeline (gồm tiền xử lý + model)
@st.cache_resource
def load_pipeline():
    return joblib.load("diabetes_pipeline.joblib")

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Không thể tải tệp mô hình `diabetes_pipeline.joblib`. Chi tiết lỗi: {e}")
    st.stop()

# Tiêu đề ứng dụng
st.title("🩺 Hệ Thống Dự Đoán Nguy Cơ Bệnh Tiểu Đường")
st.write("Nhập các chỉ số xét nghiệm lâm sàng của bệnh nhân để dự đoán mức độ rủi ro.")

st.markdown("---")

# Tạo 3 cột để nhập liệu cân đối
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Thông tin cá nhân & BMI")
    gender = st.selectbox("Giới tính (Gender)", options=["F", "M"], help="F: Nữ, M: Nam")
    age = st.number_input("Tuổi (AGE)", min_value=1, max_value=120, value=50, step=1)
    bmi = st.number_input("Chỉ số BMI", min_value=10.0, max_value=60.0, value=30.5, step=0.1)
    hba1c = st.number_input("Chỉ số HbA1c (%)", min_value=3.0, max_value=16.0, value=6.8, step=0.1)

with col2:
    st.subheader("Chức năng thận & Mỡ máu (1)")
    urea = st.number_input("Urea (mmol/L)", min_value=0.0, max_value=50.0, value=4.7, step=0.1)
    cr = st.number_input("Creatinine - Cr (µmol/L)", min_value=0.0, max_value=1000.0, value=46.0, step=1.0)
    chol = st.number_input("Cholesterol (mmol/L)", min_value=0.0, max_value=20.0, value=4.2, step=0.1)
    tg = st.number_input("Triglycerides - TG (mmol/L)", min_value=0.0, max_value=20.0, value=0.9, step=0.1)

with col3:
    st.subheader("Chỉ số Mỡ máu (2)")
    hdl = st.number_input("HDL (mmol/L)", min_value=0.0, max_value=10.0, value=2.4, step=0.1)
    ldl = st.number_input("LDL (mmol/L)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    vldl = st.number_input("VLDL (mmol/L)", min_value=0.0, max_value=10.0, value=0.5, step=0.1)

st.markdown("---")

# Định nghĩa nhãn kết quả
label_map = {
    "N": ("Khỏe mạnh (Non-Diabetic)", "success"),
    "P": ("Tiền tiểu đường (Predict-Diabetic)", "warning"),
    "Y": ("Mắc bệnh tiểu đường (Diabetic)", "error")
}

# Nút thực thi dự đoán
if st.button("🔍 Tiến Hành Chẩn Đoán", type="primary", use_container_width=True):
    # Gom nhóm dữ liệu đầu vào thành DataFrame 1 dòng
    input_data = pd.DataFrame([{
        "Gender": gender,
        "AGE": age,
        "Urea": urea,
        "Cr": cr,
        "HbA1c": hba1c,
        "Chol": chol,
        "TG": tg,
        "HDL": hdl,
        "LDL": ldl,
        "VLDL": vldl,
        "BMI": bmi
    }])

    # Thực thi suy luận qua Pipeline
    pred_class = str(pipeline.predict(input_data)[0]).strip().upper()
    desc, status_type = label_map.get(pred_class, ("Không xác định", "info"))

    st.subheader("📊 Kết Quả Dự Đoán")
    
    if status_type == "success":
        st.success(f"**Phân loại:** {pred_class} — {desc}")
    elif status_type == "warning":
        st.warning(f"**Phân loại:** {pred_class} — {desc}")
    else:
        st.error(f"**Phân loại:** {pred_class} — {desc}")

    # Hiển thị xác suất dự đoán nếu mô hình hỗ trợ
    if hasattr(pipeline, "predict_proba"):
        probs = pipeline.predict_proba(input_data)[0]
        classes = [str(c).strip().upper() for c in pipeline.classes_]
        
        prob_df = pd.DataFrame([probs], columns=classes)
        st.write("**Xác suất chi tiết cho từng lớp:**")
        st.dataframe(prob_df.style.format("{:.2%}"), use_container_width=True)