import streamlit as st
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="AI Sentiment & Extraction Audit",
    page_icon="📊",
    layout="wide"
)

# Custom Styling to match dashboard UI
st.markdown("""
    <style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        border-left: 5px solid #4e73df;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .stMetric label {
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

st.title("📊 AI ML Extraction & Sentiment Analysis Audit")
st.caption("Performance benchmarking (Gemini3 vs. Legacy Bak3) and sentiment extractions catalog.")

# Sidebar Filters & Uploaders
with st.sidebar:
    st.header("⚙️ Data Configuration")
    
    audit_file = st.file_uploader("Upload Audit File (Excel)", type=["xlsx"])
    
    st.divider()
    st.subheader("Filter Options")
    selected_brand = st.text_input("Filter by Brand", value="")
    hitl_filter = st.selectbox("HITL Override Status", ["All", "Validated Only", "Ignored Only"])

# Dummy / Default Data Load
@st.cache_data
def get_sample_audit_data():
    return pd.DataFrame({
        "Source.Name": ["sample_01.json", "sample_02.json", "sample_03.json", "sample_04.json"],
        "Products_Brand": ["BrandA", "BrandB", "BrandA", "BrandC"],
        "Products_Model": ["Model-X", "Model-Y", "Model-Z", "Model-W"],
        "Present_Gemini3": ["Y", "Y", "", "Y"],
        "Ignore_by_HITL": ["", "Y", "", ""],
        "Bak3_Products_Brand": ["BrandA", "BrandB", "BrandA", "BrandC"],
        "Bak3_Products_Model": ["Model-X", "Model-Y", "Model-Z", "Model-W"],
        "Present_Bak3": ["Y", "", "Y", "Y"],
        "Bak3_Ignore_by_HITL": ["", "", "", "Y"]
    })

if audit_file is not None:
    df_audit = pd.read_excel(audit_file, sheet_name="Extractions")
else:
    df_audit = get_sample_audit_data()

# Data Filtering
filtered_df = df_audit.copy()
if selected_brand:
    filtered_df = filtered_df[filtered_df["Products_Brand"].str.contains(selected_brand, case=False, na=False)]

if hitl_filter == "Validated Only":
    filtered_df = filtered_df[filtered_df["Ignore_by_HITL"] == ""]
elif hitl_filter == "Ignored Only":
    filtered_df = filtered_df[filtered_df["Ignore_by_HITL"] == "Y"]

# Measure Calculations (DAX Equivalent)
def calc_metrics(df, model_col, hitl_col):
    tp = len(df[(df[model_col] == "Y") & (df[hitl_col] == "")])
    fp = len(df[(df[model_col] == "Y") & (df[hitl_col] == "Y")])
    fn = len(df[(df[model_col] == "") & (df[hitl_col] == "")])
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return tp, fp, fn, precision, recall, f1

tp_g, fp_g, fn_g, prec_g, rec_g, f1_g = calc_metrics(filtered_df, "Present_Gemini3", "Ignore_by_HITL")
tp_b, fp_b, fn_b, prec_b, rec_b, f1_b = calc_metrics(filtered_df, "Present_Bak3", "Bak3_Ignore_by_HITL")

# Main Layout
tab1, tab2 = st.tabs(["🎯 Model Benchmarking (Audit)", "🔍 Data Dictionary & Inspection"])

with tab1:
    st.subheader("Model Performance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🟢 Gemini3 Engine")
        m1, m2, m3 = st.columns(3)
        m1.metric("Precision", f"{prec_g:.1%}")
        m2.metric("Recall", f"{rec_g:.1%}")
        m3.metric("F1 Score", f"{f1_g:.1%}")
        
        st.caption(f"**True Positives:** {tp_g} | **False Positives:** {fp_g} | **False Negatives:** {fn_g}")

    with col2:
        st.markdown("### 🔵 Bak3 Engine (Legacy)")
        m4, m5, m6 = st.columns(3)
        m4.metric("Precision", f"{prec_b:.1%}")
        m5.metric("Recall", f"{rec_b:.1%}")
        m6.metric("F1 Score", f"{f1_b:.1%}")
        
        st.caption(f"**True Positives:** {tp_b} | **False Positives:** {fp_b} | **False Negatives:** {fn_b}")

    st.divider()
    st.subheader("Extraction Audit Log")
    st.dataframe(filtered_df, use_container_width=True)

with tab2:
    st.subheader("Audit Data Schema")
    schema_df = pd.DataFrame([
        {"Field": "Products_Model", "Type": "String", "Role": "Foreign Key", "Description": "Model name reference"},
        {"Field": "Present_Gemini3", "Type": "String", "Role": "Extraction Status", "Description": "'Y' if extracted by Gemini3"},
        {"Field": "Ignore_by_HITL", "Type": "String", "Role": "HITL Status", "Description": "'Y' if flagged as false extraction by human reviewer"}
    ])
    st.table(schema_df)
