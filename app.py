# import the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import streamlit as st
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="VinoPredic | Premium Wine Quality",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@400;700&display=swap');

    /* Global Changes */
    .stApp {
        font-family: 'Lato', sans-serif;
    }
    
    /* Headings */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #4A1C25;
    }
    
    h1 {
        font-size: 3.5rem;
        font-weight: 700;
        background: -webkit-linear-gradient(#722F37, #A52A2A);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }

    /* Cards */
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        text-align: center;
        border-left: 5px solid #722F37;
    }
    
    .feature-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.03);
        margin-bottom: 10px;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8F5F2;
        border-right: 1px solid #E6DCD3;
    }
    
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #4A1C25 !important;
    }

    [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #4A1C25 !important;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #722F37 0%, #900C3F 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-size: 1.1rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(114, 47, 55, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(114, 47, 55, 0.3);
        background: linear-gradient(135deg, #900C3F 0%, #722F37 100%);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: white !important;
        border-radius: 8px;
        color: #4A1C25 !important;
    }
    
    .streamlit-expanderContent {
        background-color: white !important;
        color: #4A1C25 !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the dataset."""
    df = pd.read_csv('winequality-red.csv')
    return df

@st.cache_resource
def train_model(df):
    """Train and cache the model."""
    X = df.drop('quality', axis=1)
    # Binary classification: 7 or higher is 'good' (1), else 'bad' (0)
    y = df['quality'].apply(lambda value: 1 if value >= 7 else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc_score = accuracy_score(y_pred, y_test)
    
    return model, acc_score

def main():
    # Header Section
    col_logo, col_title = st.columns([1, 4])
    with col_title:
        st.markdown("<h1>VinoPredic</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size: 1.2rem; color: #666;'>Premier Wine Quality Assessment AI</p>", unsafe_allow_html=True)

    try:
        df = load_data()
        model, acc_score = train_model(df)
    except Exception as e:
        st.error(f"Error initializing system: {e}")
        return

    # Sidebar: Input Panel
    with st.sidebar:
        # Fixed: Updated deprecated parameter use_column_width to use_container_width
        st.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?ixlib=rb-1.2.1&auto=format&fit=crop&w=500&q=60", use_container_width=True)
        
        st.markdown("## ⚙️ Wine Configuration")
        st.markdown("---")
        
        input_data = {}
        feature_names = df.columns[:-1]
        
        # Group sliders into expandable sections for cleaner UI
        with st.expander("⚗️ Chemical Properties", expanded=True):
            for feature in feature_names[:5]:
                label = feature.replace('_', ' ').title()
                min_v, max_v = float(df[feature].min()), float(df[feature].max())
                mean_v = float(df[feature].mean())
                input_data[feature] = st.slider(label, min_v, max_v, mean_v, (max_v-min_v)/100, key=feature)

        with st.expander("🧪 Physical Properties", expanded=False):
            for feature in feature_names[5:]:
                label = feature.replace('_', ' ').title()
                min_v, max_v = float(df[feature].min()), float(df[feature].max())
                mean_v = float(df[feature].mean())
                input_data[feature] = st.slider(label, min_v, max_v, mean_v, (max_v-min_v)/100, key=feature)
    
    input_df = pd.DataFrame(input_data, index=[0])

    # Main Dashboard
    st.markdown("---")
    
    # 1. Prediction Hero Section
    st.markdown("### 🔍 Live Analysis")
    
    col_pred_l, col_pred_m, col_pred_r = st.columns([1, 2, 1])
    
    with col_pred_m:
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.button("✨ Evaluate Wine Quality"):
            with st.spinner("Analyzing chemical composition..."):
                prediction = model.predict(input_df)[0]
                prediction_proba = model.predict_proba(input_df)[0][prediction]
                
                st.markdown("<br>", unsafe_allow_html=True)
                if prediction == 1:
                    st.success("Analysis Complete")
                    st.markdown(f"""
                        <div class="metric-card" style="background-color: #E8F5E9; border-left: 5px solid #2E7D32;">
                            <h2 style="color: #2E7D32; margin:0;">🌟 EXQUISITE</h2>
                            <p style="font-size: 1.1rem; margin-top: 10px;">This wine exhibits superior characteristics.</p>
                            <p style="font-size: 0.9rem; color: #666;">Confidence: {prediction_proba:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
                    st.balloons()
                else:
                    st.warning("Analysis Complete")
                    st.markdown(f"""
                        <div class="metric-card" style="background-color: #FFEBEE; border-left: 5px solid #C62828;">
                            <h2 style="color: #C62828; margin:0;">📉 MEDIOCRE</h2>
                            <p style="font-size: 1.1rem; margin-top: 10px;">This wine falls below premium standards.</p>
                            <p style="font-size: 0.9rem; color: #666;">Confidence: {prediction_proba:.2%}</p>
                        </div>
                    """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # 2. Data Insights Section
    st.markdown("### 📊 Dataset Insights")
    tab1, tab2, tab3 = st.tabs(["Overview", "Correlation Matrix", "Distribution"])
    
    with tab1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Samples", df.shape[0])
        c2.metric("Features", df.shape[1]-1)
        c3.metric("Model Accuracy", f"{acc_score:.2%}")
        st.dataframe(df.head(), use_container_width=True)
        
    with tab2:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(df.corr(), annot=True, cmap='RdBu', fmt=".2f", ax=ax, linewidths=0.5)
        st.pyplot(fig)
        
    with tab3:
        cols = st.columns(2)
        with cols[0]:
            fig2 = plt.figure(figsize=(6, 4))
            sns.countplot(x='quality', data=df, palette='Reds_r')
            plt.title("Quality Distribution")
            st.pyplot(fig2)
        with cols[1]:
             fig3 = plt.figure(figsize=(6, 4))
             sns.boxplot(x='quality', y='alcohol', data=df, palette='Reds_r')
             plt.title("Alcohol vs Quality")
             st.pyplot(fig3)

if __name__ == '__main__':
    main()
