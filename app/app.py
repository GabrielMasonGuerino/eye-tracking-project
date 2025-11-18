# app.py
import streamlit as st
import pandas as pd, numpy as np, joblib, os, cv2
from PIL import Image
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Eye-Tracking Demo")

st.title("Eye Tracking - Visualização e Classificação de Atenção")

model_path = 'model/rf_attention.pkl'
scaler_path = 'model/scaler.pkl'

if os.path.exists(model_path):
    clf = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
else:
    st.warning("Modelo não encontrado. Treine e salve o modelo no diretório /model")
    clf = None

uploaded = st.file_uploader("Envie um CSV de gaze (gerado pelo collector)", type=['csv'])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Amostras:", len(df))
    st.dataframe(df.head())

    st.sidebar.header("Heatmap")
    stimulus = st.sidebar.text_input("Nome do stimulus (arquivo em /stimuli)", value=df['stimulus'].iloc[0])
    if st.sidebar.button("Gerar heatmap"):
        # gerar heatmap rápido
        try:
            bg = cv2.imread(os.path.join('..','stimuli', stimulus))
            if bg is None:
                st.error("Stimulus não encontrado em /stimuli/" + stimulus)
            else:
                h,w,_ = bg.shape
                heat = np.zeros((h,w), dtype=np.float32)
                for _,r in df[df['stimulus']==stimulus].iterrows():
                    x = int(np.clip(r['x'],0,w-1)); y=int(np.clip(r['y'],0,h-1))
                    heat[y,x] += 1
                heat = cv2.GaussianBlur(heat, (0,0), sigmaX=25, sigmaY=25)
                heat_norm = (heat/heat.max()*255).astype(np.uint8)
                heat_col = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)
                overlay = cv2.addWeighted(bg, 0.6, heat_col, 0.4, 0)
                st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Heatmap overlay")
        except Exception as e:
            st.error(str(e))

    st.sidebar.header("Classificação")
    if clf is not None:
        # Aggregate features quickly (global)
        fix_count = df.shape[0]
        total_fix_time = df['timestamp_ms'].max() - df['timestamp_ms'].min()
        mean_fix = total_fix_time / max(1, fix_count)
        median_fix = mean_fix
        revisits = max(0, fix_count-1)
        X = np.array([[fix_count, total_fix_time, mean_fix, median_fix, revisits]])
        Xs = scaler.transform(X)
        ypred = clf.predict(Xs)
        proba = clf.predict_proba(Xs)[0,1] if hasattr(clf,'predict_proba') else None
        st.write("Predição: ", "Alta atenção" if ypred[0]==1 else "Baixa atenção")
        if proba is not None:
            st.write("Probabilidade (alta): {:.2f}".format(proba))
