import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os
from PIL import Image

st.set_page_config(layout="wide", page_title="Eye Tracking Demo")

st.title("Eye Tracking - Visualização e Heatmaps")

uploaded = st.file_uploader("Envie um CSV de gaze (gerado pelo collector)", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Amostras:", len(df))
    st.dataframe(df.head())

    # Painel lateral para Heatmap
    st.sidebar.header("Heatmap")
    stimulus = st.sidebar.text_input("Nome do stimulus (arquivo em /stimuli)", value=df['stimulus'].iloc[0])

    if st.sidebar.button("Gerar heatmap"):
        img_path = os.path.join("stimuli", stimulus)

        if not os.path.exists(img_path):
            st.error("Arquivo não encontrado em /stimuli/" + stimulus)
        else:
            # Carrega a imagem
            bg = cv2.imread(img_path)
            bg_rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
            h, w, _ = bg_rgb.shape

            # Cria heatmap
            heat = np.zeros((h, w), dtype=np.float32)
            df_stim = df[df["stimulus"] == stimulus]

            for _, row in df_stim.iterrows():
                x = int(np.clip(row["x"], 0, w-1))
                y = int(np.clip(row["y"], 0, h-1))
                heat[y, x] += 1

            heat = cv2.GaussianBlur(heat, (0,0), sigmaX=25, sigmaY=25)
            heat_norm = (heat / heat.max() * 255).astype(np.uint8)
            heat_col = cv2.applyColorMap(heat_norm, cv2.COLORMAP_JET)

            overlay = cv2.addWeighted(bg_rgb, 0.6, heat_col, 0.4, 0)

            st.image(overlay, caption="Heatmap overlay gerado com sucesso!")
