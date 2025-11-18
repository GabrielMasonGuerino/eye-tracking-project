import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter

st.set_page_config(layout="wide", page_title="Eye Tracking Demo")

st.title("Eye Tracking - Visualização e Heatmaps")

uploaded = st.file_uploader("Envie um CSV de gaze (gerado pelo collector)", type=['csv'])

if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Amostras:", len(df))
    st.dataframe(df.head())

    st.sidebar.header("Heatmap")
    stimulus = st.sidebar.text_input("Nome do stimulus (arquivo em /stimuli)", value=df["stimulus"].iloc[0])

    if st.sidebar.button("Gerar heatmap"):
        img_path = os.path.join("stimuli", stimulus)

        if not os.path.exists(img_path):
            st.error("Arquivo não encontrado: " + img_path)
        else:
            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            heat = np.zeros((h, w))

            df_stim = df[df["stimulus"] == stimulus]

            for _, row in df_stim.iterrows():
                x = int(np.clip(row["x"], 0, w - 1))
                y = int(np.clip(row["y"], 0, h - 1))
                heat[y, x] += 1

            heat = gaussian_filter(heat, sigma=25)
            heat_norm = heat / (heat.max() + 1e-9)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.imshow(img)
            ax.imshow(heat_norm, cmap="jet", alpha=0.4)
            ax.axis("off")

            st.pyplot(fig)
