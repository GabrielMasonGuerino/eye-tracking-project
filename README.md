# Eye Tracking – Visualização e Classificação de Atenção

Este projeto implementa um pipeline completo de **visualização, processamento e classificação de dados de eye-tracking**, incluindo geração de heatmaps e predição automática de regiões de atenção (AOIs) usando modelos de aprendizado de máquina.

---

## 📌 Funcionalidades

### 🔥 Heatmaps
- Upload de arquivos CSV contendo coordenadas de gaze.
- Seleção do estímulo correspondente (imagem em `/stimuli`).
- Geração automática de heatmaps sobre o estímulo.

### 🎯 Classificação de Atenção
- Predição da região visual onde o olhar está fixado:
  - AOI top-left
  - AOI bottom-right
  - Outras regiões (fora das AOIs)

### 🧠 Pipeline Completo
- Normalização com MinMaxScaler  
- Extração de features (x, y)  
- Classificação via Random Forest  
- Visualização integrada com Streamlit  

---

## 📁 Estrutura do Projeto

```
eye-tracking-project/
│
├── app/
│   └── app.py                # Aplicação Streamlit
│
├── model/
│   ├── rf_attention.pkl      # Modelo Random Forest treinado
│   └── scaler.pkl            # Normalizador MinMaxScaler
│
├── notebooks/
│   └── eye_tracking_model.ipynb
│
├── results/
│   ├── gaze_P01.csv          # Dados coletados
│   ├── gaze_P02.csv
│   └── gaze_P03.csv
│
├── stimuli/
│   ├── stim_A_01.jpg
│   ├── stim_A_02.jpg
│   ├── stim_A_03.jpg
│   ├── stim_B_01.jpg
│   └── stim_B_02.jpg
│
├── requirements.txt
└── README.md
```

---

## 🚀 Como Executar Localmente

### 1. Instalar dependências
```bash
pip install -r requirements.txt
```

### 2. Rodar o aplicativo
```bash
streamlit run app/app.py
```

### 3. Acessar pelo navegador:
```
http://localhost:8501
```

---

## 📊 Exemplos de Resultados

### Heatmap sobre estímulo
O sistema gera automaticamente um heatmap combinando:
- Imagem original do estímulo  
- Densidade das coordenadas de gaze  

---

## 🧪 Modelo de Classificação

O projeto utiliza um **RandomForestClassifier (200 árvores)**.

Features usadas pelo modelo:
- `x_scaled`
- `y_scaled`

As AOIs são definidas em formato normalizado:

```json
{
  "top_left": [0, 0, 0.5, 0.5],
  "bottom_right": [0.5, 0.5, 1, 1]
}
```

---

## ⚙️ Tecnologias Utilizadas

- Python 3.12  
- Streamlit  
- Scikit-learn  
- Pandas / NumPy  
- OpenCV  
- Matplotlib  

---

## 👤 Autor

**Gabriel Mason Guerino**  
Projeto acadêmico envolvendo análise visual e IA aplicada a eye-tracking.

