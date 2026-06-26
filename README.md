# NASA C-MAPSS Turbofan Engine Health Monitoring

End-to-end predictive maintenance pipeline built on the NASA C-MAPSS dataset as part of Data Science & Analytics, 3rd Year Mechatronics Engineering, UET Lahore.

## What This Project Does
- Predicts **Remaining Useful Life (RUL)** of jet engines in flight cycles using regression
- Classifies engine **health stage** (Healthy / Degrading / Critical) using Random Forest
- Applies **K-Means clustering (k=3)** to discover operating condition patterns in PCA space
- Interactive **Streamlit dashboard** with live sensor input and real-time predictions across 4 tabs

## Tech Stack
Python · Scikit-learn · Pandas · NumPy · Streamlit · Matplotlib · Seaborn · Joblib

## Project Structure
```
nasa-cmapss-engine-health/
├── DSA Project/
│   ├── dashboard/
│   │   ├── app.py
│   │   ├── models/              ← download from Google Drive
│   │   │   ├── regression_model.pkl
│   │   │   ├── classification_model.pkl
│   │   │   ├── kmeans_model.pkl
│   │   │   ├── pca_model.pkl
│   │   │   ├── scaler.pkl
│   │   │   ├── feature_columns.json
│   │   │   └── train_stats.json
│   │   ├── data/                ← download from Google Drive
│   │   │   └── cleaned_data.csv
│   │   └── requirements.txt
│   └── notebook/
│       └── notebook.ipynb
```

## How to Run
**2. Download models & data from Google Drive**  
📁 [Google Drive — Models & Dataset](https://drive.google.com/drive/folders/1Kugxvb7g2LyUGrqWVKkouvrq5S011KKX?usp=sharing)
Place files as shown in the project structure above.
**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**4. Run the dashboard**
```bash
streamlit run app.py
```

## Dataset
NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)  
4 sub-datasets: FD001–FD004 across 6 operating conditions and 2 fault modes (HPC degradation, HPC + Fan degradation)  
128,287 training samples · 27 features after engineering

## Health Stage Thresholds
| Stage | RUL |
|-------|-----|
| 🟢 Healthy | > 80 cycles |
| 🟡 Degrading | 30 – 80 cycles |
| 🔴 Critical | < 30 cycles |

## Key Finding
K-Means clustering discovered **operating condition clusters** rather than health stage clusters — demonstrating that altitude × Mach × throttle combinations dominate sensor variance more than engine degradation does.
