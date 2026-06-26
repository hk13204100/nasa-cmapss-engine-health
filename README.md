# NASA C-MAPSS Engine Health Monitoring Dashboard
**MCT-341L Final Project | Roll No: 2023-MC-67**

## Dataset
NASA C-MAPSS Turbofan Engine Degradation Simulation — 160,359 flight 
cycle records from 708 simulated jet engines across 4 sub-datasets.

## Installation
```bash
pip install -r requirements.txt
```

## How to Run
```bash
cd dashboard
streamlit run app.py
```

## Tab Descriptions

**Tab A — Dataset Overview**
Browse the cleaned dataset, view per-column distributions with 
interactive selector, and examine the full correlation heatmap.

**Tab B — Exploratory Analysis**
Select any two features for a scatter plot with optional trend line 
and colour-by-class. Filter data to a numeric range using a slider 
to zoom into subpopulations.

**Tab C — Model Prediction**
Enter engine sensor readings using number inputs to get simultaneous 
RUL prediction (Ridge Regression) and health stage classification 
(Random Forest) with confidence probabilities.

**Tab D — Cluster Explorer**
Visualise K-Means clusters (k=3) in PCA space. Select individual 
clusters to highlight and compare their feature profiles against the 
overall dataset mean.