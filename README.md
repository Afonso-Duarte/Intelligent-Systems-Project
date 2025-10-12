# Intelligent-Systems-Project
Project and Assignments done for the Intelligent Systems Course 25/26

AirQualityPrediction/
│
├── data/
│   ├── raw/                     # Original dataset from UCI (unmodified)
│   ├── processed/               # Cleaned and preprocessed data (after handling -200, scaling, etc.)
│   ├── drift_analysis/          # Files/results related to drift detection or correction
│   └── README.md                # Notes on data sources, preprocessing steps, etc.
│
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_data_preprocessing.ipynb
│   ├── 03_fuzzy_model.ipynb
│   ├── 04_neuro_fuzzy_model.ipynb
│   ├── 05_deep_models_lstm_cnn.ipynb
│   └── 06_evaluation_and_comparison.ipynb
│
├── src/
│   ├── __init__.py
│   ├── preprocessing/
│   │   ├── handle_missing.py
│   │   ├── scaling.py
│   │   └── drift_correction.py
│   │
│   ├── models/
│   │   ├── fuzzy_model.py
│   │   ├── neuro_fuzzy_model.py
│   │   ├── lstm_model.py
│   │   └── cnn_model.py
│   │
│   ├── utils/
│   │   ├── metrics.py           # RMSE, MAE, R²
│   │   ├── visualization.py     # Plot predictions, errors, membership functions
│   │   └── config.py            # Global settings (paths, hyperparameters)
│   │
│   └── main.py                  # Main execution script (can integrate all models for comparison)
│
├── results/
│   ├── fuzzy/                   # Plots, metrics, and models (.pkl or .h5)
│   ├── neuro_fuzzy/
│   ├── deep_learning/
│   └── comparison_summary.csv
│
├── models/                      # Saved trained model weights or checkpoints
│   ├── fuzzy_model.pkl
│   ├── neuro_fuzzy_model.h5
│   ├── lstm_model.h5
│   └── cnn_model.h5
│
├── reports/
│   ├── figures/                 # Graphs and plots for the report
│   ├── final_report.pdf
│   └── presentation_slides.pptx
│
├── requirements.txt             # Python dependencies (NumPy, pandas, scikit-fuzzy, TensorFlow, etc.)
├── README.md                    # Overview of the project and instructions
└── .gitignore                   # Ignore large data or model files
