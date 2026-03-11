# Bangalore House Price Prediction – End-to-End ML Project

An automated, production-grade house price prediction system for Bangalore real estate using real-world data (~13.3k records). From raw messy data to a drift-resilient, FastAPI-deployed model with full MLOps pipeline.

## 🎯 Project Highlights

- **Final Model Performance**  
  Random Forest Regressor (tuned) → **Test R² = 0.82** | RMSE ≈ 0.7 lakhs  
  → Improved from Linear Regression baseline (R² 0.66) and tuned Decision Tree (R² 0.74)

- **Data Cleaning & Preprocessing**  
  - Original: ~13,300 rows, 9 columns  
  - Removed: 529 duplicates + ~5–8% outliers (sqft >20k, price/sqft outside [2k–25k], illogical bath & location anomalies)  
  - Handled missing values: society (~41%), bath (4%), others <2%  
  - Custom unit conversion (sqft, yards, acres, guntha) + range averaging (e.g., 1200-1400 → 1300)

- **Feature Engineering**  
  - log(total_sqft) — skewness reduced from ~12.5 → ~0.8  
  - Ratios: bath/BHK, sqft/BHK  
  - Flags: extra bathrooms, has_society (binary)  
  - Target encoding on location (top-20 threshold + 'other') — leakage-safe  
  - Standard scaling + one-hot encoding (area_type, etc.)

- **Data Drift Mitigation**  
  Reduced from **37.5% (3/9 columns)** → **7.69% (1/13 columns)**  
  → Dropped low-importance 'availability', created stable proxies for society & location

- **MLOps Pipeline (7 modular components)**  
  - MongoDB ingestion (50k row batch limit)  
  - Schema validation  
  - Data transformation & feature engineering  
  - Model training (Random Forest best params)  
  - Evaluation vs. production model  
  - Conditional S3 model pusher (versioned: `model_YYYYMMDD.pkl`)  
  - DVC for data versioning + timestamped artifacts

- **Production & Observability**  
  - FastAPI backend + simple HTML/CSS frontend  
  - Avg prediction latency: ~9.8 seconds (logged)  
  - Structured logging with timestamps & error tracing  


## 📈 Results at a Glance

| Stage       | Model              | Test R² | RMSE (lakhs) | Notes                  |
|-------------|--------------------|---------|--------------|------------------------|
| Baseline    | Linear Regression  | 0.66    | ~1.2         | Underfitting           |
| Tuned Tree  | Decision Tree      | 0.74    | ~0.9         | Overfit → tuned        |
| Final       | Random Forest      | **0.82**| **~0.7**     | Best generalization    |

## 🛠️ Tech Stack

- **Languages & Frameworks**: Python, FastAPI  
- **Data Handling**: Pandas, NumPy  
- **ML**: Scikit-learn (RandomForestRegressor, GridSearchCV, StandardScaler, OneHotEncoder)  
- **Database**: MongoDB  
- **Storage & Versioning**: AWS S3, DVC  
- **Logging & Monitoring**: Custom structured logging  
- **Frontend**: HTML + CSS

## 🛠️ Deployed

-- **Docker** ECR , EC2 instance


