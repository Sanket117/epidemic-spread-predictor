# Epidemic Spread Predictor

A machine learning and epidemiological modeling system for predicting COVID-19 outbreak risk, forecasting case growth, and detecting hotspots across 201 countries.

Built for Track C — Epidemic Spread Prediction (Epidemiology + AI) Hackathon.

---

## Project Structure
```
epidemic-spread-predictor/
│
├── data/
│   ├── raw/                        # Downloaded datasets
│   │   ├── jhu_confirmed_global.csv
│   │   ├── jhu_deaths_global.csv
│   │   └── owid_covid_data.csv
│   └── processed/                  # Cleaned and merged datasets
│       ├── covid_merged.csv
│       ├── risk_scores.csv
│       └── latest_risk_snapshot.csv
│
├── notebooks/
│   ├── 01_preprocess.ipynb         # Data cleaning and merging
│   ├── 02_eda.ipynb                # Exploratory data analysis
│   ├── 03_forecasting.ipynb        # Prophet + SEIR modeling
│   └── 04_hotspot_risk.ipynb       # Risk scoring and hotspot detection
│
├── app.py                          # Streamlit dashboard
├── download_data.py                # Dataset downloader
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Datasets Used

| Dataset | Source | Usage |
|---|---|---|
| JHU COVID-19 Time Series | Johns Hopkins CSSE | Daily confirmed cases and deaths across 201 countries |
| Our World in Data COVID-19 | OWID | Vaccination rates, testing, reproduction rate, demographics |

---

## Models

### Prophet (Facebook)
Time-series forecasting model trained per country on 7-day rolling average new cases. Captures yearly and weekly seasonality with multiplicative mode. Evaluated using MAE and RMSE on a 30-day held-out test set.

| Country | MAE | RMSE |
|---|---|---|
| US | 34,720 | 34,875 |
| India | 211 | 224 |
| Brazil | 7,292 | 7,643 |
| United Kingdom | 3,226 | 3,349 |
| Germany | 11,080 | 12,065 |

### SEIR Compartmental Model
Epidemiological transmission model tracking four population compartments — Susceptible, Exposed, Infectious, Recovered. Parameters calibrated to COVID-19 estimates (incubation period 5.2 days, recovery period 14 days).

### Composite Risk Score
A weighted 0–100 risk score per country per week combining:
- Cases per million (40%) — current burden
- Case growth rate week-over-week (25%) — trajectory
- Reproduction rate Rt (20%) — transmission speed
- Positive rate (15%) — undetected spread

Risk tiers: Low (0–25), Moderate (25–50), High (50–75), Critical (75–100)

---

## Dashboard Pages

| Page | Description |
|---|---|
| 🌍 Global Overview | Global daily cases, total KPIs, top 10 countries by cases and deaths |
| 📈 Country Analysis | Per-country case trajectory, vaccination progress, reproduction rate |
| 🔥 Hotspot Detection | Adjustable hotspot detection with heatmap across time |
| 🗺️ Risk Map | Interactive choropleth world map with weekly risk scores |
| 📊 Model Forecast | Live Prophet forecast with configurable horizon and accuracy metrics |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/your-username/epidemic-spread-predictor.git
cd epidemic-spread-predictor
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download datasets
```bash
python download_data.py
```

### 5. Run preprocessing pipeline

Run all notebooks in order:
- `notebooks/01_preprocess.ipynb`
- `notebooks/02_eda.ipynb`
- `notebooks/03_forecasting.ipynb`
- `notebooks/04_hotspot_risk.ipynb`

### 6. Launch dashboard
```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## Requirements

- Python 3.10+
- pandas, numpy, matplotlib, seaborn
- plotly, streamlit
- scikit-learn, statsmodels
- prophet
- scipy
- jupyter

---

## Data Sources

- JHU COVID-19 Dataset: https://github.com/CSSEGISandData/COVID-19
- Our World in Data: https://github.com/owid/covid-19-data

---

## Acknowledgements

Data provided by the Center for Systems Science and Engineering (CSSE) at Johns Hopkins University and Our World in Data.