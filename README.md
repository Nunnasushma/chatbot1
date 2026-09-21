# 🚗 Car Price Predictor

A Machine Learning web application built using **Python**, **Scikit-Learn**, and **Streamlit** to estimate the selling price of a car based on historical dataset features.

---

## 🚀 Live Demo

Try the deployed Car Price Predictor application here:

[👉 Open Car Price Predictor](https://nunnasushma-chatbot1-app-vbr3w4.streamlit.app/)

---

## 1. Project Title
**Car Price Predictor**

## 2. Project Objective
To build an end-to-end Machine Learning pipeline that predicts the expected car selling price given its model, manufacturing year, kilometers driven, fuel type, seller type, transmission, and previous ownership history.
## 3. Problem Statement
When buying or selling a car, determining a fair market price is challenging due to multiple influencing factors such as car brand/model, age, mileage, fuel type, transmission, and ownership count. This project answers the question:
> *"If I have this particular car, with these characteristics, what could its expected price be?"*

## 4. How the Project Works
1. **Data Ingestion**: Loads historical car sales data from `dataset/car_data.csv`.
2. **Preprocessing Pipeline**: Categorical columns (`name`, `fuel`, `seller_type`, `transmission`, `owner`) are encoded using `OneHotEncoder(handle_unknown='ignore')`, while numerical features (`year`, `km_driven`) pass through untouched via `ColumnTransformer`.
3. **Model Training & Selection**: Trains `LinearRegression` and `RandomForestRegressor` models using an 80/20 train/test split. Compares MAE, RMSE, and R² scores to select the top performing model.
4. **Pipeline Persistence**: Saves the complete preprocessing + trained model pipeline to `model/car_price_model.pkl`.
5. **Interactive UI**: A Streamlit web application loads the saved pipeline, accepts user inputs, and displays instant car price predictions.

---

## 5. Input Features
| Feature | Type | Description |
| :--- | :--- | :--- |
| **Car Model / Car Type (`name`)** | Categorical | Brand & specific model name of the car |
| **Manufacturing Year (`year`)** | Numerical | Year the vehicle was manufactured |
| **Kilometers Driven (`km_driven`)** | Numerical | Total distance driven in kilometers |
| **Fuel Type (`fuel`)** | Categorical | Fuel variant (`Petrol`, `Diesel`, `CNG`, `LPG`) |
| **Seller Type (`seller_type`)** | Categorical | Channel of sale (`Individual`, `Dealer`, `Trustmark Dealer`) |
| **Transmission (`transmission`)** | Categorical | Gearbox type (`Manual`, `Automatic`) |
| **Previous Owners (`owner`)** | Categorical | Ownership count (`First Owner`, `Second Owner`, etc.) |

## 6. Target Variable
- **`selling_price`**: Selling price of the car in Indian Rupees (INR). Displayed in **₹ Lakhs** in the application UI.

---

## 7. Dataset
- **File Location**: `dataset/car_data.csv`
- **Total Records**: 291 car listings
- **Source**: Kaggle Used Car Dataset

---

## 8. Data Preprocessing
- **Duplicate Removal**: Automatically checks for and removes duplicate records.
- **Missing Value Handling**: Cleans rows with missing feature values.
- **Pipeline Integration**: Implements a Scikit-Learn `ColumnTransformer` with `OneHotEncoder` configured with `handle_unknown='ignore'`. This prevents data leakage and ensures seamless prediction on new/unseen car models.

---

## 9. Machine Learning Algorithms
1. **Linear Regression**: Baseline regression model.
2. **Random Forest Regressor**: Ensemble learning technique with `n_estimators=200` and `random_state=42`.

---

## 10. Model Evaluation
Evaluated on an 80/20 test split (`random_state=42`):

| Algorithm | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | R² Score |
| :--- | :--- | :--- | :--- |
| **Linear Regression** | ₹268,083.72 | ₹399,374.59 | 0.4763 |
| **Random Forest Regressor (Selected)** | **₹175,494.14** | **₹289,910.27** | **0.7240** |

---

## 11. Project Structure
```text
used-car-resale-price-predictor/
│
├── dataset/
│   └── car_data.csv          # Kaggle used car dataset
│
├── model/
│   └── car_price_model.pkl   # Saved Scikit-learn model + preprocessing pipeline
│
├── train_model.py            # Data loading, cleaning, pipeline training, evaluation & saving
├── app.py                    # Streamlit web application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 12. Installation

```bash
# Clone or navigate to project directory
cd "d:/car price prediction"

# Install required dependencies
pip install -r requirements.txt
```

---

## 13. Training Instructions

To re-train the model, execute:

```bash
python train_model.py
```

Outputs model performance metrics and updates `model/car_price_model.pkl`.

---

## 14. Streamlit Instructions

To launch the web interface:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 15. Example Prediction

```text
Inputs:
- Car Model: Honda City
- Present Price: ₹10.0 Lakhs
- Manufacturing Year: 2020
- KMs Driven: 30,000
- Fuel Type: Petrol
- Seller Type: Dealer
- Transmission: Manual
- Previous Owners: 1

Output:
- Estimated Resale Price: ₹7.20 Lakhs
```

---

## 16. Future Improvements
- Expand dataset size with more recent vehicle models.
- Incorporate vehicle brand/manufacturer extraction as a secondary feature.
- Include hyperparameter tuning via `GridSearchCV`.
- Deploy to Streamlit Community Cloud.
](https://nunnasushma-chatbot1-app-vbr3w4.streamlit.app/)
