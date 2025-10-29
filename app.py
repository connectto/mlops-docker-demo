# app.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib
import os
from datetime import datetime

# 1. 더미 시계열 데이터 생성
def generate_data():
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    values = np.random.randn(100).cumsum() + 100
    return pd.DataFrame({'date': dates, 'value': values}).set_index('date')

# 2. 모델 훈련 및 저장
def train_and_save_model():
    df = generate_data()
    df['lag1'] = df['value'].shift(1)
    df.dropna(inplace=True)
    
    X = df[['lag1']]
    y = df['value']
    
    model = LinearRegression()
    model.fit(X, y)
    
    # volume 폴더 생성
    os.makedirs('volume', exist_ok=True)
    
    # 모델 저장
    model_path = 'volume/timeseries_model.pkl'
    joblib.dump(model, model_path)
    print(f"[{datetime.now()}] Model saved to {model_path}")

# 3. 모델 로드 및 예측 (테스트)
def load_and_predict():
    model_path = 'volume/timeseries_model.pkl'
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        pred = model.predict([[120]])
        print(f"Prediction for lag1=120: {pred[0]:.2f}")
    else:
        print("Model file not found!")

if __name__ == "__main__":
    train_and_save_model()
    load_and_predict()