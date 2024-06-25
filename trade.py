import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import tkinter as tk
from tkinter import ttk

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def prepare_data():
    print("Preparing data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    data = yf.download("AAPL", start=start_date, end=end_date, interval="5m")
    
    data = data.reset_index()
    data['Datetime'] = data['Datetime'].dt.tz_localize(None)
    data['Difference'] = data['Close'] - data['Open']
    data['SimpleMovingAvrg6'] = data['Close'].rolling(window=6).mean()
    data['Gain'] = data['Difference'].clip(lower=0)
    data['Loss'] = -data['Difference'].clip(upper=0)
    data['gainAvrg'] = data['Gain'].rolling(window=6).mean()
    data['lossAvrg'] = data['Loss'].rolling(window=6).mean()
    data['RSI'] = 100 - (100 / (1 + data['gainAvrg'] / data['lossAvrg']))
    data['High-Low'] = data['High'] - data['Low']
    data['High-PrevClose'] = abs(data['High'] - data['Close'].shift(1))
    data['Low-PrevClose'] = abs(data['Low'] - data['Close'].shift(1))
    data['TR'] = data[['High-Low', 'High-PrevClose', 'Low-PrevClose']].max(axis=1)
    data['ATR'] = data['TR'].rolling(window=6).mean()
    data['Volume Rate'] = data['Volume'] / data['Volume'].rolling(window=6).mean()
    
    columns_to_keep = ['Open', 'High', 'Low', 'Close', 'Volume', 'Difference', 
                       'SimpleMovingAvrg6', 'gainAvrg', 'lossAvrg', 'RSI', 'ATR', 'Volume Rate']
    
    data = data[columns_to_keep].dropna()
    
    data['Buy/Sell'] = np.where(data['Close'] > data['SimpleMovingAvrg6'], 1, 0)
    
    data.to_excel('appleData.xlsx', index=False)
    print("Data prepared and saved to appleData.xlsx")
    return data

def train_model(data):
    print("Training model...")
    X = data.drop('Buy/Sell', axis=1)
    y = data['Buy/Sell']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression()
    model.fit(X_train, y_train)
    
    weights = model.coef_[0]
    bias = model.intercept_[0]
    
    model_data = {'Feature': ['Bias'] + list(X.columns),
                  'Weight': np.concatenate(([bias], weights))}
    
    pd.DataFrame(model_data).to_excel('model_weights.xlsx', index=False)
    print("Model trained and weights saved to model_weights.xlsx")
    return model

def create_gui(model):
    def calculate():
        try:
            values = [float(entry.get()) for entry in entries]
            prediction = model.predict([values])[0]
            probability = model.predict_proba([values])[0][1]
            
            if prediction == 1:
                result_var.set(f"Prediction: Buy (Probability: {probability:.4f})")
            else:
                result_var.set(f"Prediction: Sell (Probability: {probability:.4f})")
        except ValueError:
            result_var.set("Invalid input. Please enter numbers only.")
        except Exception as e:
            result_var.set(f"An error occurred: {str(e)}")

    root = tk.Tk()
    root.title("Stock Trading Predictor")

    labels = ['Open', 'High', 'Low', 'Close', 'Volume', 'Difference', 
              'Simple Moving Average', 'gainAvrg', 'lossAvrg', 'RSI', 'ATR', 'Volume Rate']

    entries = []
    for i, label_text in enumerate(labels):
        row = i // 3
        col = i % 3
        label = ttk.Label(root, text=f"{label_text}:")
        label.grid(row=row, column=col*2, padx=5, pady=5, sticky="e")
        entry = ttk.Entry(root, width=10)
        entry.grid(row=row, column=col*2+1, padx=5, pady=5)
        entries.append(entry)

    calculate_button = ttk.Button(root, text="Predict", command=calculate)
    calculate_button.grid(row=4, column=2, columnspan=2, pady=10)

    result_var = tk.StringVar()
    result_label = ttk.Label(root, textvariable=result_var)
    result_label.grid(row=5, column=0, columnspan=6, pady=5)

    root.mainloop()

def main():
    data = prepare_data()
    model = train_model(data)
    create_gui(model)

if __name__ == "__main__":
    main()