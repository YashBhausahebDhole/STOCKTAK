import torch
import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yfinance as yf
import matplotlib.pyplot as plt

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Function to perform inference on headlines for a specific stock
def perform_inference_for_stock(stock, headlines_list):
    # Initialize a list to store results
    results = []

    # Convert all elements in lines to strings and then concatenate
    lines = " ".join(map(str, headlines_list))

    inputs = tokenizer(lines, padding=True, truncation=True, return_tensors='pt', max_length=512, return_attention_mask=True)
    outputs = model(**inputs)

    prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)

    for headline, pos, neg, neutr in zip(headlines_list, prediction[:, 0].tolist(), prediction[:, 1].tolist(), prediction[:, 2].tolist()):
        results.append((headline, stock, pos, neg, neutr))

    return results

# Function to get sentiment message
def get_sentiment_message(sentiment):
    if sentiment == "Positive":
        return "Yes, it is a good time to invest today!"
    elif sentiment == "Negative":
        return "No, it's risky to invest today."
    else:
        return "Hahaha, the market is unpredictable today."

# Function to calculate predicted line based on overall sentiment
def calculate_predicted_line(stock_data, overall_sentiment):
    if overall_sentiment == "Positive":
        return stock_data['Close'].mean() * 1.05
    elif overall_sentiment == "Negative":
        return stock_data['Close'].mean() * 0.95
    else:
        return stock_data['Close'].mean()

# Streamlit app
def main():
    st.title("Financial Sentiment Analysis with FinBERT")

    # Load headlines data
    headlines_df = pd.read_csv('3000_stock_headlines.csv')

    # Sidebar for selecting stock
    selected_stock = st.sidebar.selectbox("Select Stock:", headlines_df['stock'].unique())

    # Filter headlines for selected stock
    selected_headlines = headlines_df[headlines_df['stock'] == selected_stock]['headline'].tolist()

    # Perform inference for selected stock
    with st.spinner('Performing inference...'):
        inference_results = perform_inference_for_stock(selected_stock, selected_headlines)

    # Display sentiment analysis results
    st.write("Financial Sentiment Analysis Table:")
    df = pd.DataFrame(inference_results, columns=["Headline", "Stock", "Positive", "Negative", "Neutral"])
    st.write(df)

    # Get overall sentiment for the selected stock
    overall_sentiment = "Neutral"  # Default to neutral if no headlines
    if len(df) > 0:
        overall_sentiment = df[['Positive', 'Negative', 'Neutral']].sum().idxmax()

    # Plot time series of stock
    st.write(f"Time Series of {selected_stock}:")
    stock_data = yf.download(selected_stock, start="2023-01-01", end="2024-01-01")

    # Calculate predicted line based on overall sentiment
    predicted_line = calculate_predicted_line(stock_data, overall_sentiment)

    # Plot time series with predicted line
    st.line_chart(stock_data['Close'])
    
    # Plot the predicted line
    st.write(f"Predicted Line: {predicted_line}")
    plt.axhline(y=predicted_line, color='r', linestyle='--')
    st.pyplot()

    # Display sentiment message
    sentiment_message = get_sentiment_message(overall_sentiment)
    st.write(sentiment_message)

if __name__ == "__main__":
    main()
