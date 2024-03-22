import torch
import streamlit as st
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import yfinance as yf

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

# Streamlit app
def main():

    # Add your logo
    st.image("logo.png", use_column_width=True)  # Make sure to replace "your_logo.png" with the path to your logo image file

    st.title("Financial Sentiment Analysis with FinBERT")

    # Load headlines data and other parts of your Streamlit app
    # Implementation of these parts is the same as in your code
    pass
   

    # Load headlines data
    headlines_df = pd.read_csv('3000_stock_headlines.csv')

    # Sidebar for selecting stock
    selected_stock = st.sidebar.selectbox("Select Stock:", headlines_df['stock'].unique())

    # Filter headlines for selected stock
    selected_headlines = headlines_df[headlines_df['stock'] == selected_stock]['headline'].tolist()

    # Perform inference for selected stock
    with st.spinner('Wait getting results...'):
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
    st.line_chart(stock_data['Close'])

    # Display sentiment message
    sentiment_message = get_sentiment_message(overall_sentiment)
    st.write(sentiment_message)

if __name__ == "__main__":
    main()
