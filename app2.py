import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Function to perform inference on headlines
def perform_inference(headlines_list, stocks_list):
    # Initialize a list to store results
    results = []

    # Chunk headlines and stocks for processing
    STRIDE = 100
    for lines, stocks in zip(chunk_list(headlines_list, STRIDE), chunk_list(stocks_list, STRIDE)):
        # Convert all elements in lines to strings and then concatenate
        lines = " ".join(map(str, lines))

        inputs = tokenizer(lines, padding=True, truncation=True, return_tensors='pt', max_length=512, return_attention_mask=True)
        outputs = model(**inputs)

        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for headline, stock, pos, neg, neutr in zip(lines, stocks, prediction[:, 0].tolist(), prediction[:, 1].tolist(), prediction[:, 2].tolist()):
            results.append((headline, stock, pos, neg, neutr))

    return results

# Function to chunk list
def chunk_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

# Function to fetch real-time stock data
def fetch_stock_data(stock_symbol):
    stock_data = yf.download(stock_symbol, start="2024-01-01", end="2024-03-21")
    return stock_data

# Function to calculate sentiment accuracy
def calculate_accuracy(predictions, actual_sentiments):
    correct_predictions = sum(np.argmax(predictions, axis=1) == actual_sentiments)
    total_predictions = len(actual_sentiments)
    accuracy = correct_predictions / total_predictions
    return accuracy

# Streamlit app
def main():
    st.title("Financial Sentiment Analysis with FinBERT")

    # Load headlines data
    headlines_df = pd.read_csv('3000_stock_headlines.csv')
    headlines_list = headlines_df['headline'].tolist()
    stocks_list = headlines_df['stock'].tolist()

    # Perform inference
    with st.spinner('Performing inference...'):
        inference_results = perform_inference(headlines_list, stocks_list)

    # Display results
    st.write("Financial Sentiment Analysis Table:")
    df = pd.DataFrame(inference_results, columns=["Headline", "Stock", "Positive", "Negative", "Neutral"])
    st.write(df)

    # Fetch real-time stock data
    selected_stock = st.selectbox("Select Stock:", stocks_list)
    stock_data = fetch_stock_data(selected_stock)

    # Calculate sentiment accuracy
    actual_sentiments = np.argmax(df[['Positive', 'Negative', 'Neutral']].values, axis=1)
    predictions = df[['Positive', 'Negative', 'Neutral']].values
    accuracy = calculate_accuracy(predictions, actual_sentiments)
    st.write(f"Accuracy of sentiment analysis for {selected_stock}: {accuracy:.2f}")

if __name__ == "__main__":
    main()
