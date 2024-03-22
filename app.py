import streamlit as st
import pandas as pd
import numpy as np
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

if __name__ == "__main__":
    main()
