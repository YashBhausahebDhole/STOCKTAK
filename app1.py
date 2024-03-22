import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

# Function to perform inference on headlines
def perform_inference(headlines_list, stocks_list, selected_stock):
    # Initialize a list to store results
    results = []

    # Filter headlines for the selected stock
    relevant_headlines = [headline for headline, stock in zip(headlines_list, stocks_list) if stock == selected_stock]

    # Chunk relevant headlines for processing
    STRIDE = 100
    for lines in chunk_list(relevant_headlines, STRIDE):
        # Convert all elements in lines to strings and then concatenate
        lines = " ".join(map(str, lines))

        inputs = tokenizer(lines, padding=True, truncation=True, return_tensors='pt', max_length=512, return_attention_mask=True)
        outputs = model(**inputs)

        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)

        for headline, pos, neg, neutr in zip(lines, prediction[:, 0].tolist(), prediction[:, 1].tolist(), prediction[:, 2].tolist()):
            results.append((headline, selected_stock, pos, neg, neutr))

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
    stocks_list = headlines_df['stock'].unique().tolist()

    # Select stock
    selected_stock = st.selectbox("Select Stock:", stocks_list)

    # Perform inference
    with st.spinner('Performing inference...'):
        inference_results = perform_inference(headlines_df['headline'].tolist(), headlines_df['stock'].tolist(), selected_stock)

    # Display results
    st.write("Financial Sentiment Analysis Table:")
    df = pd.DataFrame(inference_results, columns=["Headline", "Stock", "Positive", "Negative", "Neutral"])
    st.write(df)

    # Plot sentiment distribution
    sentiment_counts = df[['Positive', 'Negative', 'Neutral']].sum()
    labels = ['Positive', 'Negative', 'Neutral']
    plt.bar(labels, sentiment_counts)
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.title('Sentiment Distribution')
    st.pyplot()

if __name__ == "__main__":
    main()
