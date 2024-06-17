import torch
import streamlit as st
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
import yfinance as yf
import requests
import plotly.graph_objs as go
from datetime import date
from yahooquery import search

# Load FinBERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("ProsusAI/finbert")
model = BertForSequenceClassification.from_pretrained("ProsusAI/finbert")

# News API key
NEWS_API_KEY = "505a7c8de757459b97e1983d933ae92a"  # Replace with your News API key

# Function to fetch live news
def fetch_live_news(stock):
    url = f"https://newsapi.org/v2/everything?q={stock}&apiKey={NEWS_API_KEY}&language=en"
    response = requests.get(url)
    if response.status_code == 200:
        articles = response.json().get('articles', [])
        headlines = [(article['title'], article['publishedAt']) for article in articles]
        return headlines
    else:
        st.error("Failed to fetch live news.")
        return []

# Function to perform inference on headlines for a specific stock
def perform_inference_for_stock(stock, headlines_list, batch_size=16):
    results = []
    for i in range(0, len(headlines_list), batch_size):
        batch_headlines = headlines_list[i:i+batch_size]
        headlines, timestamps = zip(*batch_headlines)
        inputs = tokenizer(list(headlines), padding=True, truncation=True, return_tensors='pt', max_length=512)
        outputs = model(**inputs)
        prediction = torch.nn.functional.softmax(outputs.logits, dim=-1)
        for headline, timestamp, pos, neg, neutr in zip(headlines, timestamps, prediction[:, 0].tolist(), prediction[:, 1].tolist(), prediction[:, 2].tolist()):
            results.append((headline, timestamp, stock, pos, neg, neutr))
    return results

# Function to get sentiment message
def get_sentiment_message(sentiment):
    if sentiment == "Positive":
        return "Yes, it is a good time to invest today!"
    elif sentiment == "Negative":
        return "No, it's risky to invest today."
    else:
        return "Hahaha, the market is unpredictable today."

# Function to create a gauge chart
def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        title = {'text': title},
        gauge = {
            'axis': {'range': [None, 100]},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 100], 'color': "gray"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90}}))
    return fig

# Function to get stock ticker from company name
def get_stock_ticker(company_name):
    search_result = search(company_name)
    if 'quotes' in search_result:
        quotes = search_result['quotes']
        if quotes:
            return quotes[0]['symbol']
    return None

# Streamlit app
def main():
    st.set_page_config(page_title="Financial Sentiment Analysis", layout="wide")
    
    # Add your logo
    st.image("logo.png", use_column_width=True)  # Make sure to replace "your_logo.png" with the path to your logo image file

    st.title("Financial Sentiment Analysis with FinBERT")

    # Search bar for selecting stock
    stock_input = st.sidebar.text_input("Enter Stock Ticker or Company Name:")
    search_button = st.sidebar.button("Search")

    if search_button and stock_input:
        # Get stock ticker from company name
        selected_stock_symbol = get_stock_ticker(stock_input)
        
        if not selected_stock_symbol:
            st.error(f"Could not find a ticker for {stock_input}. Please enter a valid stock ticker or company name.")
            return

        st.write(f"Selected Stock Ticker: {selected_stock_symbol}")

        # Fetch live news for selected stock
        with st.spinner('Fetching live news...'):
            live_headlines = fetch_live_news(selected_stock_symbol)

        if not live_headlines:
            st.write("No news found for the given stock.")
            return

        # Perform inference for selected stock
        with st.spinner('Performing sentiment analysis...'):
            inference_results = perform_inference_for_stock(selected_stock_symbol, live_headlines)

        # Display sentiment analysis results
        st.write("Financial Sentiment Analysis Table:")
        df = pd.DataFrame(inference_results, columns=["Headline", "Timestamp", "Stock", "Positive", "Negative", "Neutral"])
        st.write(df)

        # Get overall sentiment for the selected stock
        overall_sentiment = "Neutral"  # Default to neutral if no headlines
        if len(df) > 0:
            overall_sentiment = df[['Positive', 'Negative', 'Neutral']].sum().idxmax()

        # Plot time series of stock
        st.write(f"Time Series of {selected_stock_symbol}:")
        try:
            end_date = date.today().strftime('%Y-%m-%d')
            start_date = (date.today() - pd.DateOffset(years=1)).strftime('%Y-%m-%d')
            stock_data = yf.download(selected_stock_symbol, start=start_date, end=end_date)
            if stock_data.empty:
                st.error(f"No data found for {selected_stock_symbol}. Please check the stock ticker.")
            else:
                st.line_chart(stock_data['Close'])
                
                # Display stock volume using gauge chart
                st.write(f"Volume of {selected_stock_symbol}:")
                latest_volume = stock_data['Volume'].iloc[-1]
                fig = create_gauge_chart(latest_volume, f"{selected_stock_symbol} Volume")
                st.plotly_chart(fig)

        except yf.shared._exceptions.YFZMissingError:
            st.error(f"The stock ticker {selected_stock_symbol} might be delisted or there is no timezone information available.")
        except Exception as e:
            st.error(f"Error fetching stock data: {e}")

        # Display sentiment message
        sentiment_message = get_sentiment_message(overall_sentiment)
        st.write(sentiment_message)

if __name__ == "__main__":
    main()
