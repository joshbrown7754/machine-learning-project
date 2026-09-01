# Machine Learning Stock Prediction

> Exploring whether machine learning can identify useful patterns in historical stock-market data to predict next-day percentage returns.

## Overview

This project investigates the use of machine learning for short-term stock return prediction.

The model uses historical market data to predict the next-day percentage change of a stock. The project focuses on the complete machine-learning pipeline, including data collection, data restructuring, time-series feature engineering, feature scaling, model implementation and evaluation.

Rather than evaluating the model in isolation, predictions are compared against a naive baseline that predicts a 0% return for every trading day. This provides a simple benchmark for determining whether the model is learning useful predictive information.

The project is also being used to investigate neural networks and understand the mathematics behind forward propagation and backpropagation before relying on PyTorch's automatic differentiation.

## Project Goals

- Collect historical stock-market data through an API.
- Build a reusable data-processing pipeline.
- Perform time-series feature engineering.
- Predict next-day percentage price changes.
- Implement and evaluate a linear regression model.
- Investigate neural-network approaches using PyTorch.
- Compare model performance against a zero-return baseline.
- Investigate how feature scaling affects model convergence.
- Develop a trading strategy based on model predictions.
- Eventually back-test the strategy against the S&P 500.

## Data

The model currently uses the following market features:

- Open price
- High price
- Low price
- Closing price
- Trading volume
- 5-day moving average
- Daily percentage price change

The prediction target is the **next-day percentage change** rather than the raw stock price.

Using percentage change allows the model to focus on relative price movement rather than absolute differences between stocks with different price levels.

## Data Processing

The raw API data is initially stored using feature/ticker combinations and Unix timestamps.

A preprocessing function restructures the data into a chronological hierarchy:

```text
Ticker
└── Date
    ├── Open
    ├── High
    ├── Low
    ├── Close
    └── Volume
