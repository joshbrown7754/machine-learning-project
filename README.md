# Machine Learning Stock Prediction

## Overview

This project explores whether machine learning can predict the next-day percentage change of stocks using historical market data.

The current model uses linear regression and is being developed as the first stage of a larger algorithmic trading project. The eventual goal is to use the model's predictions to develop a trading strategy and back-test its performance against the S&P 500.

The project is also being used to investigate how feature engineering, data preprocessing and model optimisation affect predictive performance on financial time-series data.

## Objectives

- Collect historical stock-market data using an API.
- Process and restructure raw financial data.
- Perform time-series feature engineering.
- Train a machine-learning model to predict next-day percentage changes.
- Compare model performance against a naive baseline.
- Investigate different feature-scaling techniques.
- Develop an algorithmic trading strategy based on model predictions.
- Back-test the strategy against the S&P 500.

## Features

The model currently uses:

- Open price
- High price
- Low price
- Closing price
- Trading volume
- 5-day moving average
- Daily percentage price change

These features are processed and scaled before being provided to the model.

## Data Processing

The raw API data is initially stored using ticker symbols, feature names and Unix timestamps.

A preprocessing pipeline restructures the data into a chronological format:

    Ticker
    └── Date
        ├── Open
        ├── High
        ├── Low
        ├── Close
        └── Volume

Unix timestamps are converted into readable dates and observations are sorted chronologically.

## Feature Engineering

### Daily Percentage Change

Daily percentage change is used as the prediction target rather than attempting to predict the raw stock price.

This allows the model to focus on relative price movement.

### 5-Day Moving Average

A 5-day moving average is calculated to provide the model with information about recent price trends.

### Trading Volume

Trading volume is included as an indicator of market activity.

### OHLC Data

Open, high, low and closing prices provide information about the daily price movement of each stock.

## Machine Learning Model

The current implementation uses linear regression to predict the next-day percentage change.

The project is also being extended to investigate neural-network approaches using PyTorch.

As part of this process, the forward pass of a neural network was implemented manually to understand the underlying calculations before using PyTorch to automate gradient calculations and backpropagation.

## Baseline

The model is evaluated against a naive zero-return baseline.

The baseline predicts:

    Next-day percentage change = 0%

for every trading day.

This provides a simple benchmark for determining whether the machine-learning model provides predictive value beyond simply predicting no change.

## Model Optimisation

The project investigates how feature scaling affects gradient-based optimisation.

Initial experiments used normalisation to scale features approximately between -1 and 1.

Further experiments are being conducted using standardisation:

    z = (x - mean) / standard deviation

The purpose of this experiment is to investigate whether the numerical scale of the features and target affects gradient magnitude and model convergence.

## Current Status

### Completed

- Historical stock data collection
- Data restructuring and timestamp conversion
- Time-series feature engineering
- OHLC and volume features
- 5-day moving average
- Percentage-change target
- Linear regression implementation
- Zero-return baseline
- Initial neural-network implementation
- Manual forward-pass calculations
- Investigation of feature scaling

### In Progress

- Standardisation experiments
- Neural-network optimisation
- Model evaluation
- Trading strategy development

### Planned

- Back-test trading strategy
- Compare strategy performance against the S&P 500
- Evaluate risk and return
- Experiment with additional features
- Investigate alternative machine-learning models

## Technologies

- Python
- PyTorch
- NumPy
- Pandas
- JSON
- Git
- GitHub

## Project Structure

    machine-learning-project/
    │
    ├── algorithimic_trading/
    │   └── ...
    │
    ├── README.md
    └── ...

## Limitations

Predicting short-term stock-price movements is inherently difficult because financial markets contain substantial noise and are influenced by many factors that are not represented by historical price data alone.

The current model only uses a limited set of market-based features and does not incorporate factors such as:

- News sentiment
- Company fundamentals
- Macroeconomic indicators
- Market-wide economic data

The project should therefore be considered an exploration of machine learning and financial time-series prediction rather than a reliable system for predicting future stock prices.

## Future Improvements

- Test additional time-series features.
- Investigate different neural-network architectures.
- Tune learning rates and optimisation algorithms.
- Add additional historical data.
- Investigate LSTM and other sequence-based models.
- Incorporate market and economic indicators.
- Develop a complete trading strategy.
- Back-test the strategy against the S&P 500.
- Analyse risk-adjusted returns.

## Author

Joshua Brown

GitHub:
https://github.com/joshbrown7754
