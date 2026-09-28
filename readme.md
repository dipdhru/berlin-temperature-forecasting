# Berlin Temperature Forecasting

Comparative evaluation of Bidirectional LSTM, Transformer, and Prophet
models for seven-day daily mean temperature forecasting in Berlin, Germany.

## Research Objective

This project investigates the ability of deep learning and statistical
forecasting models to predict daily mean temperature for the next seven
days using historical meteorological data.

## Models

- Bidirectional LSTM
- Transformer
- Facebook Prophet

## Data

Historical daily weather data for Berlin are obtained from the Open-Meteo
historical weather API.

Location:

- Latitude: 52.52
- Longitude: 13.405

Meteorological variables include:

- Mean temperature
- Maximum temperature
- Minimum temperature
- Precipitation
- Maximum wind speed
- Radiation
- Evapotranspiration
- Relative humidity

Cyclical temporal features are also used.

## Forecasting

The neural models use historical sequences of 30 days and perform
seven-day recursive forecasting.

## Evaluation

Models are evaluated using:

- MAE
- RMSE
- R²
- sMAPE

MAPE is retained only as a supplementary metric because temperature
values can be zero or negative.

## Reproducibility

The notebooks contain the complete data preparation, model training,
hyperparameter tuning, forecasting, and evaluation pipeline.

## Author

Dipanshu Singh