# AI-Powered Li-Ion Battery RUL Prediction System

## Project Overview

This project predicts the Remaining Useful Life (RUL) of lithium-ion batteries using machine learning and provides a natural language interface powered by an LLM.

## Dataset
Lithium-Ion Battery Degradation Dataset
**Source:**
https://www.kaggle.com/datasets/programmer3/lithium-ion-battery-degradation-dataset?select=Battery_dataset.csv

This dataset is modeled after the NASA Ames Prognostics Center of Excellence lithium-ion battery degradation dataset. It simulates the charge-discharge behavior and aging process of lithium-ion batteries across multiple cycles, capturing realistic trends in battery health over time.

The dataset features three virtual battery cells B0005 (B5), B0006 (B6) and B0007 (B7) and includes average values per cycle for key parameters such as:

Charging/Discharging Current

Charging/Discharging Voltage

Charging/Discharging Temperature

Battery Capacity (BCt)

State of Health (SOH)

Remaining Useful Life (RUL)


## Project Goals

- Train ML models for battery RUL prediction
- Track experiments using MLflow
- Build an LLM-powered interface
- Deploy a production-style ML application
