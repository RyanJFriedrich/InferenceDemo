
# Financial Model Inference API: Enhanced Fedformer Adaptation

## Overview
This repository showcases an advanced adaptation of the Fedformer model, initially designed for research purposes, into a robust tool for real-time financial trend prediction. The original model, limited in its flexibility and performance, underwent significant modifications to enhance its forecasting capabilities, particularly for short-term financial market movements.

This is a simple demonstration file that inferences a proprietary trained model.

This was private work done for a customer by myself with this specific code being entirely written by myself and approved for release by the customer.

## Key Enhancements

### Model Flexibility
- **Hyperparameterization**: Extensively refactored to support a broader range of hyperparameters, overcoming the original model's limitations such as the ability to specify more than 8 heads.

### Performance Optimization
- **Efficiency and Accuracy**: Modifications have significantly improved training and inference performance, ensuring efficient real-time predictions with high accuracy.

## Application Highlights

### "Hurricane Forecast" Model
- The demonstrated API feeds a "hurricane forecast" model, similar to that seen in weather prediction. This is particularly helpful in financial markets when you have many trained models with similar MSE, MAE, and MAPE statistics. They will often predict things slightly different but it gives us a possible path the financial hurricane will take. In demo I built very small ChartJS and TradingView charts to demonstrate a simple GUI for the model inference / API output.

## Getting in Touch
For a full demonstration of the code or further explanations, interested parties are encouraged to reach out via email at Ryan.J.Friedrich@gmail.com .
