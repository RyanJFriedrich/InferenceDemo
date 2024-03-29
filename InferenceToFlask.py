## Stock Market Prediction Inferencing Module
# Built by Ryan Friedrich
# Adapted from from FEDFormer model for stock market prediction.
# FEDFormer research done by Alibaba, based on Autoformer model research at Tsinghua University.
# Copyright 2024
# License - ??
################################
import torch # Obviously used
import numpy as np # Numpy needed for Numpy
import pandas as pd # Pandas needed for input / output
import joblib # Joblib needed for loading files
import pickle # Pickle ended up being the best method to load models with architecture
import csv # CSV for CSV handling
import time # for benchmarking
import os # It's OS. 
import requests # Requests for API requests
import json # JSON for JSON handling

from flask import Flask, jsonify
from flask_cors import CORS

from pathlib import Path
from models import FEDformer # FEDformer - Modified to be purpose built for task
from utils.tools import StandardScaler # Standard scaler used for scaling data
from utils.timefeatures import time_features # Time features Not Used?

app = Flask(__name__)
CORS(app)

class Config:
    def __init__(self, params):
        for key, value in params.items():
            if key in ['seq_len', 'label_len', 'pred_len', 'factor', 'd_model', 
                       'n_heads', 'e_layers', 'd_layers', 'd_ff', 'num_workers', 
                       'itr', 'batch_size', 'enc_in', 'dec_in', 'c_out', 'modes', 'device']:
                value = int(value)
            elif key in ['dropout']:
                value = float(value)
            # Add any other keys that should be converted to int or float here
            setattr(self, key, value)
            # Prints Keys / Values of Config for debugging
            # print(f"Key: {key}, Type: {type(value)}")

def Get_Binance_Data():
    # API endpoint
    url = "https://api.binance.com/api/v3/uiKlines"

    # Parameters
    params = {
        "symbol": "XRPUSDT",
        "interval": "5m",
        "limit": 50
    }

    # Sending GET request to the API
    response = requests.get(url, params=params)

    # Check if the request was successful
    if response.status_code == 200:
        data = response.json()

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Drop the first column
        df = df.drop(df.columns[0], axis=1)

        # Rearrange the columns
        df = df.iloc[:, [0, 3, 1, 2, 4]]

        # Slicing off trailing zeros by specifying precision
        df[1] = pd.to_numeric(df[1], errors='coerce').round(4)
        df[2] = pd.to_numeric(df[2], errors='coerce').round(4)
        df[3] = pd.to_numeric(df[3], errors='coerce').round(4)
        df[4] = pd.to_numeric(df[4], errors='coerce').round(4)
        df[5] = pd.to_numeric(df[5], errors='coerce').round(1)


    else:
        # Handle non-200 status codes
        error_message = f"Error: response status is {response.status_code}"
        try:
            # Try to parse the error message from the response
            error_data = response.json()
            error_message += f"\nResponse body\n{json.dumps(error_data, indent=2)}"
        except json.JSONDecodeError:
            # In case the response is not a valid JSON
            error_message += f"\nResponse body could not be parsed as JSON. Raw response: {response.text}"

        # Throw the error message
        raise Exception(error_message)

    return df


def load_config(config_path):
    # Parse the CSV config file and return a dictionary of parameters
    params = {}
    with open(config_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            key, value = row
            # Type casting config parameters based on those used in training file. Reference -- run.py
            if key in ['seq_len', 'label_len', 'pred_len', 'factor', 'd_model', 
                       'n_heads', 'e_layers', 'd_layers', 'd_ff', 'num_workers', 'itr', 
                       'batch_size', 'device']:
                params[key] = int(value)
            elif key in ['dropout', 'attn_prob']:
                params[key] = float(value)
            elif key in ['time', 'use_amp', 'output_attention', 'mix', 'padding', 'distil']:
                params[key] = value.lower() in ['true', '1', 'yes']
            else:
                params[key] = value
    
    return Config(params)
    #return params

def load_model(model_path):
    # Initialize the model architecture -- All stored inside pick file
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    #Send loaded model back
    return model


def run_inference(model, input_dataframe, params, scaler_path):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    scaler = joblib.load(scaler_path)

    # Scale data to model
    scaled_data = scaler.transform(input_dataframe.values)

    ## Shapes printed for debugging purposes
    print("Shape of Scaled data:", scaled_data.shape)
    print("Value of Label Length:", params.label_len)
    print("Value of Pred Length:", params.pred_len)
    print("Value of Sequence Length:", params.seq_len)
    
    # Prepare model_input
    model_input = scaled_data[-params.seq_len:]
    model_input = torch.tensor(model_input, dtype=torch.float32).unsqueeze(0).to(device)

    # Preparing the decoder input
    label_data = scaled_data[-params.label_len:]
    
    ## Shapes printed for debugging purposes
    print("shape of label_data after slice:", label_data.shape)

    # Create a tensor of pred_len filled with zeros
    decoder_input = torch.zeros((1, params.pred_len + params.label_len, input_dataframe.shape[1]), dtype=torch.float32).to(device)

    # Fill the first part (label_len) with historical data
    decoder_input[0, :params.label_len, :] = torch.tensor(label_data, dtype=torch.float32).to(device)

    # List to store predictions
    all_predictions = []
    
     ## Shapes printed for debugging purposes
    print("Shape of model_input:", model_input.shape)
    print("Shape of decoder_input:", decoder_input.shape)

    # Prepare the model for evaluation
    start_time = time.time()
    model.eval()
    with torch.no_grad():
        
        # Output comes from the model input and decoder. The decoder is preloaded with historical
        # data in the first chunk of the second dimension. 
        # Dimensions are [Batch, Data, Features]
        # Model is feature designed to output 5 features with only one feature tuned to loss per model.
        # Features are Open, Close, High, Low, Volume in that order. 

        outputs = model(model_input, decoder_input)

        outputs = outputs.detach().cpu().numpy()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Inference took {elapsed_time:.2f} seconds.")

    all_predictions = outputs

    # Rescale the predictions

    all_predictions = scaler.inverse_transform(outputs.reshape(-1, input_dataframe.shape[1]))

    # Return the predictions
    return np.array(all_predictions)

def main():
    return main

@app.route('/api/GetData', methods=['GET'])
def GetData():
    API_time = time.time()    

    ## Model Parameters within the checkpoint directory
    # Config Args path
    config_path = "args.csv"
    # Model path
    model_path = "model.pkl"
    # Scaler path
    scaler_path = "scaler.pkl"

    # Get Data From Binance
    input_dataframe = Get_Binance_Data()

    # Output data path
    output_path = "InferenceOutput.csv"

    current = "close"
    
    # Define the base directory for checkpoints
    base_checkpoint_dir = "checkpoints"
    # List all subdirectories in the base checkpoint directory
    model_dirs = next(os.walk(base_checkpoint_dir))[1]
    model_dirs.sort(key=int)  # Sort directories numerically
    predictions = []

    for i in range(len(model_dirs)):
        # Grabs the model directory from the list of directories
        model_dir = model_dirs[i]

        current_config_path = os.path.join(base_checkpoint_dir, model_dir, config_path)
        current_model_path = os.path.join(base_checkpoint_dir, model_dir, model_path)
        current_scaler_path = os.path.join(base_checkpoint_dir, model_dir, scaler_path)

        # Load configuration
        params = load_config(current_config_path)

        # Load model
        model = load_model(current_model_path)

        # Run inference
        start_time = time.time()
        prediction = run_inference(model, input_dataframe, params, current_scaler_path)
        # Prediction comes back as a Numpy array
        end_time = time.time()
        print(f"inferencing model {model_dir} took {end_time - start_time:.2f} seconds.")
        # Inference end

        # Append predictions of ONLY Close values [:,1] here... As per model training
        predictions.append(prediction[:,1])

    ## Round the dataframe
    predictions_df = pd.DataFrame(predictions).round(5)
    ## Print to CSV
    predictions_df.to_csv(output_path, index=False)
    
    # Json everything
    historical_json = input_dataframe.to_json(orient='records', date_format='iso')
    predictions_json = predictions_df.to_json(orient='records', date_format='iso')
    #Combine in to separate Json objects
    combined_json = {
    "historical": json.loads(historical_json),
    "predictions": json.loads(predictions_json)
    }
    #Pretty Json and output to file
    json_output = json.dumps(combined_json, indent=4)
    
    with open('output.json', 'w') as file:
        file.write(json_output)

    # Flask API eventual return output
    API_end_time = time.time()
    print(f"Entire API took {API_end_time - API_time:.2f} seconds.")

    return json_output



if __name__ == "__main__":
    app.run(debug=True)