# Traffic-Flow-Prediction-System - Assignment 2B
The Part B of Assignment 2 (2B) requires you to work with your group to implement ML algorithms to train ML models for traffic prediction and integrate the traffic predictor with Part A to develop a fully functioned TBRGS.

# Overview 
This Project provides lengthy Python Implementation alongside Jupyter NoteBook, to showcase basic, deep learning techniques, data parsing & machine langugage algorithms + training & testing with provided datasets
It includes a Travel Time Estimation based on each edge of the map from the isolated Boroondara area.

# Features
1. Long Short-Term Memory (LSTM)
2. Gated Recurrent Unit (GRU)
3. (The) Sparse Autoencoder (SAE)

# Installation
1. Clone this repository:
    Either Via Option A | Option B
    ----
    Option A - via terminal: (bash)
    ```bash
    git clone https://github.com/sithkeo/Traffic-Flow-Prediction-System-
    ```
    ----
    Option B - Click the Blue Clone Feature to open a query, when followed, produces a cloned Git Repo, onto your device.

    ![evaluate](/images/clone.png)

2. (Optional) Create a Virtual Environment and Activate it:
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Used for Linux/Mac Users
    .venv\Scripts\activate.bat # Used for Windows Users
    ```
3. Install the Required Dependencies (Provided inside the `requirements.txt`):
    ```bash
    pip install -r `requirements.txt`
    ```

# Usage
1. Train all models on a specific SCATS site:
    ```bash
    python main.py output/Scats_Data_October_2006_parsed.csv --site 970
    ```

2. Train models on all sites:
    ```bash
    python main.py output/Scats_Data_October_2006_parsed.csv --all-sites
    ```

3. Train only selected models:
    ```bash
    python main.py output/Scats_Data_October_2006_parsed.csv --site 970 --models gru_model lstm_model
    ```

4. Adjust training settings:
    ```bash
    python main.py output/Scats_Data_October_2006_parsed.csv --site 970 --epochs 100 --batch_size 16
    ```

# generate_predicted_site_volumes.py
- Predict average traffic volume for each SCATS site using a trained model.
    ```bash
    python generate_predicted_site_volumes.py
    ```
- Prompts for SCATS CSV input and model selection (GRU, LSTM, SAE)
Automatically saves output to:
    ```bash
    output/predicted/{model}_site_predictions.csv
    ```
Used as input for model-based route estimation

# routing.py
Generates a road network from SCATS sensor data, calculates edge weights using model-predicted traffic, and computes optimal routes.

# python routing.py
Uses `output/predicted/{model}_site_predictions.csv for travel time weights`
- Prompts for origin and destination SCATS IDs
- Computes top 5 shortest paths, selects the fastest based on estimated time
- Travel time is calculated using speed derived from predicted flow and includes a 30-second delay per SCATS-controlled intersection
- Saves output map to `scats_route_map.html`

# python gui.py
Opens a GUI Version of the application that presents SCAT ID's that can be entered, alongside the Model Type to be used
Which then generates routes and visually graphs it onto `index.html`

# Contact
- Maintainer (@sithkeo)
- Repo Link: [https://github.com/sithkeo/Traffic-Flow-Prediction-System-]
