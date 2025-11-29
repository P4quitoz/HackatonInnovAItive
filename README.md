Machine Learning Energy Disaggregation – Fridge Prediction Hackathon

This repository contains our full pipeline for the Machine Learning Energy Challenge hackathon.
The task is to reconstruct the fridge power consumption from aggregated household smart-meter data based on the Chain2 transmission protocol.

We focus on clean structure, reproducibility, and a common preprocessing layer so that multiple teammates can experiment independently without breaking each other's pipelines.

📁 Project Structure
hackaton/
├─ data/
│  ├─ raw/
│  │  ├─ train.csv
│  │  └─ test.csv
│  └─ processed/
│     ├─ train_1min.csv
│     ├─ test_1min.csv
│     ├─ train_features.parquet
│     └─ test_features.parquet
│
├─ src/
│  ├─ common/                 # Shared pipeline used by ALL models
│  │  ├─ config.py            # global paths & constants
│  │  ├─ resampling.py        # Chain2 → 1-minute preprocessing
│  │  ├─ features.py          # feature engineering for ML models
│  │  └─ evaluation.py        # metrics, plots, CV helpers
│  │
│  ├─ datasets/
│  │  ├─ make_train.py        # builds processed training dataset
│  │  ├─ make_test.py         # builds processed test dataset
│  │  └─ (optional) cli.py    # run full pipeline from command line
│  │
│  ├─ experiments/            # each teammate/model has its own folder
│  │  ├─ tudor_lgbm/
│  │  │  ├─ train.py
│  │  │  ├─ predict.py
│  │  │  └─ config.py
│  │  ├─ alice_cnn/
│  │  ├─ bob_baseline/
│  │  └─ ...
│  │
│  └─ utils/
│     └─ logging.py
│
├─ models/                    # Saved artifacts per experiment
│  ├─ tudor_lgbm/
│  │  └─ model.pkl
│  ├─ alice_cnn/
│  │  └─ model.pt
│  └─ bob_baseline/
│     └─ model.pkl
│
├─ notebooks/
│  ├─ 01_eda.ipynb            # exploratory data analysis
│  └─ 02_signal_plots.ipynb   # visualization helpers
│
├─ submission/
│  ├─ tudor_lgbm.csv
│  ├─ alice_cnn.csv
│  └─ bob_baseline.csv
│
├─ requirements.txt
└─ README.md

🚀 What This Project Does
✓ Resamples irregular Chain2 smart-meter data

The raw data has variable sampling intervals (15-minute mandatory samples + additional samples every 300W threshold crossing).
We normalize everything to a regular 1-minute grid using a zero-order hold strategy.

✓ Adds consistent ML features

All models share the same standardized feature set:

Lag features (1, 2, 5, 10, 30, 60 mins)

Rolling means / stds

Power gradients

Time-of-day features (hour_sin, hour_cos, day-of-week)

Cleaned fridge target

✓ Supports multiple independent ML models

Every teammate has their own folder under src/experiments/ and can:

train their own model

tune hyperparameters

generate predictions

save results separately

No one overwrites anyone else’s work.

🛠️ Setup
Clone the repo
git clone <repo-url>
cd hackaton

Install dependencies
pip install -r requirements.txt

📦 Preprocessing Pipeline
1. Build training dataset
python -m src.datasets.make_train


This:

loads data/raw/train.csv

resamples it to 1-minute resolution (per home)

builds standardized features

saves results to data/processed/

2. Build test dataset
python -m src.datasets.make_test

🧠 Training Your Model

Each person's model lives in src/experiments/<your_model>/.

Example:

python -m src.experiments.tudor_lgbm.train


This:

loads the processed training features

trains the model

performs validation

saves the trained model to models/tudor_lgbm/model.pkl

📈 Making Predictions

Each model folder contains a predict.py script:

python -m src.experiments.tudor_lgbm.predict


This will:

load processed test features

apply the model

generate a submission file under submission/

👥 Adding a New Model (Team Workflow)

To add your own model:

Create a folder under:

src/experiments/<your_name_or_model>/


Add:

train.py

predict.py

config.py (optional)

Your code will automatically benefit from:

shared preprocessing

consistent feature engineering

clean data structure

This keeps the project clean, scalable, and easy for multiple contributors.

📬 Final Submission

Your final submission should be placed here:

submission/<your_model_name>.csv


Format must follow the hackathon’s expected output.

🤝 Contributing

If you want to add improvements to the shared preprocessing pipeline:

open a PR

or discuss with the team
since it affects all experiments.
