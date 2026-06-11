# EV Range Anxiety Predictive Pipeline

An end-to-end cloud-integrated machine learning pipeline designed to predict range anxiety risk for Electric Vehicle (EV) drivers based on trip telemetry and ambient environmental conditions. 

## Architecture Overview
This project separates scalable cloud data storage, heavy compute optimization, and local model inference:
1. **Data Ingestion:** Raw historical charging session telemetry is moved securely via AWS CLI to Amazon S3.
2. **Cloud Training Environment:** Amazon SageMaker ingests raw data streams via an optimized IAM Execution Role configuration.
3. **Feature Engineering:** Programmatic target masking maps low-battery states (State of Charge < 15%) into a unified target class matrix (`Risk`).
4. **Model Architecture:** Implements a Random Forest Classifier using scikit-learn.
5. **Persistence & Deployment:** Serialized model binaries (`.pkl`) are managed via S3 artifacts, allowing lightweight distributed client environments to load the model over an active SDK connection (`boto3`) for inference tasks.

## Key Engineering Focus: Severe Class Imbalance
The raw operational dataset presented a significant distribution bias, where over 90% of logged profiles represented non-critical, safe charging events. An unoptimized machine learning model would simply default to predicting the safe class to trick baseline accuracy benchmarks while failing to flag critical anomalies.

To address this:
* Implemented the `class_weight='balanced'` optimization flag.
* This automatically shifts cost function penalties inversely proportional to minor-class frequencies, heavily penalizing the decision trees for misclassifying high-risk events.
* Result: Scaled high-risk recall performance up to **~73%**, drastically reducing critical False Negatives.

## Project Directory Structure
```text
ev-charging-project/
├── data/                  # Local temporary storage for historical telemetry
├── models/                # Ignored path (binaries persistent in S3)
├── scripts/
│   ├── preprocessing.py   # Distributed client script utilizing S3 model artifacts
└── README.md