# Cloud ECG Analyzer with Random Forest

CA3 mini-project · ICC02 · Group 39

A Flask web application that classifies ECG beats using the original 200-tree Random Forest pipeline. The browser shows a filtered waveform, per-beat labels and vote probabilities, beat distribution, heart rate, and RR interval variability. JSON and CSV downloads and browser printing are available.

## Team

Siddarth Singh Kushwah, Rishi Dwivedi, Ronit Kamble, and Shubankar Bhan.

## Run locally

Use Python 3.12 and an isolated environment. The scikit-learn version must remain 1.3.2 for the original model artifact.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5001. Local development omits password protection unless APP_PASSWORD is set. The app does not load .env automatically; set environment variables in the shell or cloud dashboard.

## Input and real sample data

- CSV with or without a header, or numeric TXT. The last column is the ECG amplitude.
- Sampling frequency is entered explicitly, 100–1000 Hz; use 360 Hz for the included MIT-BIH samples.
- 3–120 seconds, maximum 2 MB. Files with non-finite, non-numeric, or constant signals are rejected.
- WFDB .dat/.hea files must be converted to CSV first. No file is saved by the analysis endpoint.
- `samples/` contains the first 30 seconds from MIT-BIH records 100, 119, and 200 and their reference annotation excerpts. Source: [PhysioNet MIT-BIH Arrhythmia Database v1.0.0](https://physionet.org/content/mitdb/1.0.0/). Recordings are provided for reproducible classroom demonstrations.

## Pipeline

1. Validate and parse the signal in memory.
2. Apply 0.5–45 Hz bandpass filtering, 50 Hz notch filtering, detrending, and z-score normalization.
3. Detect R peaks with the existing adaptive detector.
4. Extract 0.2 seconds before and 0.4 seconds after each eligible interior peak.
5. Extract the 37 features in the saved training report's order, including amplitude, morphology, frequency, and RR features.
6. Apply the saved StandardScaler + RandomForestClassifier pipeline.
7. Decode model classes directly: N, V, S, F, Q.

The compatible extractor comes from the source project's `evaluate_metrics_full.py`. Training provenance is incomplete: matching the feature names does not prove that every feature formula matches the historical training script. Current-pipeline evaluations are recorded separately from the old training report.

## Security and cloud deployment

See [SECURITY.md](SECURITY.md) for implemented controls, checks, and limits.

Deploy this repository as a Render Python web service. A reproducible `render.yaml` is included. Build: `pip install -r requirements.txt`. Start: `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120 --access-logfile -`. Health: `/health`.

Required production variables: APP_ENV=production, SECRET_KEY (random, at least 32 characters), APP_PASSWORD (at least 12 characters). Set OMP_NUM_THREADS=1 and OPENBLAS_NUM_THREADS=1. Render terminates HTTPS. Production cookies are Secure, HttpOnly, and SameSite=Lax. Keep one worker: the semaphore and rate limiter are process-local.

The free service may sleep when idle; wake it before the evaluation. No persistent database or disk is required. Do not upload patient-identifying records to this classroom prototype.

## Verification

```powershell
python -m unittest test_app -v
```

All 7 tests passed locally on 13 October 2026. Tests cover actual saved-model inference, invalid inputs, upload size, CSRF, rate limits, login/logout, and security headers. Do not use recursive test discovery on preserved legacy source files in the working copy.

## Results and limitations

`reports/training_report.json` is the inherited training report, not an independently reproduced benchmark. Its 98.80% accuracy must not be presented as the accuracy of this deployment. This implementation does not retrain or compare new models. Short demonstration excerpts are not held-out clinical validation; they may have been included in training. Confidence is the forest vote probability. RR variability includes ectopic beats and is not validated NN-only HRV. The app is an educational prototype, not a diagnostic device.

The current pipeline sanity check matched 107 classified beats against reference annotations from records 100, 119, and 200: 79 correct (73.83% accuracy), weighted precision 82.24%, weighted recall 73.83%, weighted F1 69.85%. Ventricular recall was only 9.09%, so this model misses many ventricular beats in these excerpts. Full evidence is in `reports/current_pipeline_check.json`. These measurements do not establish performance on unseen patients.

The original local project is unchanged. Copied legacy folders are preserved locally but excluded from Git; this deployable application uses only `app.py`, `ecg_features.py`, `ecg_preprocess.py`, `templates/index.html`, `templates/login.html`, and `static/ca3.*`.
