"""CA3 ECG analyzer. Inference uses the original trusted Random Forest artifact."""
import io
import json
import os
import secrets
import threading
import time
import tempfile
from collections import Counter, OrderedDict, deque
from datetime import timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for
from scipy.signal import butter, filtfilt, detrend, iirnotch
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash, generate_password_hash

from ecg_features import REQUIRED_FEATURES, extract_compatible_features
from ecg_preprocess import adaptive_r_peak_detection, preprocess_ecg_fixed

BASE = Path(__file__).resolve().parent
PRODUCTION = os.getenv('APP_ENV') == 'production'
PASSWORD = os.getenv('APP_PASSWORD', '')
SECRET = os.getenv('SECRET_KEY', '')
if PRODUCTION and (len(SECRET) < 32 or len(PASSWORD) < 12):
    raise RuntimeError('Production requires SECRET_KEY (32+ chars) and APP_PASSWORD (12+ chars).')
PASSWORD_HASH = generate_password_hash(PASSWORD) if PASSWORD else None
app = Flask(__name__)
app.config.update(
    SECRET_KEY=SECRET or secrets.token_hex(32),
    MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=PRODUCTION,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=timedelta(minutes=30),
    MAX_FORM_PARTS=10,
)

# Load during import so Gunicorn and the development server behave identically.
MODEL = joblib.load(BASE / 'models' / 'ecg_rf_pipeline.joblib')
if MODEL.n_features_in_ != len(REQUIRED_FEATURES) or set(MODEL.classes_) != {'F', 'N', 'Q', 'S', 'V'}:
    raise RuntimeError('The model artifact is incompatible with this feature schema.')
CLASS_LABELS = {'N': 'Normal', 'V': 'PVC', 'S': 'SVE', 'F': 'Fusion', 'Q': 'Unclassifiable'}
RATE_LOCK = threading.Lock()
RATE_BUCKETS = OrderedDict()
INFERENCE_SLOT = threading.BoundedSemaphore(1)


def csrf_token():
    if 'csrf' not in session:
        session['csrf'] = secrets.token_urlsafe(32)
    return session['csrf']


app.jinja_env.globals['csrf_token'] = csrf_token


def rate_limited(scope, limit):
    # Deliberately per-process; one production worker. Never trust arbitrary X-Forwarded-For.
    key = (scope, request.remote_addr or 'unknown')
    now = time.monotonic()
    with RATE_LOCK:
        while RATE_BUCKETS and next(iter(RATE_BUCKETS.values()))[-1] < now - 60:
            RATE_BUCKETS.popitem(last=False)
        bucket = RATE_BUCKETS.setdefault(key, deque())
        while bucket and bucket[0] < now - 60:
            bucket.popleft()
        if len(bucket) >= limit:
            return True
        bucket.append(now)
        RATE_BUCKETS.move_to_end(key)
        if len(RATE_BUCKETS) > 2048:
            RATE_BUCKETS.popitem(last=False)
    return False


@app.before_request
def protect_requests():
    if request.endpoint in {'health', 'static'}:
        return
    if request.method == 'POST':
        supplied = request.headers.get('X-CSRF-Token', '') or request.form.get('csrf_token', '')
        if not supplied or not secrets.compare_digest(str(supplied), session.get('csrf', '')):
            return jsonify(success=False, error='Session expired. Refresh the page and retry.'), 403
        limit = 5 if request.endpoint == 'login' else 15
        if rate_limited(request.endpoint, limit):
            return jsonify(success=False, error='Too many requests. Please retry in one minute.'), 429
    if PASSWORD_HASH and request.endpoint != 'login' and not session.get('authenticated'):
        if request.path.startswith('/api/'):
            return jsonify(success=False, error='Sign in to continue.'), 401
        return redirect(url_for('login'))


@app.after_request
def security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'"
    response.headers['Cache-Control'] = 'no-store' if request.endpoint != 'static' else 'public, max-age=3600'
    if PRODUCTION:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000'
    return response


@app.errorhandler(Exception)
def handle_error(error):
    if isinstance(error, HTTPException):
        messages = {413: 'Upload must be smaller than 2 MB.', 400: 'Invalid request.'}
        return jsonify(success=False, error=messages.get(error.code, error.name)), error.code
    app.logger.exception('Request failed')
    return jsonify(success=False, error='The request could not be completed.'), 500


@app.route('/login', methods=['GET', 'POST'])
def login():
    if not PASSWORD_HASH:
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        if check_password_hash(PASSWORD_HASH, request.form.get('password', '')):
            session.clear()
            session['authenticated'] = True
            session.permanent = True
            csrf_token()
            return redirect(url_for('index'))
        error = 'Incorrect access password.'
    return render_template('login.html', error=error), (401 if error else 200)


@app.post('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


@app.get('/')
def index():
    return render_template('index.html', protected=bool(PASSWORD_HASH))


@app.get('/health')
def health():
    return jsonify(status='ok', model_loaded=True)


@app.get('/api/model')
def model_info():
    return jsonify(algorithm='Random Forest', trees=200, features=37, classes=CLASS_LABELS,
                   version='ca3-1.0', validation='Historical training report is included in the repository; current deployment accuracy has not been established.')


@app.get('/sample.csv')
def sample():
    return send_file(BASE / 'samples' / 'mitbih_100_30s.csv', as_attachment=True, download_name='mitbih_100_30s.csv')


def load_signal(upload, sampling_rate):
    extension = Path(upload.filename or '').suffix.lower()
    if extension not in {'.csv', '.txt', '.dat'}:
        raise ValueError('Upload a CSV, TXT, or DAT ECG signal.')
    if not 100 <= sampling_rate <= 1000:
        raise ValueError('Sampling rate must be between 100 and 1000 Hz.')
    raw = upload.read(2 * 1024 * 1024 + 1)
    if not raw or len(raw) > 2 * 1024 * 1024:
        raise ValueError('The file is empty or larger than 2 MB.')
    if extension == '.dat':
        header = request.files.get('header')
        if header and header.filename:
            if Path(header.filename).suffix.lower() != '.hea':
                raise ValueError('The companion WFDB header must be a .hea file.')
            header_raw = header.read(64 * 1024 + 1)
            if not header_raw or len(header_raw) > 64 * 1024:
                raise ValueError('The .hea header is empty or too large.')
            try:
                header_text = header_raw.decode('ascii')
                lines = header_text.splitlines()
                first = lines[0].split()
                if len(first) < 3:
                    raise ValueError('The .hea file is not a valid WFDB header.')
                source_name = Path(first[0]).name
                if Path(source_name).stem.casefold() != Path(upload.filename).stem.casefold():
                    raise ValueError('Choose the matching .dat and .hea files from the same ECG record.')
                first[0] = 'ecg_record'
                lines[0] = ' '.join(first)
                referenced_data = 0
                for i in range(1, len(lines)):
                    line = lines[i].strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split()
                    if parts[0].lower().endswith(('.dat', '.mat')):
                        if Path(parts[0]).name.casefold() != Path(upload.filename).name.casefold():
                            raise ValueError('This WFDB header references other data files; upload a single-file record.')
                        parts[0] = 'ecg_record.dat'
                        lines[i] = ' '.join(parts)
                        referenced_data += 1
                if referenced_data == 0:
                    raise ValueError('The .hea header does not reference the selected .dat file.')
                import wfdb
                with tempfile.TemporaryDirectory(prefix='ecg-wfdb-') as temp_dir:
                    base = Path(temp_dir) / 'ecg_record'
                    base.with_suffix('.dat').write_bytes(raw)
                    base.with_suffix('.hea').write_text('\n'.join(lines) + '\n', encoding='ascii')
                    record = wfdb.rdrecord(str(base), channels=[0])
                    values = np.asarray(record.p_signal[:, 0], dtype=float)
                    sampling_rate = int(record.fs)
            except ValueError:
                raise
            except Exception as error:
                raise ValueError('Could not read this WFDB .dat/.hea pair. Check that the files belong to the same record.') from error
        else:
            encoding = request.form.get('dat_encoding', 'wfdb212')
            if encoding == 'wfdb212':
                if len(raw) % 3:
                    raise ValueError('MIT-BIH WFDB 212 data must contain complete 3-byte sample groups. Upload its matching .hea file or select raw 16-bit encoding.')
                # Headerless MIT-BIH records conventionally contain two multiplexed
                # channels. Ask WFDB to decode channel zero instead of treating
                # interleaved channel samples as one faster signal.
                try:
                    import wfdb
                    with tempfile.TemporaryDirectory(prefix='ecg-wfdb-') as temp_dir:
                        base = Path(temp_dir) / 'ecg_record'
                        base.with_suffix('.dat').write_bytes(raw)
                        frames = len(raw) // 3
                        base.with_suffix('.hea').write_text(
                            f'ecg_record 2 {sampling_rate} {frames}\n'
                            'ecg_record.dat 212 200/mV 12 0 0 0 0 ECG1\n'
                            'ecg_record.dat 212 200/mV 12 0 0 0 0 ECG2\n',
                            encoding='ascii')
                        record = wfdb.rdrecord(str(base), channels=[0])
                        values = np.asarray(record.p_signal[:, 0], dtype=float)
                except Exception as error:
                    raise ValueError('Could not decode this headerless MIT-BIH 212 recording. Upload its matching .hea file for exact signal details.') from error
            elif encoding == 'int16':
                if len(raw) % 2:
                    raise ValueError('A raw signed 16-bit .dat file must contain an even number of bytes.')
                values = np.frombuffer(raw, dtype='<i2').astype(float)
            else:
                raise ValueError('Select MIT-BIH WFDB 212 or raw signed 16-bit DAT encoding.')
    if extension in {'.csv', '.txt'}:
        if b'\x00' in raw:
            raise ValueError('The file is not a text signal.')
        try:
            text = raw.decode('utf-8-sig')
            if extension == '.csv':
                # Header is optional. The LAST column is the ECG amplitude.
                first_line = text.splitlines()[0].split(',')
                try:
                    [float(value) for value in first_line]
                    has_header = False
                except ValueError:
                    has_header = True
                frame = pd.read_csv(io.StringIO(text), header=0 if has_header else None)
                values = pd.to_numeric(frame.iloc[:, -1], errors='raise').to_numpy(dtype=float)
            else:
                values = np.loadtxt(io.StringIO(text), ndmin=1)
                if values.ndim == 2:
                    values = values[:, -1]
        except (ValueError, UnicodeError, pd.errors.ParserError, IndexError) as error:
            raise ValueError('File must contain numeric ECG amplitudes in its last column.') from error
    if not 100 <= sampling_rate <= 1000:
        raise ValueError('Sampling rate in the WFDB header must be between 100 and 1000 Hz.')
    max_duration = 1860 if extension == '.dat' else 120
    if values.ndim != 1 or len(values) < 3 * sampling_rate or len(values) > max_duration * sampling_rate:
        duration_hint = '3 seconds and 31 minutes' if extension == '.dat' else '3 and 120 seconds'
        raise ValueError(f'Provide a signal lasting between {duration_hint} at the selected sampling rate.')
    if not np.isfinite(values).all() or np.std(values) < 1e-8:
        raise ValueError('Signal must vary and contain only finite numeric values.')
    return values, sampling_rate


def analyze_signal(values, fs):
    filtered = preprocess_ecg_fixed(values, fs)
    peaks = np.unique(adaptive_r_peak_detection(filtered, fs)).astype(int)
    if len(peaks) < 3:
        raise ValueError('Too few heartbeats were detected. Upload a longer or clearer ECG recording.')
    rr = np.diff(peaks) / fs
    rr_padded = np.insert(rr, 0, np.mean(rr))
    before, after = int(0.2 * fs), int(0.4 * fs)
    vectors, positions = [], []
    for i in range(1, len(peaks) - 1):
        peak = peaks[i]
        if peak - before < 0 or peak + after >= len(filtered):
            continue
        vector = extract_compatible_features(
            filtered[peak-before:peak+after], fs, before,
            {'rr_interval': rr_padded[i], 'prev_rr': rr_padded[i-1], 'next_rr': rr_padded[i+1]})
        if not np.isfinite(vector).all():
            raise ValueError('Feature extraction produced invalid values. Try a clearer ECG signal.')
        vectors.append(vector)
        positions.append(int(peak))
    if not vectors:
        raise ValueError('No complete beat windows were available for classification.')
    feature_frame = np.asarray(vectors, dtype=float)
    probabilities = MODEL.predict_proba(feature_frame)
    predictions = MODEL.classes_[np.argmax(probabilities, axis=1)]
    beats = [{'index': i+1, 'type': str(label), 'label': CLASS_LABELS[str(label)],
              'confidence': float(np.max(probabilities[i])), 'time_seconds': round(pos/fs, 3)}
             for i, (label, pos) in enumerate(zip(predictions, positions))]
    counts = Counter(str(label) for label in predictions)
    rr_ms = rr * 1000
    delta = np.diff(rr_ms)
    # Preserve mV amplitude for visualization; model input remains normalized.
    b, a = butter(4, [0.5/(fs/2), 45/(fs/2)], btype='band')
    visual = filtfilt(b, a, values)
    bn, an = iirnotch(50/(fs/2), 30)
    visual = detrend(filtfilt(bn, an, visual))
    stride = max(1, int(np.ceil(len(values) / 4000)))
    return {
        'success': True, 'algorithm': 'Random Forest', 'sampling_rate': fs,
        'duration_seconds': round(len(values)/fs, 2), 'detected_beats': len(peaks),
        'classified_beats': len(beats), 'heart_rate_bpm': round(60/float(np.mean(rr)), 1),
        'rr_variability': {'sdnn_ms': round(float(np.std(rr_ms, ddof=1)), 2),
                           'rmssd_ms': round(float(np.sqrt(np.mean(delta**2))), 2),
                           'pnn50_percent': round(float(np.mean(np.abs(delta)>50)*100), 2)},
        'distribution': {label: counts.get(label, 0) for label in CLASS_LABELS},
        'beats': beats,
        'waveform': {'seconds': (np.arange(0, len(values), stride)/fs).round(4).tolist(),
                     'amplitude': visual[::stride].round(6).tolist(),
                     'beat_seconds': [round(pos/fs, 4) for pos in positions],
                     'beat_amplitude': [round(float(visual[pos]), 6) for pos in positions]},
        'notes': ['Edge beats without complete context are excluded from classification.',
                  'Confidence is the forest vote probability; it is not a validated clinical certainty.',
                  'RR variability includes detected beats of all types and is not validated NN-only HRV.',
                  'Educational prototype. Predictions have not been validated for diagnosis.'],
    }


@app.post('/api/analyze')
def analyze():
    if not INFERENCE_SLOT.acquire(blocking=False):
        return jsonify(success=False, error='The analyzer is busy. Please retry shortly.'), 503
    try:
        upload = request.files.get('file')
        if not upload or not upload.filename:
            return jsonify(success=False, error='Choose an ECG file.'), 400
        try:
            fs = int(request.form.get('sampling_rate', '360'))
            values, fs = load_signal(upload, fs)
            result = analyze_signal(values, fs)
        except ValueError as error:
            return jsonify(success=False, error=str(error)), 422
        return jsonify(result)
    finally:
        INFERENCE_SLOT.release()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=int(os.getenv('PORT', '5001')), debug=False)
