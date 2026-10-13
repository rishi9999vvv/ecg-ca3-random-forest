import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def preprocess_ecg_fixed(ecg_signal, fs=360, use_wfdb_gain=False, wfdb_record=None):
    """
    Fixed preprocessing pipeline for ECG signals using the proven approach:
    1. Bandpass filter (0.5-45 Hz, 4th order) - removes baseline wander and high-frequency noise
    2. Notch filter (50 Hz, Q=30) - removes powerline interference
    3. Detrend - removes linear trend
    4. Z-score normalization (signal - mean) / std
    
    This approach produces clean, properly scaled ECG signals without distortion.
    
    Parameters:
    -----------
    ecg_signal : array-like
        Raw ECG signal (1D array)
    fs : int
        Sampling frequency (Hz)
    use_wfdb_gain : bool
        If True and wfdb_record provided, use physical units from record
    wfdb_record : wfdb.Record
        WFDB record object (optional, for gain correction)
    
    Returns:
    --------
    filtered_signal : array
        Preprocessed ECG signal
    """
    from scipy.signal import butter, filtfilt, iirnotch, detrend
    
    # Remove NaN and infinite values
    ecg_signal = np.array(ecg_signal, dtype=np.float64)
    
    # Handle NaN values
    if np.isnan(ecg_signal).sum() > 0:
        ecg_signal = np.nan_to_num(ecg_signal, nan=np.nanmean(ecg_signal))
    ecg_signal = np.nan_to_num(ecg_signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    # If signal is 2D, take first channel (MLII)
    if ecg_signal.ndim > 1:
        ecg_signal = ecg_signal[:, 0]
    
    # Apply WFDB gain correction if available
    if use_wfdb_gain and wfdb_record is not None:
        try:
            # Get gain from record
            if hasattr(wfdb_record, 'adc_gain'):
                gain = wfdb_record.adc_gain[0] if isinstance(wfdb_record.adc_gain, (list, np.ndarray)) else wfdb_record.adc_gain
            elif hasattr(wfdb_record, 'adc'):
                gain = wfdb_record.adc[0] if isinstance(wfdb_record.adc, (list, np.ndarray)) else wfdb_record.adc
            else:
                gain = 1.0
            
            if gain > 0:
                ecg_signal = ecg_signal / gain
        except Exception as e:
            print(f"Warning: Could not apply WFDB gain correction: {e}")
    
    # Step 1: Bandpass filter (0.5-45 Hz, 4th order) - removes baseline wander and high-frequency noise
    # This is the proven approach that produces clean ECG signals
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 45.0 / nyq
    
    if low < 1.0 and high < 1.0:
        b, a = butter(4, [low, high], btype='band')
        filtered_signal = filtfilt(b, a, ecg_signal)  # filtfilt for forward-backward filtering (zero phase shift)
    else:
        print(f"Warning: Filter cutoff too high. Using original signal.")
        filtered_signal = ecg_signal.copy()
    
    # Step 2: Notch filter to remove powerline interference (50 Hz, Q=30)
    # This removes 50 Hz powerline interference (European standard)
    try:
        b_notch, a_notch = iirnotch(50.0 / (fs / 2), Q=30)
        notched_signal = filtfilt(b_notch, a_notch, filtered_signal)
    except Exception as e:
        print(f"Warning: Notch filter failed: {e}. Using bandpass filtered signal.")
        notched_signal = filtered_signal
    
    # Step 3: Detrend - removes linear trend
    detrended_signal = detrend(notched_signal)
    
    # Step 4: Z-score normalization (proper normalization)
    # (signal - mean) / std
    # Note: For visualization, you might want to skip normalization
    # to preserve the original amplitude scale
    mean_val = np.mean(detrended_signal)
    std_val = np.std(detrended_signal)
    
    if std_val > 1e-6:  # Avoid division by zero
        filtered = (detrended_signal - mean_val) / std_val
    else:
        print(f"Warning: Signal has zero variance. Skipping normalization.")
        filtered = detrended_signal - mean_val  # At least center the signal
    
    return filtered

def adaptive_r_peak_detection(signal_data, fs):
    """
    Advanced R-peak detection with adaptive thresholding
    """
    # Bandpass filter for QRS complex (5-15 Hz)
    b, a = butter(2, [5/(fs/2), 15/(fs/2)], btype='band')
    filtered = filtfilt(b, a, signal_data)
    
    # Differentiate
    diff = np.diff(filtered)
    
    # Square
    squared = diff ** 2
    
    # Moving window integration
    window_size = int(0.150 * fs)  # 150ms window
    mwi = np.convolve(squared, np.ones(window_size)/window_size, mode='same')
    
    # Adaptive thresholding
    threshold = np.mean(mwi) + 0.5 * np.std(mwi)
    
    # Find peaks with adaptive threshold
    min_distance = int(0.200 * fs)  # Minimum 200ms between beats
    peaks, properties = find_peaks(mwi, height=threshold, distance=min_distance)
    
    # Refine to actual R-peaks
    refined_peaks = []
    search_window = int(0.040 * fs)  # 40ms window
    
    for peak in peaks:
        start = max(0, peak - search_window)
        end = min(len(signal_data), peak + search_window)
        
        # Find maximum in the window (actual R-peak)
        window_signal = signal_data[start:end]
        if len(window_signal) > 0:
            local_max = np.argmax(np.abs(window_signal))
            refined_peaks.append(start + local_max)
    
    return np.array(refined_peaks)


