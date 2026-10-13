import numpy as np
from scipy import stats, signal

REQUIRED_FEATURES = [
    "duration", "max_amplitude", "min_amplitude", "amplitude_range", "mean_amplitude", 
    "std_amplitude", "skewness", "kurtosis_val", "rms_energy", "total_energy",
    "zero_crossing_rate", "num_peaks", "dominant_frequency", "spectral_energy",
    "low_freq_power", "mid_freq_power", "high_freq_power", "spectral_centroid",
    "spectral_entropy", "r_peak_position", "r_peak_amplitude", "q_amplitude",
    "q_position", "s_amplitude", "s_position", "qrs_width", "beat_area",
    "max_slope", "min_slope", "rr_interval", "prev_rr", "next_rr",
    "rr_ratio_prev", "rr_ratio_next", "heart_rate", "rr_diff_prev", "rr_diff_next"
]

def extract_compatible_features(beat, fs, r_peak_idx_in_beat, rr_stats):
    """
    Extract features compatible with the trained model (37 features)
    """
    feat = {}
    
    # Time Domain Basic
    feat['duration'] = len(beat) / fs
    feat['max_amplitude'] = np.max(beat)
    feat['min_amplitude'] = np.min(beat)
    feat['amplitude_range'] = feat['max_amplitude'] - feat['min_amplitude']
    feat['mean_amplitude'] = np.mean(beat)
    feat['std_amplitude'] = np.std(beat)
    feat['skewness'] = stats.skew(beat)
    feat['kurtosis_val'] = stats.kurtosis(beat)
    feat['rms_energy'] = np.sqrt(np.mean(beat**2))
    feat['total_energy'] = np.sum(beat**2)
    feat['zero_crossing_rate'] = np.sum(np.diff(np.signbit(beat))) / len(beat)
    
    peaks, _ = signal.find_peaks(beat)
    feat['num_peaks'] = len(peaks)
    
    # Frequency Domain
    try:
        freqs, psd = signal.welch(beat, fs, nperseg=min(256, len(beat)))
        max_idx = np.argmax(psd)
        feat['dominant_frequency'] = freqs[max_idx]
        feat['spectral_energy'] = np.sum(psd)
        
        # Power bands roughly 0-5, 5-15, 15+
        feat['low_freq_power'] = np.sum(psd[(freqs >= 0) & (freqs < 5)])
        feat['mid_freq_power'] = np.sum(psd[(freqs >= 5) & (freqs < 20)])
        feat['high_freq_power'] = np.sum(psd[(freqs >= 20)])
        
        centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)
        feat['spectral_centroid'] = centroid
        
        psd_norm = psd / (np.sum(psd) + 1e-10)
        feat['spectral_entropy'] = -np.sum(psd_norm * np.log2(psd_norm + 1e-10))
    except:
        feat.update({k: 0 for k in ['dominant_frequency', 'spectral_energy', 'low_freq_power', 
                                    'mid_freq_power', 'high_freq_power', 'spectral_centroid', 'spectral_entropy']})

    # Morphological
    # R peak is expected to be the max amplitude or near center
    # In segmented beat, R peak is usually centered
    r_idx = r_peak_idx_in_beat
    feat['r_peak_position'] = r_idx / fs
    feat['r_peak_amplitude'] = beat[r_idx]
    
    # Q and S detection (simple search around R)
    # Q: min before R
    q_search = beat[max(0, r_idx-15):r_idx]
    if len(q_search) > 0:
        q_idx = max(0, r_idx-15) + np.argmin(q_search)
        feat['q_amplitude'] = beat[q_idx]
        feat['q_position'] = (r_idx - q_idx) / fs
    else:
        feat['q_amplitude'] = 0
        feat['q_position'] = 0
        
    # S: min after R
    s_search = beat[r_idx:min(len(beat), r_idx+15)]
    if len(s_search) > 0:
        s_idx = r_idx + np.argmin(s_search)
        feat['s_amplitude'] = beat[s_idx]
        feat['s_position'] = (s_idx - r_idx) / fs
    else:
        feat['s_amplitude'] = 0
        feat['s_position'] = 0
        
    feat['qrs_width'] = feat['q_position'] + feat['s_position'] # Approx
    feat['beat_area'] = np.sum(np.abs(beat))
    
    diff = np.diff(beat)
    feat['max_slope'] = np.max(diff) if len(diff)>0 else 0
    feat['min_slope'] = np.min(diff) if len(diff)>0 else 0
    
    # RR Features
    # rr_stats dict: {rr_interval, prev_rr, next_rr}
    feat['rr_interval'] = rr_stats.get('rr_interval', 0)
    feat['prev_rr'] = rr_stats.get('prev_rr', 0)
    feat['next_rr'] = rr_stats.get('next_rr', 0)
    feat['rr_ratio_prev'] = feat['rr_interval'] / (feat['prev_rr'] + 1e-6)
    feat['rr_ratio_next'] = feat['rr_interval'] / (feat['next_rr'] + 1e-6)
    
    if feat['rr_interval'] > 0:
        feat['heart_rate'] = 60 / feat['rr_interval']
    else:
        feat['heart_rate'] = 0
        
    feat['rr_diff_prev'] = feat['rr_interval'] - feat['prev_rr']
    feat['rr_diff_next'] = feat['rr_interval'] - feat['next_rr']
    
    # Return array in correct order
    return [feat.get(name, 0) for name in REQUIRED_FEATURES]


