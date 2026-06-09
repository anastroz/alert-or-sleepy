import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk
import pandas as pd

##############################

"""

Step 1

The first step involves defining a function to assist with preparing the JSON for export. 

Step 2

ECG processing using the modified function from NeuroKit 2 (MIT license). Modifications: skipped ecg_delineate and ecg_phase, additional Butterworth filter added.

URL of the original function: https://github.com/neuropsychology/NeuroKit/blob/master/neurokit2/ecg/ecg_process.py (version commit https://github.com/neuropsychology/NeuroKit/commit/ff419d983568ef492eb8d229af643c0ef0100b32)

"""


def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


def ecg_process_localver(ecg_signal, sampling_rate, method="neurokit", **kwargs):
    '''
    Modification of the original NeuroKit 2 ecg_process function. 
    '''

    # Sanitize and clean input
    ecg_signal = nk.signal_sanitize(ecg_signal)
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=sampling_rate, method=method, **kwargs)
    ecg_cleaned_filtered = nk.signal_filter(ecg_cleaned, sampling_rate=sampling_rate, lowcut=2,
                                            method="butterworth", order=2)
    # Detect R-peaks
    instant_peaks, info = nk.ecg_peaks(
        ecg_cleaned=ecg_cleaned_filtered,
        sampling_rate=sampling_rate,
        method=method,
        correct_artifacts=True,
    )

    # Calculate heart rate
    rate = nk.signal_rate(
        info, sampling_rate=sampling_rate, desired_length=len(ecg_cleaned_filtered)
    )

    # Assess signal quality
    quality = nk.ecg_quality(
        ecg_cleaned_filtered, rpeaks=info["ECG_R_Peaks"], sampling_rate=sampling_rate
    )

    # Merge signals in a DataFrame
    signals = pd.DataFrame(
        {
            "ECG_Raw": ecg_signal,
            "ECG_Clean": ecg_cleaned_filtered,
            "ECG_Rate": rate,
            "ECG_Quality": quality,
        }
    )

    # Add additional information to signals DataFrame
    signals = pd.concat(
        [signals, instant_peaks], axis=1
    )

    # return signals DataFrame and R-peak locations
    return signals, info

