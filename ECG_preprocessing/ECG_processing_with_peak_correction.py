import numpy as np
import matplotlib.pyplot as plt
import obci_readmanager.signal_processing.read_manager as obci_rm
import os
import mne
import neurokit2 as nk
import pandas as pd
from ecg_peak_correction import correct_ecg_peaks_gui, plot_ecg_peak_correction
from local_ecg_helper import ecg_process_localver


""" STEP 1

- Reading the dataset and selecting proper channels (1.1)
- If needed to invert the signal, see Step 1.2.
- Overview of the detected peaks for the whole signal (only for preview) (1.3)
- break_blocks tags onsets are identified (1.4)
"""

''' STEP 1.1  '''

os.chdir('.../DATA_TO_PROCESS/P090_NIGHT/corrected_tags')
participant_file = "P090_NIGHT_2025-11-15-04-25-00"
filename = participant_file + "_corrected_tag_positions"
rawdata = obci_rm.ReadManager(filename+".obci.xml", filename+".obci.raw", filename+".obci.tag")
rawdata = rawdata.get_mne_raw()
signals = rawdata.pick(['ECG', 'Photo'])
sampling_freq = signals.info['sfreq']
ecg_mne = signals.copy().pick(['ECG'])  # only a subset of channels is picked
photo_mne = signals.copy().pick(['Photo'])
ecg_mne.copy().plot(duration=10)
del rawdata
del signals

''' STEP 1.2 '''
# If inversion is needed:
#
# ecg_mne.apply_function(lambda x: -x, picks=['ECG'])
# ecg_mne.plot(duration=10)

''' KROK 1.3 '''
###
# Extracting data for the preview drawing
whole_ecg = ecg_mne.copy().get_data()[0]
whole_signal, whole_signal_info = ecg_process_localver(whole_ecg, sampling_rate=sampling_freq)
nk.ecg_plot(whole_signal, whole_signal_info)



''' KROK 1.4 '''
# break_block onset tags identification. In a full dataset it should occur 4 times.

del whole_signal
del whole_signal_info

break_block_onsets = np.array([ann['onset'] for ann in ecg_mne.annotations if "break_blocks" in ann['description']])
if not len(break_block_onsets) == 4:
    raise ValueError("There are not four annotations!")

# Rough identification of blocks' borders. Later on, these will be cut precisely to trials' onsets & ends.

tmin = ecg_mne.times[0]
block_boundaries = [(tmin, break_block_onsets[0]),  # Block 1
                    (break_block_onsets[0], break_block_onsets[1]),  # Block 2
                    (break_block_onsets[1], break_block_onsets[2]),  # Block 3
                    (break_block_onsets[2], break_block_onsets[3])  # Block 4
                    ]

# Checkpoint to see if the borders location is correct

plt.plot(photo_mne.times, photo_mne.get_data()[0]*4, color='blue', alpha=0.3)
plt.vlines([bd for bl_pair in block_boundaries for bd in bl_pair], ymin=0, ymax=0.10, color='red')
plt.plot(ecg_mne.times, ecg_mne.get_data()[0], color='green', alpha=0.5)
plt.show()

#####################################

""" 
Step 2

In this loop, we go through the block boundaries and cut out the signal within those boundaries. We identify
the type of block we are dealing with, and then find the ‘break_trials’ tags in that block (in the complete dataset,
there should be exactly 5 per block, or 20 in total). Then, in a loop going through ‘trial_boundaries’, we go through each
of the trials and trim the investigated section counting from the onset of the first event in the trial to + 300 seconds – this way we get
precise trial duration windows (now only including the trial, without breaks or the moment of responding on the scales). 
Next, we process the ECG signal. 

We obtain:
- A dictionary with cleaned blocks and trials, with identified peaks, clean signal etc.
- ECG plot for each trial separately. Neurokit style.
- A dataframe with QC assessment.
"""


cwd = os.getcwd()
out_dir = os.path.join(cwd, "Results_ECG_GUI")
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)



cleaned_ecg_blocks_dict = {}
block_idx = 1
TRIAL_DURATION = 300

all_trial_boundaries = []
qc_rows = []
for start, end in block_boundaries:
    block = ecg_mne.copy().crop(tmin=start, tmax=end)
    block_type = None
    for ann in block.annotations:
        if 'wskazowka' in ann['description']:
            block_type = 'MCT'
            break
        elif 'licznik_start' in ann['description']:
            block_type = 'PVT'
            break
    assert block_type is not None
    break_trial_onsets = np.array([ann['onset'] for ann in block.annotations if "break_trials" in ann['description']])
    if not len(break_trial_onsets) == 5:
        raise ValueError("Jest mniej lub więcej niż pięć triali w bloku " + str(block_idx))
    trial_boundaries = [(start, break_trial_onsets[0]),  # Trial 1
                        (break_trial_onsets[0], break_trial_onsets[1]),  # Trial 2
                        (break_trial_onsets[1], break_trial_onsets[2]),  # Trial 3
                        (break_trial_onsets[2], break_trial_onsets[3]),  # Trial 4
                        (break_trial_onsets[3], break_trial_onsets[4])]  # Trial 5
    trial_number = 1
    block_trials = []
    block_trials_dict = {}
    precise_trial_boundaries = []
    for start_trial, end_trial in trial_boundaries:
        next = input("nowy trial?")
        trial = ecg_mne.copy().crop(tmin=start_trial, tmax=end_trial)
        precise_onset = None
        for annotation in trial.annotations:
            if block_type == "PVT":
                if 'isi_start' in annotation['description']:
                    precise_onset = annotation['onset']
                    break
            elif block_type == "MCT":
                if 'wskazowka' in annotation['description']:
                    precise_onset = annotation['onset']
                    break
        assert precise_onset is not None
        precise_trial = ecg_mne.copy().crop(tmin=precise_onset, tmax=precise_onset+TRIAL_DURATION)
        precise_trial_boundaries.append(precise_onset)

        ecg_signal = precise_trial.get_data()[0]
        sampling_freq = precise_trial.info['sfreq']

        signals, info = ecg_process_localver(ecg_signal, sampling_rate=sampling_freq)

        # local indices
        local_samples = info["ECG_R_Peaks"]

        # clean signal
        ecg_clean = signals["ECG_Clean"].values
        corrected_local, added_local, removed_local = correct_ecg_peaks_gui(ecg_clean, local_samples)
        plot_ecg_peak_correction(
            participant_file,
            block_idx,
            trial_number,
            ecg_clean,
            original_peaks=local_samples,
            added_peaks=added_local,
            removed_peaks=removed_local,
            final_peaks=corrected_local,
        )
        corrected_global = corrected_local + precise_trial.first_samp
        added_global = added_local + precise_trial.first_samp
        removed_global = removed_local + precise_trial.first_samp
        qc_rows.append({
            "dataset": participant_file,
            "block": block_idx,
            "trial": trial_number,

            "original_peaks": list(local_samples + precise_trial.first_samp),
            "added_peaks": list(added_global),
            "removed_peaks": list(removed_global),
            "final_peaks": list(corrected_global),

            "N_original": len(local_samples),
            "N_added": len(added_global),
            "N_removed": len(removed_global),
            "N_final": len(corrected_global)
        })
        n_samples = len(ecg_signal)
        ECG_R_Peaks = np.zeros(n_samples)
        ECG_R_Peaks[corrected_local] = 1
        ECG_Rate = nk.signal_rate(
            peaks=corrected_local,
            sampling_rate=sampling_freq,
            desired_length=n_samples
        )

        df_trial = pd.DataFrame({
            "ECG_Raw": ecg_signal,
            "ECG_Clean": ecg_clean,
            "ECG_Rate": ECG_Rate,
            "ECG_Quality": signals["ECG_Quality"].values,
            "ECG_R_Peaks": ECG_R_Peaks
        })
        info_plot = {
            "ECG_R_Peaks": corrected_local,
            "sampling_rate": sampling_freq
        }

        nk.ecg_plot(df_trial, info_plot)

        fig = plt.gcf()
        fig.savefig(
            participant_file
            + "_ecg_block_" + str(block_idx)
            + "_trial_" + str(trial_number)
            + "_corrected_via_gui.png"
        )
        plt.close(fig)

        precise_trial_times = precise_trial.times
        precise_trial_vals = precise_trial.get_data()[0]
        block_trials_dict[block_type + "_block_" + str(block_idx) + "_trial_" + str(trial_number)] = df_trial
        block_trials.append({'trial_no': trial_number, 'trials': ecg_mne.copy().crop(tmin=precise_onset, tmax=precise_onset+TRIAL_DURATION).get_data()[0],
                             'trials_times': ecg_mne.copy().crop(tmin=precise_onset, tmax=precise_onset+TRIAL_DURATION).times})
        trial_number += 1
    all_trial_boundaries.append(precise_trial_boundaries)
    cleaned_ecg_blocks_dict[block_type + "_block_" + str(block_idx)] = block_trials_dict
    block_idx += 1


''' Step 2.2'''

# Creating a dataframe with clean ECG to export to CSV

dfs = []
for block_name, block in cleaned_ecg_blocks_dict.items():
    for trial_name, trial_df in block.items():
        trial_df["block"] = block_name
        trial_df["trial"] = trial_name
        dfs.append(trial_df)
cleaned_ecg_df = pd.concat(dfs)
cleaned_ecg_df.to_csv(participant_file+"_ecg_blocks_cleaned.csv", sep=",", na_rep="NA", index=False)
# Drop columns block and trial just in case
for block_name, block in cleaned_ecg_blocks_dict.items():
    for trial_name, trial_df in block.items():
        trial_df.drop(columns=["block", "trial"], inplace=True)
# Checkpoint with plotted borders of blocks and trials to visually check if the split was correct

all_trial_boundaries_plot = [boundary for tr_boundaries in all_trial_boundaries for boundary in tr_boundaries]
plt.plot(ecg_mne.times, ecg_mne.get_data()[0], color='blue', alpha=0.7)
plt.plot(photo_mne.times, photo_mne.get_data()[0]*4, color='orange', alpha=0.7)
plt.vlines(x=all_trial_boundaries_plot, ymin=0, ymax=0.08, color='green')
plt.vlines(x=[elem+TRIAL_DURATION for elem in all_trial_boundaries_plot], ymin=0, ymax=0.08, colors='red')
plt.savefig(participant_file+"_precise_blocks_trials_split_overview_ECG.png")
plt.show()

qc_dataframe = pd.DataFrame(qc_rows)
qc_filename = f"{participant_file}_ECGpeaks_QC.csv"
qc_dataframe.to_csv(qc_filename, index=False)

""" Step 3


ECG analysis -- calculation of the signal's properties for each 5 minute trial.
We go through the dictionary of blocks, and within them through dict of trial for the block, containing the dataframes being
the result of ecg_process(...).

"""

analysis_subset = ['Participant_ID', 'Full_name', 'Block_type', 'Block_order', 'Block_order_per_type', 'Trial',
                   'ECG_Rate_Mean', 'HRV_SDNN', 'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_LFHF']

ecg_features_all = []
ecg_features_subset = []

for block_key in cleaned_ecg_blocks_dict.keys():
    for trial_key in cleaned_ecg_blocks_dict[block_key].keys():
        features = nk.ecg_analyze(cleaned_ecg_blocks_dict[block_key][trial_key], sampling_rate=sampling_freq, method="interval-related")
        tkey = trial_key.split("_")
        features.insert(0, 'Participant_ID', participant_file.split("_")[0])
        features.insert(1, 'Full_name', trial_key)
        features.insert(2, 'Block_type', tkey[0])
        features.insert(3, 'Block_order', int(tkey[2]))
        if int(tkey[2]) in [1, 2]:
            features.insert(4, 'Block_order_per_type', 1)
        elif int(tkey[2]) in [3, 4]:
            features.insert(4, 'Block_order_per_type', 2)
        features.insert(5, 'Trial', int(tkey[-1]))
        if len(features.index) != 1:
            raise ValueError("Jest inna liczba wierszy niż powinna być!")
        for colname in features.columns:
            if isinstance(features.loc[0, colname], np.ndarray):
                features.loc[0, colname] = features.loc[0, colname].item()
        subset = features.filter(items=analysis_subset, axis='columns')
        ecg_features_all.append(features)
        ecg_features_subset.append(subset)

assert len(ecg_features_all) == 20
assert len(ecg_features_subset) == 20
ecg_features_all_df = pd.concat(ecg_features_all)
ecg_features_subset_df = pd.concat(ecg_features_subset)

ecg_features_all_df.to_csv(participant_file+"_ecg_analyzed_all_features.csv", sep=",", na_rep="NA", index=False)
ecg_features_subset_df.to_csv(participant_file+"_ecg_analyzed_subset_features.csv", sep=",", na_rep="NA", index=False)

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12,12))
ax1.plot(np.arange(1, 21), ecg_features_subset_df['ECG_Rate_Mean'], linestyle='dashed', color='blue')
ax1.title.set_text('ECG Rate Mean')
ax1.set_xlabel('Trials [n]')
ax1.set_ylabel('ECG Rate [bpm]')
ax1.set_ylim([40, 130])
ax2.plot(np.arange(1, 21), ecg_features_subset_df['HRV_SDNN'], linestyle='dashed', color='blue')
ax2.title.set_text('HRV SDNN')
ax2.set_xlabel('Trials [n]')
ax2.set_ylabel('HRV SDNN [ms]')
ax2.set_ylim([30, 180])
ax3.plot(np.arange(1, 21), ecg_features_subset_df['HRV_LFHF'], linestyle='dashed', color='blue')
ax3.title.set_text('HRV LF/HF ratio')
ax3.set_xlabel('Trials [n]')
ax3.set_ylabel('HRV LF/HF ratio')
ax3.set_ylim([0, 7])
ax4.plot(np.arange(1, 21), ecg_features_subset_df['HRV_LF'], linestyle='dashed', color='blue')
ax4.plot(np.arange(1, 21), ecg_features_subset_df['HRV_HF'], linestyle='dashed', color='green')
ax4.plot(np.arange(1, 21), ecg_features_subset_df['HRV_VLF'], linestyle='dashed', color='orange')
ax4.title.set_text('All freqs overview (VLF, LF, HF)')
ax4.legend(['LF', 'HF', 'VLF'])
ax4.set_xlabel('Trials [n]')
ax4.set_ylabel('Spectral power of HRV [ms^2]')
ax4.set_ylim([0, 0.15])
plt.suptitle(participant_file.split("_")[0])
plt.savefig(participant_file+"_ECG_overview.png")
fig.show()

##
