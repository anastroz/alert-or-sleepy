import numpy as np
import matplotlib.pyplot as plt
import obci_readmanager.signal_processing.read_manager as obci_rm
import os
import mne
import xml.etree.ElementTree as ET


'''

The code below is used to correct tags that are offset relative to the photodiode peaks. Tags for the tasks being performed
are shifted based on the identified photodiode peaks, and the position of tags not associated with photodiode peaks (e.g. tags
relating to scale responses or tags relating to pauses) is adjusted relative to the tags corrected for the photodiode.

'''


def detect_peaks_windowed_overlap(
    signal,
    baseline,
    threshold,
    sfreq,
    window_sec=1.5,
    overlap_sec=0.5,
    min_distance_sec=0.5
):

    """
    
    The purpose of this function is to identify photodiode peaks by using a sliding window that detects the largest possible deviation
    and exceeds a specified threshold. In the second step, the function deduplicates the identified peaks, as it is possible that a given peak
    be detected twice if, after the window has moved, it is still within the peak search area.

    :param signal: input data from the photodiode (numpy array)
    :param baseline: manually determined baseline for the photodiode signal
    :param threshold: manually determined threshold for the photodiode signal
    :param sfreq: sampling frequency of the photodiode signal
    :param window_sec: length of the sliding window
    :param overlap_sec: length of the overlap section
    :param min_distance_sec: minimum distance between two consecutive photodiode peaks
    :return: returns the indices and values of the identified peaks

    """

    n_samples = len(signal)  # no of samples in the signal

    window_size = int(window_sec * sfreq)  # sliding window length
    overlap_size = int(overlap_sec * sfreq)  # overlap lengthh
    step = window_size - overlap_size  # step length

    peak_indices = []
    peak_values = []

    # here we go from 0 to the end with step 
    for start in range(0, n_samples, step):
        end = min(start + window_size, n_samples)  # at the end of the signal it will use a shorter window if needed
        segment = signal[start:end]

        if len(segment) == 0:  # if any segment would be of 0 length (shouldn't be the case)
            raise ValueError("Zerowy segment")

        deviation = np.abs(segment - baseline)  # for the whole segment the abs value of diff is computed per each sample 

        if not np.any(deviation >= threshold):  # if no sample is sufficiently distant, we keep searching 
            continue

        idx = np.argmax(deviation)  # picking the largest deviation

        peak_indices.append(start + idx)  # keeping peak index
        peak_values.append(signal[start + idx])  # keeping peak value

    # --- deduplication ---
    if not peak_indices:  # if empty, we finish here
        return np.array([]), np.array([])

    peak_indices = np.array(peak_indices)  # creating numpy arrays
    peak_values = np.array(peak_values)

    keep_idx = [0]  # starting index (of first peak -- if the second is larger, its index will jump in 
    min_dist = int(min_distance_sec * sfreq)  # minimal distance to assume that peaks are separate (half a second is assumed)

    for i in range(1, len(peak_indices)):  # we start with the second peak and keep deduplicating
        if peak_values[i] > 0.02:
            continue
        if peak_indices[i] - peak_indices[keep_idx[-1]] >= min_dist:  # if their diff is greater than a min_dist, we keep them both
            keep_idx.append(i)
        else:
            # keep the peak with a larger deviation
            prev = keep_idx[-1]
            if abs(peak_values[i] - baseline) > abs(peak_values[prev] - baseline):  # overwriting the index of the previous if the latter is larger
                keep_idx[-1] = i  # overwrite

    return peak_indices[keep_idx], peak_values[keep_idx]  # returns sublists of these peaks which are left after the deduplication



# Reading the signal based on the tutorial available here: http://moduly-analizy-danych.braintech.pl/rozdz2.html


# Reading SVAROG dataset:

os.chdir('.../DATA_TO_PROCESS/datasets_with_issues/P041_NIGHT')
filename = "P041_NIGHT_2025-05-29-23-55-00"
rawdata = obci_rm.ReadManager(filename+".obci.xml", filename+".obci.raw", filename+".obci.tag")
rawdata = rawdata.get_mne_raw()
signals = rawdata.pick(['ECG', 'Photo'])
sampling_freq = signals.info['sfreq']
del(rawdata)

# Preview of the signal with tags (it may take a while)

photo = signals.pick(['Photo'])
photo.plot()

# Preview of signal without tags

photo_wo_annotations = photo.copy()
photo_wo_annotations.set_annotations(mne.Annotations([], [], []))
photo_wo_annotations.plot()

# Plot to visually determine the threshold and baseline:

plt.plot(photo.get_data()[0])

# Setting the threshold and baseline plus the initial samples to trim

detection_baseline = 0.017
detection_threshold = 0.012
beginning_to_cut = 20000
end_to_cut = 0

# Source data (already in numpy)
photo_array = photo_wo_annotations.get_data()[0]
photo_times = photo_wo_annotations.times

# Preview of trimmed data
photo_array = photo_array[beginning_to_cut:len(photo_array)-end_to_cut]
photo_times = photo_times[beginning_to_cut:len(photo_times)-end_to_cut]
plt.plot(photo_times, photo_array)

# Tags filtration (only "wskazowka" and "licznik_start", because these are associated with a photodiode trigger)

filtered_annotations = photo.annotations.copy()

KEYWORDS = ("licznik_start", "wskazowka")

# here only these two tags are left, because we want to investigate the diff between their onset and identified peaks

filtered_ann = [
    ann for ann in filtered_annotations
    if any(k in ann["description"] for k in KEYWORDS)
]

# Photodiode peaks detection:

peak_idx, peak_vals = detect_peaks_windowed_overlap(photo_array, detection_baseline, detection_threshold, sampling_freq)
peak_times = photo_times[peak_idx]  # we want to have a time vector for peaks

filtered_ann_times = [ann['onset'] for ann in filtered_ann]  # using the tags onset
filtered_ann_vals = np.zeros(len(filtered_ann_times))

# Sanity check of peak detection:

print("Liczba wykrytych peaków w fotodiodzie: ", len(peak_idx))
print("Liczba zdarzeń z annotacji: ", len(filtered_ann))
assert len(peak_idx) == len(filtered_ann)

plt.plot(photo_times, photo_array)
plt.plot(filtered_ann_times, filtered_ann_vals, 'go')
plt.plot(peak_times, peak_vals, 'ro')
plt.show()

# Plot with diffs in tags onset 
diff_peaks_tags = peak_times-filtered_ann_times
plt.plot(peak_times, diff_peaks_tags)

# Reading the XML obci.tag and replacement of onset for wskazowka and licznik_start with the corrected one


tree = ET.parse(filename+".obci.tag")
root = tree.getroot()

TARGET_NAMES = {"licznik_start", "wskazowka"}
pos_iter = iter(peak_times)

offsets = []
old_pos_times = []
for tag in root.iter("tag"):
    name = tag.attrib.get("name")

    if name in TARGET_NAMES:
        try:
            new_pos = next(pos_iter)
        except StopIteration:
            raise RuntimeError("Brak wystarczającej liczby nowych pozycji!")

        old_pos = float(tag.attrib.get("position"))
        tag.set("position", str(float(new_pos)))
        offsets.append(abs(float(old_pos) - new_pos))  # offsets needed for subsequent interpolation
        old_pos_times.append(old_pos)  # old positions

        # print(f"{name}: {old_pos} → {new_pos}")

# shifts all the tags except TARGET_NAMES
offsets = [0] + offsets + [offsets[-1]]
# the last one is duplicated to make it easier to go through the regions (region before the first tag with a photodiode,
# then in between, and after the last tag with a photodiode)
old_pos_times = [0] + old_pos_times + [old_pos_times[-1]]
region = 0

print("here")
for tag in root.iter("tag"):
    name = tag.attrib.get("name")

    if name in TARGET_NAMES:
        region += 1
        continue

    old_pos = float(tag.attrib.get("position"))
    offset_l = offsets[region]
    offset_r = offsets[region + 1]
    pos_l = old_pos_times[region]
    pos_r = old_pos_times[region + 1]

    # explanation: below we calculate a weighted average based on the offsets identified earler (relative to the ground truth,
    # i.e., the photodiode). If we have a tag that is not accompanied by a photodiode, but there were photodiode peaks before
    # and after it, we also have their offsets, so we can interpolate the offset for the tag we are working on using a weighted
    # average (giving greater weight to the offset from the tag that is closer to the tag we are working on).

    if pos_l == pos_r:
        diff = offset_r
    else:
        diff = offset_l * abs(old_pos - pos_r) / abs(pos_l - pos_r) + offset_r * abs(old_pos - pos_l) / abs(pos_r - pos_l)

    new_pos = old_pos + diff

    tag.set("position", str(float(new_pos)))
    assert abs(old_pos - new_pos) < 1
    print(f"{name}: {old_pos} → {new_pos}")


cwd = os.getcwd()
out_dir = os.path.join(cwd, "corrected_tags")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir,  filename+"_corrected_tag_positions.obci.tag")

tree.write(
    out_path,
    encoding="utf-8",
    xml_declaration=True
)

os.chdir(out_dir)
with open(filename+"_detection_params.txt", 'w') as f:
    f.write('Detection baseline: ' + str(detection_baseline) + '\n')
    f.write('Detection threshold: ' + str(detection_threshold) + '\n')
    f.write('Beginning to cut: ' + str(beginning_to_cut) + '\n')
    f.write('End to cut: ' + str(end_to_cut) + '\n')
    f.write('Sampling frequency: ' + str(sampling_freq) + '\n')
