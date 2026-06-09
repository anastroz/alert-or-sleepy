import numpy as np
import pandas as pd
import os
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


'''
The following scripts enable extraction of participants' behavioral data from the .json files, which were created from the .tag files.
The outcomes of these scripts are the .csv files separately for the MCT and the PVT tasks, with information regarding the block and trial order.
'''


def load_participant_dataset(participant_folder):
    """
    Enters participant's directory, searches for the .json file in PREPROCESSED_DATA,
    reads it and returns the data + filename.

    :param participant_folder: path to participant's directory e.g., ".../P001/"
    :return: (dataset_name, dataset_data)
    """

    # Path to PREPROCESSED_DATA
    preproc_path = os.path.join(participant_folder, "PREPROCESSED_DATA")

    if not os.path.isdir(preproc_path):
        raise FileNotFoundError(f"Brak katalogu: {preproc_path}")

    # Searching for .json
    json_files = [f for f in os.listdir(preproc_path) if f.lower().endswith(".json")]

    if len(json_files) == 0:
        raise FileNotFoundError(f"Nie znaleziono pliku .json w: {preproc_path}")
    if len(json_files) > 1:
        print(f"UWAGA: znaleziono więcej niż jeden JSON. Biorę pierwszy: {json_files[0]}")

    json_file = json_files[0]
    dataset_name = os.path.splitext(json_file)[0]

    json_path = os.path.join(preproc_path, json_file)

    # Reading JSON
    with open(json_path, "r", encoding="utf-8") as f:
        dataset_data = json.load(f)

    return dataset_name, dataset_data


def split_into_blocks(events):
    """
    Splits the events into four blocks, based on the "break_blocks" tag.
    If there are less than 4 complete blocks: if the block was completed only partially, it will be rejected.
    Only complete blocks are taken into consideration.
    The function splits the blocks according to PVT/MCT category as well.

    Returns:
        blocks_PVT  – a tuple of blocks containing tag "PVT_run"
        blocks_MCT  – a tuple of blocks containing tag "wskazowka"

    Throws errors when:
    - break_blocks != 4
    - the block would not contain the tags PVT_run/wskazowka
    - the block would contain both tags 
    - there are not exactly 2 PVT and 2 MCT blocks 

    :param events: list with dicts containing all the tags occurring during the experiment

    """

    # --- step 1: find break_blocks --- it has to find the indices of the dicts with the tag break_blocks. 
    # Such tags occurred after each completed block.
    break_indices = [i for i, ev in enumerate(events) if ev.get("name") == "break_blocks"]

    if len(break_indices) != 4:
        raise ValueError(
            f"Oczekiwano dokładnie 4 'break_blocks', znaleziono {len(break_indices)}"
        )

    b1, b2, b3, b4 = break_indices

    # --- step 2: split into blocks -- everything before "break_blocks" is a part of the previous block.
    block_1 = events[:b1]
    block_2 = events[b1 + 1: b2]
    block_3 = events[b2 + 1: b3]
    block_4 = events[b3 + 1: b4]

    blocks = [block_1, block_2, block_3, block_4]

    # --- step 3: splitting blocks per type ---
    blocks_PVT = []
    blocks_MCT = []

    for idx, block in enumerate(blocks, start=1):
        has_PVT = any(ev.get("name") == "PVT_run" for ev in block)
        has_MCT = any(ev.get("name") == "wskazowka" for ev in block)

        # Error: there is no tag "PVT_run" or "wskazowka"
        if not has_PVT and not has_MCT:
            raise ValueError(f"Block {idx} nie zawiera ani 'PVT_run', ani 'wskazowka'.")

        # Error: there are both tags
        if has_PVT and has_MCT:
            raise ValueError(f"Block {idx} zawiera jednocześnie PVT i MCT — błąd danych.")

        # Grouping:
        if has_PVT:
            blocks_PVT.append(block)
        else:
            blocks_MCT.append(block)

    # --- step 4: final validation -- in a correct dataset there should be exactly two blocks of each type (MCT, PVT)
    if len(blocks_PVT) != 2 or len(blocks_MCT) != 2:
        raise ValueError(
            f"Nie udało się podzielić bloków na 2 PVT i 2 MCT — "
            f"PVT: {len(blocks_PVT)}, MCT: {len(blocks_MCT)}"
        )

    return tuple(blocks_PVT), tuple(blocks_MCT)


def split_blocks_into_trials(blocks_tuple, task_type):
    """
    Arguments:
        blocks_tuple – a tuple of two blocks (e.g., blocks_PVT lub blocks_MCT)
        task_type   – 'PVT' or 'MCT'

    How it works:
        1. Splits each block into five trials (using "break_trials" tag). Everything before the tag will be a part of the previous trial.
        2. Cleans each trial, so that only these dicts are left:
             - 'PVT_run'  if task_type == 'PVT'
             - 'MCT_run'  if task_type == 'MCT'
        We want to have here only the behavioral outcomes -- other tags (e.g., self-assessment or technical tags) are not needed here.
        3. Returns:
             (trials_block1, trials_block2)
        where each variable is a tuple:
             (trial_1, trial_2, trial_3, trial_4, trial_5)
    """

    # checking the arguments
    if task_type not in ("PVT", "MCT"):
        raise ValueError("task_type musi być 'PVT' albo 'MCT'")

    if len(blocks_tuple) != 2:
        raise ValueError("blocks_tuple musi zawierać dokładnie dwa bloki")

    # helping function to split block into trials
    def split_block_into_trials(block):
        break_indices = [i for i, ev in enumerate(block) if ev.get("name") == "break_trials"]

        if len(break_indices) != 5:
            raise ValueError(
                f"Oczekiwano dokładnie 5 'break_trials' w bloku, znaleziono {len(break_indices)}"
            )

        t1, t2, t3, t4, t5 = break_indices

        trial_1 = block[:t1]
        trial_2 = block[t1 + 1: t2]
        trial_3 = block[t2 + 1: t3]
        trial_4 = block[t3 + 1: t4]
        trial_5 = block[t4 + 1: t5]

        return (trial_1, trial_2, trial_3, trial_4, trial_5)

    # we decide which key should be left
    if task_type == "PVT":
        keep_name = "PVT_run"
    else:
        keep_name = "MCT_run"

    cleaned_blocks_trials = []

    # we process each of the two blocks
    for bl in blocks_tuple:
        # --- 1. Split into trials ---
        raw_trials = split_block_into_trials(bl)

        # --- 2. Cleaning trials of tags which won't be used ---
        cleaned_trials = tuple(
            [ev for ev in trial if ev.get("name") == keep_name]
            for trial in raw_trials
        )

        cleaned_blocks_trials.append(cleaned_trials)

    # returns the tuple of two block x trials. In the tuple there are two tuples with five lists, and each list contains the dictionaries with events from a given trial.
    return tuple(cleaned_blocks_trials)


def create_pvt_dataframe(pvt_trials_block1, pvt_trials_block2):
    """
    Creates a DataFrame for PVT with the following columns:
    "Block", "BlockTrial", "Position", "ISI", "RT", "Accuracy".

    :param pvt_trials_block1: first PVT block
    :param pvt_trials_block2: second PVT block
    """
    df_rows = []

    # going through the two blocks
    for block_idx, block_trials in enumerate([pvt_trials_block1, pvt_trials_block2], start=1):
        # block_trials is a tuple of 5 trials
        for trial_idx, trial in enumerate(block_trials, start=1):
            for event in trial:
                df_rows.append({
                    "Block_type": "PVT",
                    "Block_order": np.nan,
                    "Block_order_per_type": block_idx,
                    "Trial": trial_idx,
                    "Position": event.get("position"),
                    "ISI": event.get("isi"),
                    "RT": event.get("RT"),
                    "Accuracy": event.get("Accuracy")
                })

    df = pd.DataFrame(df_rows)
    return df


def create_mct_dataframe(mct_trials_block1, mct_trials_block2):
    """
    Creates a DataFrame for the MCT with the following columns:
    "Block", "BlockTrial", "Position", "Trial_type", "RT", "Accuracy".

    :param mct_trials_block1: first MCT block
    :param mct_trials_block2: second MCT block
    """
    df_rows = []

    # going through the two blocks
    for block_idx, block_trials in enumerate([mct_trials_block1, mct_trials_block2], start=1):
        # block_trials is a tuple of 5 trials
        for trial_idx, trial in enumerate(block_trials, start=1):
            for event in trial:
                df_rows.append({
                    "Block_type": "MCT",
                    "Block_order": np.nan,
                    "Block_order_per_type": block_idx,
                    "Trial": trial_idx,
                    "Position": event.get("position"),
                    "Trial_type": event.get("Trial_type"),
                    "RT": event.get("RT"),
                    "Accuracy": event.get("Accuracy")
                })

    df = pd.DataFrame(df_rows)
    return df


datapath = ".../DATA_TO_PROCESS/"

participants_data = ['P001_DAY', 'P003_DAY', 'P004_DAY', 'P005_DAY', 'P006_DAY', 'P007_DAY', 'P009_NIGHT', 'P010_NIGHT',
                     'P012_NIGHT', 'P013_DAY', 'P014_NIGHT', 'P015_NIGHT', 'P016_NIGHT', 'P017_NIGHT', 'P018_NIGHT',
                     'P019_NIGHT', 'P020_DAY', 'P021_NIGHT', 'P022_DAY', 'P023_DAY', 'P024_DAY', 'P025_DAY', 'P027_DAY',
                     'P028_DAY', 'P029_NIGHT', 'P030_NIGHT', 'P031_DAY', 'P032_DAY', 'P033_DAY', 'P034_DAY', 'P035_NIGHT',
                     'P036_NIGHT', 'P037_DAY', 'P038_NIGHT', 'P039_DAY', 'P040_DAY', 'P042_NIGHT', 'P043_DAY', 'P044_NIGHT',
                     'P045_DAY', 'P046_DAY', 'P048_DAY', 'P049_DAY', 'P050_DAY', 'P051_NIGHT', 'P052_DAY', 'P053_NIGHT',
                     'P054_DAY', 'P055_NIGHT', 'P056_NIGHT', 'P057_DAY', 'P058_NIGHT', 'P059_NIGHT', 'P060_DAY',
                     'P061_NIGHT', 'P062_NIGHT', 'P063_NIGHT', 'P064_NIGHT', 'P065_DAY', 'P066_DAY', 'P067_NIGHT',
                     'P068_DAY', 'P069_NIGHT', 'P070_NIGHT', 'P071_NIGHT', 'P072_NIGHT', 'P073_NIGHT', 'P074_NIGHT',
                     'P075_NIGHT', 'P076_NIGHT', 'P078_NIGHT', 'P079_NIGHT', 'P080_NIGHT', 'P081_DAY', 'P082_NIGHT',
                     'P083_DAY', 'P084_NIGHT', 'P085_NIGHT', 'P086_NIGHT', 'P087_DAY', 'P088_NIGHT', 'P089_DAY',
                     'P090_NIGHT']


for person in participants_data:
    participant_ID = person
    participant_path = datapath+participant_ID+"/corrected_tags/"
    os.chdir(participant_path)
    data_name, dataset = load_participant_dataset(participant_path)

    bl_PVT, bl_MCT = split_into_blocks(dataset)
    pvt_trials_bl1, pvt_trials_bl2 = split_blocks_into_trials(bl_PVT, "PVT")
    mct_trials_bl1, mct_trials_bl2 = split_blocks_into_trials(bl_MCT, "MCT")
    # for PVT
    pvt_df = create_pvt_dataframe(pvt_trials_bl1, pvt_trials_bl2)
    pvt_path = os.path.join(os.path.join(participant_path, "PREPROCESSED_DATA"), data_name + "_PVT-CT.csv")
    pvt_df.to_csv(pvt_path, index=False)
    # for MCT
    mct_df = create_mct_dataframe(mct_trials_bl1, mct_trials_bl2)
    mct_path = os.path.join(os.path.join(participant_path, "PREPROCESSED_DATA"), data_name + "_MCT-CT.csv")
    mct_df.to_csv(mct_path, index=False)
