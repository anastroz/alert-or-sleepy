# import numpy as np
# import pandas as pd
# import os
# import json
# import xml.etree.ElementTree as ET
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("TkAgg")
#
# """
# ONLY FOR THE DATASETS: P002_DAY, P008_NIGHT, P011_NIGHT, P041_NIGHT
#
#
# I. Code to extract the data from the obci.tag. Below there is also code (which should be run separately, it's not a single pipeline)
# for behavioral data in datasets with issues (for complete datasets there is "Extract_behavioral_data"). The resulting files will be integrated manually
# with the other outcomes, using NA values where needed due to lacking data.
#
# """
#
#
# def strip_obci_tag_extension(filename):
#
#     """
#     Goal: remove the extension .obci.tag from the filename to use the name later on
#
#     :param filename: filename .obci.tag to strip
#     """
#
#     # Remove ".tag"
#     name_without_tag, _ = os.path.splitext(filename)
#     # Remove ".obci"
#     name_without_obci, _ = os.path.splitext(name_without_tag)
#     return name_without_obci
#
#
# def save_preprocessed_json(data, filename="tags.json"):
#     """
#     Reads a list of dictionaries (which it receives within the parse_tags function)
#     and saves it to the JSON file in PREPROCESSED_DATA directory.
#
#     :param data: list of dicts extracted from the XML file (obci.tag)
#     :param filename: name for the resulting JSON file#
#     """
#
#     current_dir = os.getcwd()
#
#     target_dir = os.path.join(current_dir, "PREPROCESSED_DATA")
#
#     os.makedirs(target_dir, exist_ok=True)
#
#     output_path = os.path.join(target_dir, filename)
#
#     with open(output_path, "w", encoding="utf-8") as f:
#         json.dump(data, f, ensure_ascii=False, indent=2)
#
#     print(f"Zapisano plik w: {output_path}")
#
#
# def parse_tags(obci_tag_file):
#
#     """
#     This function has to chronologically extract all the tags from the participant's XML file. At the same time, some of
#     the tags are rejected/omitted (e.g., channelNumber or length), as they are useless for us. 
#    
#     :param obci_tag_file: source file with tags with extension obci.tag.
#
#     """
#     # Reading the XML structure
#     tree = ET.parse(obci_tag_file)
#     root = tree.getroot()
#
#     results = []  
#
#     tag_container = root.find("tagData/tags")
#     if tag_container is None:
#         raise RuntimeError("Nie znaleziono tagów w sekcji tagData/tags")
#
#     # Przejście chronologicznie po tagach
#     for tag in tag_container.findall("tag"):
#         entry = {
#             "name": tag.get("name"),
#             "position": float(tag.get("position"))
#         } 
#
#         name = entry["name"]
#
#         if name in ["KSS_answer", "NSSs_answer", "NSSa_answer"]:
#             entry["number"] = int(tag.find("number").text)
#             entry["label"] = tag.find("label").text
#
#         elif name == "VASf_answer":
#             entry["number"] = float(tag.find("number").text)
#
#         elif name == "PVT_run":
#             entry["isi"] = float(tag.find("isi").text)
#             entry["RT"] = tag.find("RTs").text
#             entry["Accuracy"] = tag.find("Accuracy").text
#
#         elif name == "MCT_run":
#             entry["Trial_type"] = tag.find("trial_type").text
#             entry["RT"] = tag.find("RT_MCT").text
#             entry["Accuracy"] = tag.find("Accuracy_MCT").text
#
#         results.append(entry)
#
#     source_data_name = strip_obci_tag_extension(obci_tag_file)
#     dataset_name = source_data_name+".json"
#     save_preprocessed_json(results, filename=dataset_name)
#     return results, source_data_name
#
#
# def find_obci_tag_file():
#
#     """
#     This function has to find the file with obci.tag extension. There should be only a single file with such extension per participant.
#     """
#
#     files = [f for f in os.listdir('.') if f.endswith(".obci.tag")]
#
#     if len(files) == 0:
#         raise FileNotFoundError("Brak pliku z rozszerzeniem .obci.tag.")
#     if len(files) > 1:
#         raise RuntimeError("Znaleziono więcej niż jeden plik .obci.tag!")
#
#     return files[0]
#
#
# def plot_number_vs_position(datasets, source_data_name, ylabel="number"):
#
#     """
#     Draws several subplots with number vs. position
#
#     :param datasets: list of lists with dicts with the data, e.g., [odpowiedzi_KSS, odpowiedzi_NSSs, ...]
#     :param source_data_name: filename without the extension "obci.tag"
#     :param ylabel: Y axis label
#
#     """
#     titles = ["KSS", "SAS_sleepiness", "SAS_alertness", "VASf"]
#     ylims = [(1, 9), (0, 4), (0, 4), (0, 100)]
#     # Creating subplots 2x2
#     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
#
#     axes = axes.flatten()
#
#     for ax, data, title, ylim in zip(axes, datasets, titles, ylims):
#         positions = [tag["position"] for tag in data]
#         numbers = [tag["number"] for tag in data]
#         ax.plot(positions, numbers, marker='o')
#         ax.set_title(title)
#         ax.set_ylabel(ylabel)
#         ax.set_ylim(ylim)
#
#     # Adding a common X axis label
#     for ax in axes[2:]:  # bottom row
#         ax.set_xlabel("position (czas)")
#
#     filepath = os.path.join("PREPROCESSED_DATA", source_data_name+"_SUBJECTIVE")
#     plt.savefig(filepath, dpi=300)
#     plt.tight_layout()
#     #plt.show()
#
#
# def create_participant_df(participant, sorted_answers, max_answers=24):
#
#     """
#     The function creates a single-row dataframe for each participant, which aggregates all their answers using a given scale, as well as the answer time.
#
#     :param participant: participant ID (PXXX_DAY or PXXX_NIGHT).
#     :param sorted_answers: chronologically sorted answers extracted from the list of dicts
#
#     """
#
#     participant_id, session = participant.split("_")
#     row_data = {"ID": participant_id, "Session": session.capitalize()}
#
#     for i in range(max_answers):
#         ans_col = f"ans{i + 1}"
#         time_col = f"ans{i + 1}_time"
#         if i < len(sorted_answers):
#             row_data[ans_col] = sorted_answers[i]["number"]
#             row_data[time_col] = sorted_answers[i]["position"]
#         else:
#             row_data[ans_col] = pd.NA
#             row_data[time_col] = pd.NA
#
#     df = pd.DataFrame([row_data])
#     return df
#
# ##########
#
#
# datapath = ".../DATA_TO_PROCESS/datasets_with_issues/"
#
# participant_ID = "P041_NIGHT"
# os.chdir(datapath+participant_ID+"/corrected_tags/")
# obci_tag_participant_file = find_obci_tag_file()
# tags_list, src_data_name = parse_tags(obci_tag_participant_file)
#
# # here we split the tags according to the answer type and sort using the "position" tag (time)
# odpowiedzi_KSS = sorted([t for t in tags_list if t["name"] == "KSS_answer"],  key=lambda x: x["position"])
# odpowiedzi_NSSs = sorted([t for t in tags_list if t["name"] == "NSSs_answer"], key=lambda x: x["position"])
# odpowiedzi_NSSa = sorted([t for t in tags_list if t["name"] == "NSSa_answer"], key=lambda x: x["position"])
# odpowiedzi_VASf = sorted([t for t in tags_list if t["name"] == "VASf_answer"], key=lambda x: x["position"])
# subj_data = [odpowiedzi_KSS, odpowiedzi_NSSs, odpowiedzi_NSSa, odpowiedzi_VASf]
# plot_number_vs_position(subj_data, src_data_name, ylabel="number")
#
# kss_row = create_participant_df(participant_ID, odpowiedzi_KSS)
# kss_row.to_csv(participant_ID+"_kss.csv", index=False, na_rep="NA")
# nss_sleep_row = create_participant_df(participant_ID, odpowiedzi_NSSs)
# nss_sleep_row.to_csv(participant_ID+"_nss_sleep.csv", index=False, na_rep="NA")
# nss_alert_row = create_participant_df(participant_ID, odpowiedzi_NSSa)
# nss_alert_row.to_csv(participant_ID+"_nss_alert.csv", index=False, na_rep="NA")
# vasf_row = create_participant_df(participant_ID, odpowiedzi_VASf, max_answers=5)
# vasf_row.to_csv(participant_ID+"_vasf.csv", index=False, na_rep="NA")


##########


# II. Part of the code to process behavioral data (from Extract_behavioral_data) for the datasets with issues. 
# 
#
import numpy as np
import pandas as pd
import os
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")


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
    Splits the events based on the "break_blocks" tag. Works with the files having 2, 3, or 4 blocks. If the block was completed
    only partially, it will be discarded. Only full, complete blocks are taken into consideration. The function
    splits the blocks according to PVT/MCT category as well.

    Returns:
        blocks_PVT  – a tuple of blocks containing tag "PVT_run"
        blocks_MCT  – a tuple of blocks containing tag "wskazowka"

    Throws errors when:
    - break_blocks == 0 or 1 --> no of blocks too small
    - the block would not contain the tags PVT_run/wskazowka
    - the block would contain both tags 
    - there are not exactly 2 PVT and 2 MCT blocks 

    :param events: list with dicts containing all the tags occurring during the experiment

    """

    # --- step 1: find break_blocks --- it has to find the indices of the dicts with the tag break_blocks. 
    # Such tags occurred after each completed block.
    break_indices = [i for i, ev in enumerate(events) if ev.get("name") == "break_blocks"]
    n_breaks = len(break_indices)

    if n_breaks < 2:
        raise ValueError(f"Za mało 'break_blocks': {n_breaks}. Dataset uszkodzony.")
    if n_breaks > 4:
        raise ValueError(f"Za dużo 'break_blocks': {n_breaks}. Maksymalnie 4.")

    # --- step 2: split into blocks -- everything before "break_blocks" is a part of the previous block.
    blocks = []
    prev = 0
    for b in break_indices:
        blocks.append(events[prev:b])
        prev = b + 1

    blocks_PVT = []
    blocks_MCT = []

    for idx, block in enumerate(blocks, start=1):

        if len(block) == 0:
            raise ValueError(f"Block {idx} jest pusty — błąd danych.")

        has_PVT = any(ev.get("name") == "PVT_run" for ev in block)
        has_MCT = any(ev.get("name") == "wskazowka" for ev in block)

        if has_PVT and has_MCT:
            raise ValueError(f"Block {idx} zawiera jednocześnie PVT i MCT — błąd danych.")

        if not has_PVT and not has_MCT:
            raise ValueError(f"Block {idx} nie zawiera ani PVT_run, ani wskazowka.")

        if has_PVT:
            blocks_PVT.append(block)
        else:
            blocks_MCT.append(block)

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


datapath = ".../DATA_TO_PROCESS/datasets_with_issues/"

participant_ID = "P011_NIGHT/first_half"
participant_path = datapath+participant_ID+"/corrected_tags/"
os.chdir(participant_path)
data_name, dataset = load_participant_dataset(participant_path)

bl1_PVT, bl1_MCT = split_into_blocks(dataset)

participant2_ID = "P011_NIGHT/second_half"
participant2_path = datapath+participant2_ID+"/corrected_tags/"
os.chdir(participant2_path)
data2_name, dataset2 = load_participant_dataset(participant2_path)

bl2_PVT, bl2_MCT = split_into_blocks(dataset2)

bl_PVT = tuple([bl1_PVT[0], bl2_PVT[0]])

bl_MCT = tuple([bl1_MCT[0], bl2_MCT[0]])



pvt_trials_bl1, pvt_trials_bl2 = split_blocks_into_trials(bl_PVT, "PVT")
mct_trials_bl1, mct_trials_bl2 = split_blocks_into_trials(bl_MCT, "MCT")
# for PVT
pvt_df = create_pvt_dataframe(pvt_trials_bl1, pvt_trials_bl2)
pvt_path = os.path.join(os.path.join(participant_path, "PREPROCESSED_DATA"), data_name + "_PVT.csv")
pvt_df.to_csv(pvt_path, index=False)
# for MCT
mct_df = create_mct_dataframe(mct_trials_bl1, mct_trials_bl2)
mct_path = os.path.join(os.path.join(participant_path, "PREPROCESSED_DATA"), data_name + "_MCT-CT.csv")
mct_df.to_csv(mct_path, index=False)

