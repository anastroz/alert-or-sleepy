import numpy as np
import pandas as pd
import os
import json
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("TkAgg")

'''
This script finds the .obci.tag file, from which it extracts the data and stores them in the .json files.
Subjective assessments are extracted here and saved in separate files.

'''


def strip_obci_tag_extension(filename):

    """
    Goal: remove the extension .obci.tag from the filename to use the name later on

    :param filename: filename .obci.tag to strip
    """

    # Remove ".tag"
    name_without_tag, _ = os.path.splitext(filename)
    # Remove ".obci"
    name_without_obci, _ = os.path.splitext(name_without_tag)
    return name_without_obci


def save_preprocessed_json(data, filename="tags.json"):
    """
    Reads a list of dictionaries (which it receives within the parse_tags function)
    and saves it to the JSON file in PREPROCESSED_DATA directory.

    :param data: list of dicts extracted from the XML file (obci.tag)
    :param filename: name for the resulting JSON file

    """

    current_dir = os.getcwd()

    target_dir = os.path.join(current_dir, "PREPROCESSED_DATA")

    os.makedirs(target_dir, exist_ok=True)

    output_path = os.path.join(target_dir, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Zapisano plik w: {output_path}")


def parse_tags(obci_tag_file):

    """
    This function has to chronologically extract all the tags from the participant's XML file. At the same time, some of
    the tags are rejected/omitted (e.g., channelNumber or length), as they are useless for us. 
    
    :param obci_tag_file: source file with tags with extension obci.tag.

    """
    # Reading the XML structure
    tree = ET.parse(obci_tag_file)
    root = tree.getroot()

    results = []  

    tag_container = root.find("tagData/tags")
    if tag_container is None:
        raise RuntimeError("Nie znaleziono tagów w sekcji tagData/tags")

    
    for tag in tag_container.findall("tag"):
        entry = {
            "name": tag.get("name"),
            "position": float(tag.get("position"))
        } 

        name = entry["name"]

        
        if name in ["KSS_answer", "NSSs_answer", "NSSa_answer"]:
            entry["number"] = int(tag.find("number").text)
            entry["label"] = tag.find("label").text

        elif name == "VASf_answer":
            entry["number"] = float(tag.find("number").text)

        elif name == "PVT_run":
            entry["isi"] = float(tag.find("isi").text)
            entry["RT"] = tag.find("RTs").text
            entry["Accuracy"] = tag.find("Accuracy").text

        elif name == "MCT_run":
            entry["Trial_type"] = tag.find("trial_type").text
            entry["RT"] = tag.find("RT_MCT").text
            entry["Accuracy"] = tag.find("Accuracy_MCT").text

        
        results.append(entry)

    source_data_name = strip_obci_tag_extension(obci_tag_file)
    dataset_name = source_data_name+".json"
    save_preprocessed_json(results, filename=dataset_name)
    return results, source_data_name


def find_obci_tag_file():

    """
    This function has to find the file with obci.tag extension. There should be only a single file with such extension per participant.
    """

    files = [f for f in os.listdir('.') if f.endswith(".obci.tag")]

    if len(files) == 0:
        raise FileNotFoundError("Brak pliku z rozszerzeniem .obci.tag.")
    if len(files) > 1:
        raise RuntimeError("Znaleziono więcej niż jeden plik .obci.tag!")

    return files[0]


def plot_number_vs_position(datasets, source_data_name, ylabel="number"):

    """
    Draws several subplots with number vs. position

    :param datasets: list of lists with dicts with the data, e.g., [odpowiedzi_KSS, odpowiedzi_NSSs, ...]
    :param source_data_name: filename without the extension "obci.tag"
    :param ylabel: Y axis label

    """
    titles = ["KSS", "SAS_sleepiness", "SAS_alertness", "VASf"]
    ylims = [(1, 9), (0, 4), (0, 4), (0, 100)]
    # Creating subplots 2x2
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes = axes.flatten()

    for ax, data, title, ylim in zip(axes, datasets, titles, ylims):
        positions = [tag["position"] for tag in data]
        numbers = [tag["number"] for tag in data]
        ax.plot(positions, numbers, marker='o')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)

    # Adding a common X axis label
    for ax in axes[2:]:  # bottom row
        ax.set_xlabel("position (czas)")

    filepath = os.path.join("PREPROCESSED_DATA", source_data_name+"_SUBJECTIVE")
    plt.savefig(filepath, dpi=300)
    plt.tight_layout()
    #plt.show()


def create_participant_df(participant, sorted_answers, max_answers=24):

    """
    The function creates a single-row dataframe for each participant, which aggregates all their answers using a given scale, as well as the answer time.

    :param participant: participant ID (PXXX_DAY or PXXX_NIGHT).
    :param sorted_answers: chronologically sorted answers extracted from the list of dicts

    """

    participant_id, session = participant.split("_")
    row_data = {"ID": participant_id, "Session": session.capitalize()}

    for i in range(max_answers):
        ans_col = f"ans{i + 1}"
        time_col = f"ans{i + 1}_time"
        if i < len(sorted_answers):
            row_data[ans_col] = sorted_answers[i]["number"]
            row_data[time_col] = sorted_answers[i]["position"]
        else:
            row_data[ans_col] = pd.NA
            row_data[time_col] = pd.NA

    df = pd.DataFrame([row_data])
    return df

##########


datapath = ".../DATA_TO_PROCESS/"
all_participants_KSS = pd.DataFrame()
all_participants_NSSs = pd.DataFrame()
all_participants_NSSa = pd.DataFrame()
all_participants_VASf = pd.DataFrame()
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
    os.chdir(datapath+participant_ID+"/corrected_tags/")
    obci_tag_participant_file = find_obci_tag_file()
    tags_list, src_data_name = parse_tags(obci_tag_participant_file)

    # here we split the tags according to the answer type and sort using the "position" tag (time)
    odpowiedzi_KSS = sorted([t for t in tags_list if t["name"] == "KSS_answer"],  key=lambda x: x["position"])
    odpowiedzi_NSSs = sorted([t for t in tags_list if t["name"] == "NSSs_answer"], key=lambda x: x["position"])
    odpowiedzi_NSSa = sorted([t for t in tags_list if t["name"] == "NSSa_answer"], key=lambda x: x["position"])
    odpowiedzi_VASf = sorted([t for t in tags_list if t["name"] == "VASf_answer"], key=lambda x: x["position"])
    subj_data = [odpowiedzi_KSS, odpowiedzi_NSSs, odpowiedzi_NSSa, odpowiedzi_VASf]
    plot_number_vs_position(subj_data, src_data_name, ylabel="number")

    kss_row = create_participant_df(participant_ID, odpowiedzi_KSS)
    all_participants_KSS = pd.concat([all_participants_KSS, kss_row], ignore_index=True)
    nss_sleep_row = create_participant_df(participant_ID, odpowiedzi_NSSs)
    all_participants_NSSs = pd.concat([all_participants_NSSs, nss_sleep_row], ignore_index=True)
    nss_alert_row = create_participant_df(participant_ID, odpowiedzi_NSSa)
    all_participants_NSSa = pd.concat([all_participants_NSSa, nss_alert_row], ignore_index=True)
    vasf_row = create_participant_df(participant_ID, odpowiedzi_VASf, max_answers=5)
    all_participants_VASf = pd.concat([all_participants_VASf, vasf_row], ignore_index=True)

os.chdir(datapath+"Corrected_tags-Ready_for_analysis/")
all_participants_KSS.to_csv("CT-KSS_data_all.csv", index=False)
all_participants_NSSs.to_csv("CT-SAS_sleepiness_data_all.csv", index=False)
all_participants_NSSa.to_csv("CT-SAS_alertness_data_all.csv", index=False)
all_participants_VASf.to_csv("CT-VASf_data_all.csv", index=False)
