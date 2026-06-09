
"""
PVT:
- Correct responses: 100-1000 ms (inclusive). Correct (normal) responses: 100-500 ms left-side inclusive,
correct (lapse) responses: 500-1000 ms (inclusive).
- False starts: button before "+" onset OR responses shorter than 100 ms
- Omissions: RT = "Brak odpowiedzi" (No answer) OR answers above 1000 ms

- Each directory [P001_DAY etc.] contains an internal directory called PREPROCESSED_DATA, and this directory should contain
a single file ending in ‘_PVT.csv’.
- Each file contains the following columns:
---- ‘Block’ denoting the task block number (earlier = 1, later = 2),
---- ‘BlockTrial’ denoting the task number within a given block (there are five repetitions of the task in the block, hence the first occurrence = 1, the last = 5),
---- ‘Position’ indicating the time of the event within the task (may take the value NA if the clock has been reset, but this is true only for a single dataset and it's handled separately). 
---- "ISI" containing the waiting time for a stimulus to show.
---- "RT" containing reaction time in milliseconds. If the button was pressed too early, there will be a text starting with "Zbyt wczesnie" (too early). If the button was not pressed at all, "Brak odpowiedzi" (no answer)
     will be displayed, but if during the message (lasting 0.5 s) the button was pressed, it was registered correctly, but with the RT above 10 000 ms.
---- "Accuracy", with value "0" if someone responded too early or did not respond, or 1 if the response was correct. This column will not be used, as it may contain incorrect information.

"""


import os
import pandas as pd
import numpy as np
import re

# ----------------------------------------------------------
# 1. Cleaning RT duplicates
# ----------------------------------------------------------


def clean_pvt_early_press_duplicates(df):
    """
    Cleans artifacts due to the button being pressed for too long during PVT:
    - if two subsequent rows contain the info that the button was pressed,
      and the second row contains a "Too early" message (in Polish "Zbyt wczesnie") with RT < 100 ms,
      the second row should be set to NA in RT_copy, as it is the result of a prolonged button press
    """

    df = df.copy()

    if isinstance(df.loc[0, "RT_copy"], str) and df.loc[0, "RT_copy"].startswith("Zbyt wczesnie"):
        match_first = re.search(r"RT\s*=\s*(\d+)", df.loc[0, "RT_copy"])
        assert match_first is not None
        rt_ms_first = int(match_first.group(1))
        if rt_ms_first < 100:
            df.loc[0, "RT_copy"] = pd.NA

    for i in range(1, len(df)):
        prev_rt = df.loc[i - 1, "RT_copy"]
        curr_rt = df.loc[i, "RT_copy"]

        if isinstance(curr_rt, str) and curr_rt == "Zbyt wczesnie, RT = 0 ms":
            df.loc[i, "RT_copy"] = pd.NA
        elif isinstance(curr_rt, str) and curr_rt.startswith("Zbyt wczesnie"):
            match = re.search(r"RT\s*=\s*(\d+)", curr_rt)
            assert match is not None

            rt_ms = int(match.group(1))

            prev_had_press = (
                pd.notna(prev_rt)
                and (
                    isinstance(prev_rt, (int, float))
                    or (
                        isinstance(prev_rt, str)
                        and prev_rt.startswith("Zbyt wczesnie")
                    )
                )
            )

            if prev_had_press and rt_ms < 100:
                df.loc[i, "RT_copy"] = pd.NA

    return df


def parse_rt(rt):
    """
    Converts RT to float if it is possible, and the texts "Zbyt wczesnie" and "No answer" to NaN.
    All the numbers stay the same.
    
    :param rt: a single data point from the RT column. It could be a string with a number or text.

    Returns:
    RT as a number, or RT as NaN.
    """
    if isinstance(rt, (int, float)):
        return float(rt)

    if isinstance(rt, str):
        if rt.startswith("Zbyt wczesnie"):
            return np.nan
        if rt.startswith("Brak odpowiedzi"):
            return np.nan

        try:
            return float(rt)
        except ValueError:
            return np.nan

    return np.nan


# ----------------------------------------------------------
# 2. Function computing the metrics for a single PVT file
# ----------------------------------------------------------

def process_single_pvt_file(csv_path, participant_id, session):

    """
    Processes a single PVT file (_PVT.csv) from PREPROCESSED_DATA.

    :param csv_path: path to the file _PVT.csv present in the participant's directory.
    :participant_id: ID as PXXX, np. P001
    :session: session type – DAY or NIGHT

    Returns:
      - dataframe with a summary (PVT_summary)
      - dataframe with ISI (PVT_ISI)
    """
    filename = os.path.basename(csv_path).replace("_PVT-CT.csv", "")

    df = pd.read_csv(csv_path, sep=";", na_values=["NA"], keep_default_na=True, quotechar='"', engine='python')  # sep=";" tylko w przypadku P011_NIGHT

    df["RT_copy"] = df["RT"]

    df = clean_pvt_early_press_duplicates(df)

    # RT_num with the function converting to floats or NaNs
    df["RT_num"] = df["RT_copy"].apply(parse_rt)

    # dataframe with ISI which will be merged with other participants' ISIs to calculate the global average ISI
    df_isi = df[["Block_order_per_type", "Trial", "ISI"]].copy()

    summary_rows = []

    # ----------------------------------------------------------
    # 3. Block x Trial analysis
    # ----------------------------------------------------------

    for block in [1, 2]:
        df_block = df[df["Block_order_per_type"] == block]

        for trial in [1, 2, 3, 4, 5]:
            df_bt = df_block[df_block["Trial"] == trial]

            total_n = len(df_bt)

            # Answers 100-1000
            mask_100_1000 = df_bt["RT_num"].between(100, 1000, inclusive="both")
            rts_100_1000 = df_bt.loc[mask_100_1000, "RT_num"] # we use a mask to obtain only the numerical values

            # Answers 100-500
            mask_100_500 = df_bt["RT_num"].between(100, 500, inclusive="left") # here we are interested in a number of such events

            # lapses 500-1000
            mask_500_1000 = df_bt["RT_num"].between(500, 1000, inclusive="both") # here we are interested in a number of such events

            # false starts
            mask_false_numeric = df_bt["RT_num"] < 100 # here we count only those below 100 ms
            mask_false_text = df_bt["RT_copy"].astype(str).str.startswith("Zbyt wczesnie") # here we count false starts
            total_false = (mask_false_numeric | mask_false_text).sum() # there is a sum of both cases

            # omissions
            mask_omission_numeric = df_bt["RT_num"] > 1000 # here we count only those above 1000 ms (plus those exceptions with > 10000 ms)
            mask_omission_text = df_bt["RT_copy"].astype(str).str.startswith("Brak odpowiedzi") # only no answer is counted here
            total_omission = (mask_omission_numeric | mask_omission_text).sum() # a sum of both cases

            row = {
                "ID": participant_id,
                "Session": session,
                "Block_order_per_type": block,
                "Trial": trial,
                "Total_no_of_resp": total_n,
                "Total_resp100_1000ms": mask_100_1000.sum(),
                "Mean_RT_resp100_1000ms": rts_100_1000.mean(),
                "SD_RT_resp100_1000ms": rts_100_1000.std(),
                "Median_RT_resp100_1000ms": rts_100_1000.median(),
                "Total_resp100_500ms": mask_100_500.sum(),
                "Total_resp_lapses500_1000ms": mask_500_1000.sum(),
                "Relative_resp_lapses500_1000ms": mask_500_1000.sum() / total_n if total_n > 0 else np.nan,
                "Total_false_starts": total_false,
                "Relative_false_starts": total_false / total_n if total_n > 0 else np.nan,
                "Total_omissions": total_omission,
                "Relative_omissions": total_omission / total_n if total_n > 0 else np.nan,
            }

            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)

    # return both dataframes
    return df_summary, df_isi, filename, df


# ----------------------------------------------------------
# 4. Function to process all the participants
# ----------------------------------------------------------

def process_all_participants(base_path, participants, output_dir):
    """
    The goal is to go through all the directories, compute the metrics with PVT files, and then aggregate the results.

    :param base_path: → directory DATA_TO_PROCESS
    :param participants: → list ["P001_DAY", ...]
    :param output_dir: → directory Ready_for_analysis

    Creates:
      - PVT_summary_all.csv
      - PVT_ISI_all.csv
    """

    PVT_summary_all = []
    PVT_ISI_all = []

    for pid in participants:
        participant_path = os.path.join(base_path, pid, "/corrected_tags/PREPROCESSED_DATA")

        # find *_PVT.csv
        files = [f for f in os.listdir(participant_path) if f.endswith("_PVT-CT.csv")]
        if len(files) == 0 or len(files) > 1:
            print(f"[WARN] Brak pliku PVT dla {pid} lub jest ich za duzo")
            continue

        csv_path = os.path.join(participant_path, files[0])

        # process
        participant, session_day_night = pid.split("_")
        df_summary, df_isi, filename, df_after_correction = process_single_pvt_file(csv_path, participant, session_day_night)
        path_corrected_df = os.path.join(participant_path, f"{filename}_PVT-CT_corrected.csv")
        df_after_correction.to_csv(path_corrected_df, index=False, na_rep="NA")

        # save per participant
        out_path = os.path.join(participant_path, f"{filename}_PVT_summary.csv")
        df_summary.to_csv(out_path, index=False, na_rep="NA")

        # aggregate
        PVT_summary_all.append(df_summary)
        PVT_ISI_all.append(df_isi)

    # merge
    df_all_summary = pd.concat(PVT_summary_all, ignore_index=True) 
    df_all_isi = pd.concat(PVT_ISI_all, ignore_index=True)

    # export
    df_all_summary.to_csv(os.path.join(output_dir, "CT-PVT_summary.csv"), index=False, na_rep="NA")
    df_all_isi.to_csv(os.path.join(output_dir, "CT-PVT_ISI.csv"), index=False, na_rep="NA")

    return df_all_summary, df_all_isi


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


output_location = os.path.join(datapath, "Corrected_tags-Ready_for_analysis")

process_all_participants(datapath, participants_data, datapath)



