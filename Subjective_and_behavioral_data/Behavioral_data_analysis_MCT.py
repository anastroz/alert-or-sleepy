#####
# MCT
'''

In each folder [P001_DAY etc.] there is an internal folder PREPROCESSED_DATA, and there should be a file ending with "_MCT.csv", which will be read.
In each of such files there are following columns:
- "Block" -- block order (earlier = 1, later = 2),
- "BlockTrial" -- order of each task trial in a given block (first = 1, last = 5),
- "Position" -- indicates the time of an event (can be NA if there was a procedure clock reset (only in datasets with issues)).
- "Trial_type" -- whether the event was a target (TGT) or non-target (NTGT).
- "RT" -- contains the reaction times in SECONDS – the following options are available:
--- if the button was pressed for the NTGT, there will be a message starting with "False positive". 
--- if the button was not pressed for the NTGT, there will be NA value.
--- if the button was pressed for the TGT, there will be a number -- RT in seconds.
--- if the button was not pressed for the TGT, there will be NA value.
- "Accuracy" -- equals to "0" if the button was pressed to the NTGT, and to "1" if pressed to the TGT. Not used here.


The following scripts enables reading the CSV data, correcting the data (in case of the button being pressed for too long, so the errors won't be summed twice), and calculating the metrics related to task performance. 

'''

import os
import pandas as pd
import numpy as np


def load_mct_csv(csv_path):
    try:
        return pd.read_csv(
            csv_path,
            sep=",",
            na_values=["NA"],
            keep_default_na=True,
            engine="python"
        )
    except pd.errors.ParserError:
        return pd.read_csv(
            csv_path,
            sep=";",
            na_values=["NA"],
            keep_default_na=True,
            engine="python"
        )


def clean_consecutive_false_positives(df):

    """
    This code first identifies two adjacent RTs in RT_copy, and if both are "False positive", it changes the current one to NA,
    while leaving the previous one as it was. This is the result of holding down the button until the next event. 

    :param df: dataframe with the column "RT_copy" already added

    Returns:
    df -- dataframe with corrections

    """

    df = df.copy()

    for i in range(1, len(df)):
        curr_rt = df.loc[i, "RT_copy"]
        prev_rt = df.loc[i - 1, "RT_copy"]

        if (
            isinstance(curr_rt, str)
            and isinstance(prev_rt, str)
            and curr_rt.startswith("False positive")
            and prev_rt.startswith("False positive")
        ):
            df.loc[i, "RT_copy"] = pd.NA

    return df


def relabel_false_positive_after_tgt(df):

    """
    This function takes a dataframe as input, copies it, and checks the copy to see whether:
    - in the current row there is an NTGT event with a "False positive" response, and in the previous row there was a TGT
    - if so, there are two possibilities:
    ---- if someone answered the TGT correctly in the previous row, then in the current row we change the value from
         "False positive" to NA, because it was not a FP, but the result of holding down the button
    ---- if the person did not respond correctly to TGT in the previous row (it was NA as RT), then in the current row
         we change it to "Late response" to distinguish this from a false positive, as it is most likely the result of a
         delayed response to the stimulus (it will still be counted as an omission, but it certainly should not end up in the
         "False positive" category).

    :param df: dataframe to be checked if there are such doubled events

    Returns:
    df -- dataframe with corrections
    """

    df = df.copy()

    def can_be_parsed_to_float(s):
        try:
            float(s)
            return True
        except ValueError:
            return False

    for i in range(1, len(df)):
        prev = df.loc[i - 1]
        curr = df.loc[i]

        curr_rt = curr["RT_copy"]

        # case in which we are currently in NTGT with "False positive", but the previous stimulus was TGT
        if (
            curr["Trial_type"] == "NTGT"
            and isinstance(curr_rt, str)
            and curr_rt.startswith("False positive")
            and prev["Trial_type"] == "TGT"
        ):
            # TGT answered (with RT)
            if pd.notna(prev["RT"]) and can_be_parsed_to_float(prev["RT"]):
                df.loc[i, "RT_copy"] = pd.NA

            # TGT not answered (TGT with NA, current NTGT with an attempt to answer -- it's a late response
            elif pd.isna(prev["RT"]):
                df.loc[i, "RT_copy"] = "Late response"

    return df


def parse_rt(df):

    """
    This function adds a column named RT_num, which is a copy of RT_copy with the following modifications: NA values
    remain as NA; entries marked as False Positive or Late Response are converted to NA; numerical values are interpreted
    as numbers and returned as milliseconds.

    :param df: dataframe

    Returns:
    df -- new dataframe with "RT_num" column added
    """
    def _parse(val):
        if pd.isna(val):
            return pd.NA

        if isinstance(val, str):
            if val.startswith("False positive") or val == "Late response":
                return pd.NA

        try:
            return float(val) * 1000  # s → ms
        except (ValueError, TypeError):  # in case of a non-numerical value, not-NA and not one of the strings above (there shouldn't be any cases like this)
            return pd.NA

    df["RT_num"] = df["RT_copy"].apply(_parse)
    return df


def compute_mct_trial_metrics(df_bt):
    """
    Computes all the metrics for each Block × BlockTrial
    """

    # here we collect information on all hits -- whether for NTGT ("False Positive" or "Late response"), or for TGT (a number) in the RT_copy column

    total_no_of_resp = df_bt["RT_copy"].notna().sum()  # summed up to obtain all the hits within the trial

    # only correct answers (range 100-1000 ms (inclusive) and TGT)

    tgt_100_1000 = (
            (df_bt["Trial_type"] == "TGT") &
            df_bt["RT_num"].between(100, 1000, inclusive="both")
    )

    # faster responses (100-500 ms (closed on the left) and TGT)

    tgt_100_500 = (
            (df_bt["Trial_type"] == "TGT") &
            df_bt["RT_num"].between(100, 500, inclusive="left")
    )

    # slower responses/lapses (500-1000 ms (inclusive) and TGT)

    tgt_500_1000 = (
            (df_bt["Trial_type"] == "TGT") &
            df_bt["RT_num"].between(500, 1000, inclusive="both")
    )

    # incorrect responses – false starts and answers to NTGT (RT starts with "False positive")

    false_starts = (
            df_bt["RT_copy"].astype(str).str.startswith("False positive") |
            (df_bt["RT_num"] < 100)
    )

    # omissions – answers above 1000 ms and lacking TGT answers 

    omissions = (
        (df_bt["RT_num"] > 1000).sum() +
        (df_bt["RT_copy"].isna() & (df_bt["Trial_type"] == "TGT")).sum()
    )

    late_responses = (df_bt["RT_copy"].astype(str).str.startswith("Late response")).sum()
    
    rt_values = df_bt.loc[tgt_100_1000, "RT_num"]

    return {
        "Total_no_of_resp": total_no_of_resp,
        "Total_respTGT_100_1000ms": tgt_100_1000.sum(),
        "Mean_RT_respTGT_100_1000ms": rt_values.mean(),
        "SD_RT_respTGT_100_1000ms": rt_values.std(),
        "Median_RT_respTGT_100_1000ms": rt_values.median(),
        "Total_respTGT_100_500ms": tgt_100_500.sum(),
        "Total_resp_lapsesTGT_500_1000ms": tgt_500_1000.sum(),
        "Relative_resp_lapsesTGT_500_1000ms": (
            tgt_500_1000.sum() / total_no_of_resp
            if total_no_of_resp > 0 else np.nan
        ),
        "Total_false_starts": false_starts.sum(),
        "Relative_false_starts": (
            false_starts.sum() / total_no_of_resp
            if total_no_of_resp > 0 else np.nan
        ),
        "Total_omissions": omissions,
        "Relative_omissions": (
            omissions / total_no_of_resp
            if total_no_of_resp > 0 else np.nan
        ),
        "Late_responses": late_responses,
        "No_of_tgt": (df_bt["Trial_type"] == "TGT").sum(),
        "Total_no_of_events": len(df_bt),
        "Ratio_tgt_all": (df_bt["Trial_type"] == "TGT").sum() / len(df_bt),
    }


def process_single_mct_file(participant_dir, participant_id, session):
    """
    Processes a single MCT file of a participant
    """

    preprocessed_path = participant_dir

    mct_files = [f for f in os.listdir(preprocessed_path) if f.endswith("_MCT-CT.csv")]
    if len(mct_files) != 1:
        print(f"[WARN] Brak pliku MCT dla {participant_id} lub jest ich za duzo")
        return None

    filename = mct_files[0].replace("_MCT-CT.csv", "")
    filepath = os.path.join(preprocessed_path, mct_files[0])

    df = load_mct_csv(filepath)  # reading the original dataset

    df["RT_copy"] = df["RT"]  # duplication of RT column to the RT_copy to run modifications on it
    df = clean_consecutive_false_positives(df)  # this goes first, as it removes false positive duplicates
    df = relabel_false_positive_after_tgt(df)  # this goes second, as it relabels the special cases of TGT and NTGT answers

    df = parse_rt(df)  # parsing of RTs while having modifications stored in RT_copy

    rows = []

    for block in [1, 2]:
        for trial in [1, 2, 3, 4, 5]:

            df_bt = df[
                (df["Block_order_per_type"] == block) &
                (df["Trial"] == trial)
            ]

            metrics = compute_mct_trial_metrics(df_bt)

            row = {
                "ID": participant_id,
                "Session": session,
                "Block_order_per_type": block,
                "Trial": trial,
                **metrics
            }

            rows.append(row)

    summary_df = pd.DataFrame(rows)

    output_path = os.path.join(
        preprocessed_path,
        f"{filename}_MCT_summary-CT.csv"
    )
    summary_df.to_csv(output_path, index=False)

    return summary_df


def run_mct_analysis(participants, base_path):
    """
    Batch data processing
    """

    all_summaries = []

    for participant in participants:
        participant_dir = os.path.join(base_path, participant, "/corrected_tags/PREPROCESSED_DATA")

        participant_name, session = participant.split("_")

        summary_df = process_single_mct_file(
            participant_dir,
            participant_id=participant_name,
            session=session
        )

        if summary_df is not None:
            all_summaries.append(summary_df)

    df_all = pd.concat(all_summaries, ignore_index=True)

    output_path = os.path.join(
        base_path,
        "CT-MCT_summary_all.csv"
    )
    #        "Corrected_tags-Ready_for_analysis",

    df_all.to_csv(output_path, index=False)

    return df_all


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


run_mct_analysis(participants_data, datapath)
