# The script is needed to extract the information about the ISI from the datasets and store it in a single CSV in order
# to calculate mean + SD of the ISI. The same data from the datasets with issues will be added manually to the major CSV file.


import os
import pandas as pd

os.chdir(".../DATA_TO_PROCESS/")
base_dir = os.getcwd()

all_data = []

for folder in sorted(os.listdir(base_dir)):
    if folder.startswith("P0") and os.path.isdir(os.path.join(base_dir, folder)):

        target_path = os.path.join(
            base_dir,
            folder,
            "corrected_tags",
            "PREPROCESSED_DATA"
        )

        if not os.path.exists(target_path):
            print(f"Brak ścieżki: {target_path}")
            continue

        for file in os.listdir(target_path):
            if file.endswith("_PVT-CT_corrected.csv"):

                file_path = os.path.join(target_path, file)

                try:
                    df = pd.read_csv(file_path)

                    if "ISI" not in df.columns:
                        print(f"Brak kolumny ISI w pliku: {file}")
                        continue

                    temp_df = pd.DataFrame({
                        "Nazwa": [file] * len(df),
                        "ISI": df["ISI"]
                    })

                    all_data.append(temp_df)

                except Exception as e:
                    print(f"Błąd przy pliku {file}: {e}")

if all_data:
    final_df = pd.concat(all_data, ignore_index=True)

    output_path = os.path.join(base_dir, "PVT_all_ISI_data.csv")
    final_df.to_csv(output_path, index=False)

    print(f"Zapisano plik: {output_path}")
else:
    print("Brak danych do zapisania.")
