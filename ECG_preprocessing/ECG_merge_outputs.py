import os
import pandas as pd

'''
The goal of this script is to merge all the ECG results to a single file.
'''


os.chdir('.../DATA_TO_PROCESS/')
base_dir = os.getcwd()
all_data = []

for folder in sorted(os.listdir(base_dir)):
    if not folder.startswith("P0"):
        continue

    person_path = os.path.join(base_dir, folder)
    if not os.path.isdir(person_path):
        continue

    corrected_tags_path = os.path.join(person_path, "corrected_tags")
    if not os.path.isdir(corrected_tags_path):
        continue

    # original name
    original_results_path = os.path.join(corrected_tags_path, "Results_ECG_GUI")

    # new name
    new_results_folder_name = f"{folder}_Results_ECG_GUI"
    new_results_path = os.path.join(corrected_tags_path, new_results_folder_name)

    # change folder name
    if os.path.isdir(original_results_path):
        if not os.path.exists(new_results_path):
            os.rename(original_results_path, new_results_path)
            print(f"Zmieniono nazwę: {original_results_path} -> {new_results_path}")
        else:
            print(f"Folder już istnieje: {new_results_path}")

    if not os.path.isdir(new_results_path):
        continue

    # searching for CSV file
    for file in os.listdir(new_results_path):
        if file.endswith("_subset_features.csv"):
            file_path = os.path.join(new_results_path, file)

            try:
                df = pd.read_csv(file_path)
                df["subject_id"] = folder
                all_data.append(df)

                print(f"Dodano: {file_path}")

            except Exception as e:
                print(f"Błąd przy pliku {file_path}: {e}")

# merging
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    output_file = os.path.join(base_dir, "merged_output.csv")
    final_df.to_csv(output_file, index=False)

    print(f"\nZapisano wynik do: {output_file}")
else:
    print("Nie znaleziono żadnych danych.")
