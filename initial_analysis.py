import pandas as pd

# data reading
data = pd.read_excel("seizures_data.xlsx", header=1)

# columns printing
print(data.columns)

# amount of people in data set
patients = data['patient_id'].nunique()
print("number of patients:", patients)

# mapping seizures by Sleep Stages
sleep_stages = data['vigilance'].unique()
print("sleep stages found:", sleep_stages)

# extracting SG seizures
# sub-data structure
generalized_seizures = data[data['classification'] == 'SG']

# calculation of SG seizures percentage from seizures
total_seizures = len(data)
total_SG_count = len(generalized_seizures)
percentage_generalized = total_SG_count / total_seizures * 100

if total_SG_count == 0:
    raise ValueError("Error: No SG seizures found.")

print(f"From {total_seizures} seizures, {total_SG_count} are secondarily generalized")
print(f"Which make up {percentage_generalized:.2f}%")
print()

# Define stages to iterate over
all_stages = ['awake', 'sleep stage I', 'sleep stage II', 'sleep stage III/IV', 'REM']
# We'll focus on the sleep stages only for the summary
sleep_stages = ['sleep stage I', 'sleep stage II', 'sleep stage III/IV', 'REM']

# Lists to store data for each stage
data_for_summary = []

# Loop over each sleep stage
for stage in all_stages:
    # total seizures in stage count
    stage_data = data[data['vigilance'] == stage]
    total_seizures_in_stage = stage_data.shape[0]

    # SG seizures count
    SG_in_stage = stage_data[stage_data['classification'] == 'SG'].shape[0]

    # Percentage of SG seizures in this stage
    if total_seizures_in_stage > 0:
        SG_percentage_in_stage = SG_in_stage / total_seizures_in_stage * 100
    else:
        SG_percentage_in_stage = 0  # no seizures in this stage

    # Percentage of SG seizures relative to total SG seizures
    if total_SG_count > 0:
        SG_percentage_from_total = SG_in_stage / total_SG_count * 100
    else:
        SG_percentage_from_total = 0  # handle case of no SG seizures

    # Append data to the summary list
    data_for_summary.append({
        'Sleep Stage': stage,
        'Total Seizures in Stage': total_seizures_in_stage,
        'SG Seizures in Stage': SG_in_stage,
        'Percentage from Seizures in Stage': SG_percentage_in_stage,
        'Percentage from Total SG Seizures': SG_percentage_from_total
    })

    # Print results for each stage
    print(f"Total seizures in {stage}: {total_seizures_in_stage}")
    print(f"Secondarily generalized seizures in {stage}: {SG_in_stage}")
    print(f"Percentage from total seizures in {stage}: {SG_percentage_in_stage:.2f}%")
    print(f"Percentage from total secondarily generalized seizures: {SG_percentage_from_total:.2f}%")
    print()

# Create a DataFrame from the summary list
summary_df = pd.DataFrame(data_for_summary)

# Print the summary DataFrame
print(summary_df)

# Save to Excel and CSV files
summary_df.to_excel('SG_isolated_summary_full.xlsx', index=False)
summary_df.to_csv('SG_isolated_summary_full.csv_tests', index=False)
