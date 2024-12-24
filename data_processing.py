import os

import pandas as pd
import numpy as np
import mne
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('QtAgg')
from datetime import datetime, timedelta
from pathlib import Path

DATA_PATH = Path("E:/Ben Gurion University Of Negev Dropbox/CPL lab members/epilepsy_data/Epilepsiea")

def fix_electrode_names(raw):
    """
    Rename old electrode names to new standard and keep only the 19 standard electrodes.

    Arguments:
    raw (mne.io.Raw): Raw EEG data

    Returns: 
    mne.io.Raw: Processed EEG data with correct electrode names
    """
    # Create a copy to avoid modifying the original
    raw_processed = raw.copy()

    # Define the standard 19 electrodes template
    standard_electrodes = {'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                           'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                           'FZ', 'CZ', 'PZ'}

    # Define the channel name mapping for old to new names
    channel_mapping = {
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8'
    }

    # Check and rename channels if necessary
    current_channels = raw_processed.ch_names
    channels_to_rename = {}

    for old_name, new_name in channel_mapping.items():
        if old_name in current_channels:
            channels_to_rename[old_name] = new_name
            print(f"Renaming channel {old_name} to {new_name}")

    if channels_to_rename:
        raw_processed.rename_channels(channels_to_rename)

    # Get current channels after renaming
    current_channels = raw_processed.ch_names

    # Find channels to drop (those not in standard_electrodes)
    channels_to_drop = [ch for ch in current_channels if ch not in standard_electrodes]

    if channels_to_drop:
        print(f"Dropping non-standard channels: {channels_to_drop}")
        raw_processed.drop_channels(channels_to_drop)

    # Verify final channel set
    final_channels = raw_processed.ch_names
    print(f"Final channels ({len(final_channels)}): {final_channels}")

    return raw_processed


def preprocess_eeg(raw):
    """
    Preprocess EEG data with standard pipeline (bandpass and avg reference).

    Args: raw (mne.io.Raw): Raw EEG data

    Returns: mne.io.Raw: Preprocessed EEG data
    """
    # Create a copy to avoid modifying the original
    raw_processed = raw.copy()

    # Fix electrode names and keep only standard ones
    raw_processed = fix_electrode_names(raw_processed)

    # Apply bandpass filter (0.5-40 Hz)
    raw_processed.filter(l_freq=0.5, h_freq=40)
    print("Applied bandpass filter (0.5-40 Hz)")

    # Set average reference
    raw_processed.set_eeg_reference(ref_channels='average')
    print("Applied average reference")

    return raw_processed


def compute_power_spectrum(raw_processed):
    """
    Compute normalized power spectrum for multiple frequency bands for each channel.
    Returns dictionary with channel-specific relative power values.
    """
    # Define frequency bands and their names (keeping your original bands)
    freq_bands = {
        'low_delta': (0.5, 2),
        'high_delta': (1, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'low_sigma': (11, 13),
        'high_sigma': (13, 16),
        'low_beta': (16, 20),
        'mid_beta': (20, 25),
        'high_beta': (25, 30),
        'low_gamma': (30, 35),
        'high_gamma': (35, 40)
    }

    channels = raw_processed.ch_names
    power_results = {f"{band}_power": 0 for band in freq_bands.keys()}

    def get_freq_power(psds, freqs, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.mean(psds[:, idx])

    # Initialize total channel power
    channel_count = len(channels)

    for channel in channels:
        # Calculate PSD using Welch's method
        spectrum = raw_processed.compute_psd(
            method='welch',
            picks=channel,
            fmin=0.5,
            fmax=40,
            n_fft=int(raw_processed.info['sfreq'] * 4),
            n_overlap=int(raw_processed.info['sfreq'] * 2)
        )

        psds = spectrum.get_data()
        freqs = spectrum.freqs

        # Calculate total power across entire frequency range (0.5-40 Hz)
        total_power = get_freq_power(psds, freqs, 0.5, 40)

        # Calculate normalized power for each band
        for band_name, (fmin, fmax) in freq_bands.items():
            band_power = get_freq_power(psds, freqs, fmin, fmax)
            # Normalize by total power and accumulate for averaging
            old_value = power_results[f"{band_name}_power"]
            normalized_power = band_power / total_power
            power_results[f"{band_name}_power"] = old_value + normalized_power

    # Average across channels
    for key in power_results:
        power_results[key] /= channel_count

    # Add total absolute power for reference
    power_results['total_power'] = total_power / channel_count

    return power_results


def analyze_spectral_power(raw, pat_num, seizure_info):
    """
    Analyze spectral power across all frequency bands and return a single row of results.

    Args:
        raw (mne.io.Raw): Raw EEG data
        pat_num (str): Patient number
        seizure_info (pd.Series): Row from seizures table with seizure metadata

    Returns:
        dict: Single row of results including patient info and power analysis
    """
    # 1. Preprocess the data
    raw_processed = preprocess_eeg(raw)

    # 2. Compute power spectrum for all frequency bands
    power_results = compute_power_spectrum(raw_processed)

    # 3. Create result dictionary combining seizure info and power analysis
    result = {
        'pat_num': pat_num,
        'seizure_num': seizure_info['seizure_num'],
        'classif.': seizure_info['classif.'],
        'onset': seizure_info['onset'],
        'offset': seizure_info['offset'],
        'vigilance': seizure_info['vigilance'],
        'origin': seizure_info['origin'],
        'file_seizure_ind': seizure_info['file_seizure_ind']
    }

    # Add power results
    result.update(power_results)

    return result


def seizure_num_to_raw_data(pat_num, seizure_index, seizures_list_table):
    """
    Finds the compatible recording for seizure.

    Args:
        Patient id and its seizures list table, desired seizure

    Returns:
        Raw EEG data of the seizure
     """

    # Base path to data
    data_path = Path("E:/Ben Gurion University Of Negev Dropbox/CPL lab members/epilepsy_data/Epilepsiea")

    # Extract the recording number for the specific seizure
    seizure_rec_num = seizures_list_table.loc[
        seizures_list_table['seizure_num'] == seizure_index,
        'file_seizure_ind'].iloc[0]

    # Load the recordings data table
    seizures_data_path = data_path / "tables" / "pat_file_tables" / f"pat_{pat_num}_surf30_file_list.csv"
    seizures_data_table = pd.read_csv(seizures_data_path)

    # Get the recording path
    seizure_recording_path = seizures_data_table.loc[seizure_rec_num, 'file_path']

    # Get the recording path as string and replace 'head' with 'data'
    file_path = seizures_data_table.loc[seizure_rec_num, 'file_path']
    file_path = file_path.replace('head', 'data')  # String replacement

    # Ensure the path ends with .data
    if not file_path.endswith('.data'):
        file_path += '.data'

    # Find the position of the substring "raw_data"
    substring = "raw_data"
    index = file_path.find(substring)
    # If the substring is found, cut the path from "raw_data" onwards
    if index != -1:
        subpath = file_path[index:]
    else:
        print("Substring not found in path")
        return None

    seizure_recording_path = data_path / subpath

    # Read the raw data
    raw = mne.io.read_raw_nicolet(seizure_recording_path, ch_type='eeg', preload=True)

    return raw
def copy_and_crop(raw, seizure_ind, seizures_list_table, sec_before=60, sec_after=60):
    """
    Cut the data +- desired sec to extract the seizure from in the recording
    """
    # Create a copy to avoid modifying the original
    raw = raw.copy()

    print("Debug step 1: Getting table index")
    # Get the row index in the table for this seizure
    matching_rows = seizures_list_table[seizures_list_table['seizure_num'] == seizure_ind]
    if matching_rows.empty:
        print(f"No matching seizure found for index {seizure_ind}")
        print("Available seizure numbers:", seizures_list_table['seizure_num'].tolist())
        raise ValueError(f"No seizure found with index {seizure_ind}")

    table_index = matching_rows.index[0]
    print(f"Found table index: {table_index}")

    print("Debug step 2: Getting recording info")
    # Get recording information in seconds
    recording_duration = raw.times[-1]
    recording_start = np.datetime64(raw.info['meas_date'])
    print(f"Recording start time: {recording_start}")

    # Get onset/offset times and print them for debugging
    print("Debug step 3: Getting onset/offset times")
    onset_str = seizures_list_table.loc[table_index, 'onset'].strip()
    offset_str = seizures_list_table.loc[table_index, 'offset'].strip()
    print(f"Raw onset time string: '{onset_str}'")
    print(f"Raw offset time string: '{offset_str}'")

    try:
        print("Debug step 4: Attempting to parse dates")
        # Using the exact format that matches your data
        seizure_start = pd.to_datetime(onset_str, format='%Y-%m-%d %H:%M:%S')
        print(f"Successfully parsed onset time: {seizure_start}")

        seizure_end = pd.to_datetime(offset_str, format='%Y-%m-%d %H:%M:%S')
        print(f"Successfully parsed offset time: {seizure_end}")

        # Convert to numpy datetime64 with microsecond precision
        seizure_start = np.datetime64(seizure_start)
        seizure_end = np.datetime64(seizure_end)
        recording_start = recording_start.astype('datetime64[us]')
        seizure_start = seizure_start.astype('datetime64[us]')
        seizure_end = seizure_end.astype('datetime64[us]')

        # Calculate times in seconds
        seizure_start_from_tmin = (seizure_start - recording_start) / np.timedelta64(1, 's')
        seizure_end_from_tmin = (seizure_end - recording_start) / np.timedelta64(1, 's')

        print(f"Seizure starts at {seizure_start_from_tmin} seconds from recording start")
        print(f"Seizure ends at {seizure_end_from_tmin} seconds from recording start")

        # Calculate crop times
        crop_start = seizure_start_from_tmin - sec_before
        crop_end = seizure_end_from_tmin + sec_after

        # Make sure we're within recording borders
        if crop_start < 0:
            print('Warning: crop_start adjusted to recording start (0)')
            crop_start = 0

        if crop_end > recording_duration:
            print(f'Warning: crop_end adjusted to recording end ({recording_duration}s)')
            crop_end = recording_duration

        # Crop the data
        raw_cropped = raw.copy().crop(tmin=crop_start, tmax=crop_end)
        return raw_cropped

    except Exception as e:
        print(f"Error occurred while processing times: {str(e)}")
        print(f"Full row data: {seizures_list_table.loc[table_index]}")
        print(f"Data types: {seizures_list_table.loc[table_index].dtypes}")
        raise

def get_seizures_list(pat_num, data_path=DATA_PATH):
    # Load seizures table
    seizures_list_path = data_path / "tables" / "seizure_tables" / f"{pat_num}_surf30_seizures.csv"
    seizures_list_table = pd.read_csv(seizures_list_path)

    return seizures_list_table


def main_analysis(pat_num, seizure_index, seizures_list_table, data_path=DATA_PATH):
    """
    Process a single seizure and return its analysis results.
    """
    # Get seizure info
    seizure_info = seizures_list_table[seizures_list_table['seizure_num'] == seizure_index].iloc[0]

    # Find and process the raw EEG data
    raw_data = seizure_num_to_raw_data(pat_num, seizure_index, seizures_list_table)
    raw_cropped = copy_and_crop(raw_data, seizure_index, seizures_list_table)

    # Analyze and return single row of results
    return analyze_spectral_power(raw_cropped, pat_num, seizure_info)


if __name__ == "__main__":
    # Define parameters
    surf = "surf30"
    surf_suffix_to_remove = '02'

    # Get list of patient directories
    pat_list = list(filter(lambda x: x.startswith("pat_"), os.listdir(DATA_PATH / "raw_data" / surf)))

    # Process patient numbers
    pat_num_list = [pat.replace('pat_', '') for pat in pat_list]
    pat_num_list = [num[:-2] if num.endswith(surf_suffix_to_remove) else num for num in pat_num_list]

    # Initialize list to store all results
    all_results = []

    # Process each patient and seizure
    for pat in pat_num_list:
        print(f"Processing patient {pat}")

        # Get seizures list for current patient
        seizures_list_table = get_seizures_list(pat)
        seizures_list = seizures_list_table['seizure_num'].tolist()

        # Process each seizure for current patient
        for seizure in seizures_list:
            print(f"Processing seizure {seizure}")
            try:
                # Run analysis for current seizure
                result = main_analysis(pat, seizure, seizures_list_table)
                all_results.append(result)
                print(f"Completed analysis for patient {pat}, seizure {seizure}")
            except Exception as e:
                print(f"Error processing patient {pat}, seizure {seizure}: {str(e)}")
                continue

    # Create final DataFrame and save results
    try:
        final_df = pd.DataFrame(all_results)
        output_path = Path('C:/Users/cognitive/Desktop/all_spectrum_seizures_analysis.csv')
        final_df.to_csv(output_path, index=False)
        print(f"Analysis complete. All results saved to {output_path}")
    except Exception as e:
        print(f"Error saving final results: {str(e)}")