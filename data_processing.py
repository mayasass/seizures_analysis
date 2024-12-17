import pandas as pd
import numpy as np
import mne
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('QtAgg')
from datetime import datetime, timedelta
from pathlib import Path



def fix_electrode_names(raw):
    """""
    Rename old electrode names to new standard and keep only the 19 standard electrodes.

    Arguments:
    raw (mne.io.Raw): Raw EEG data

    Returns: 
    mne.io.Raw: Processed EEG data with correct electrode names
    """""
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
    """""
    Preprocess EEG data with standard pipeline (bandpass and avg reference).

    Args: raw (mne.io.Raw): Raw EEG data

    Returns: mne.io.Raw: Preprocessed EEG data
    """""
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
    """""
    Compute power spectrum and analyzed frequency bands for each channel.

    Args:
        raw_processed (mne.io.Raw): Preprocessed EEG data

    Returns:
        pd.DataFrame: Results containing power values for different frequency bands
    """""
    # Get channel names
    channels = raw_processed.ch_names

    # Initialize lists to store results
    results = []

    # Helper function to get power in specific frequency range
    def get_freq_power(psds, freqs, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.mean(psds[:, idx])

    # Calculate power spectrum for each channel
    for channel in channels:
        # Get the spectrum object
        spectrum = raw_processed.compute_psd(
            method='welch',
            picks=channel,
            fmin=0.5,
            fmax=40,
            n_fft=int(raw_processed.info['sfreq'] * 4),
            n_overlap=int(raw_processed.info['sfreq'] * 2)
        )

        # Get the data from spectrum
        psds = spectrum.get_data()
        freqs = spectrum.freqs

        # Calculate powers for different frequency bands
        delta_power_0p5_2 = get_freq_power(psds, freqs, 0.5, 2)
        delta_power_1_4 = get_freq_power(psds, freqs, 1, 4)
        total_power = get_freq_power(psds, freqs, 0.5, 40)

        # Store results
        results.append({
            'channel': channel,
            'delta_power_0p5_2': delta_power_0p5_2,
            'delta_power_1_4': delta_power_1_4,
            'total_power': total_power
        })

    # Create DataFrame from results
    df_results = pd.DataFrame(results)
    return df_results


def analyze_delta_power(raw):
    """""
    Main function to analyze delta power in EEG data.

    Args:
        raw (mne.io.Raw): Raw EEG data

    Returns:
        pd.DataFrame: Results of power analysis
    """""
    # 1. Preprocess the data
    raw_processed = preprocess_eeg(raw)

    # 2. Compute power spectrum and analyze
    df_results = compute_power_spectrum(raw_processed)

    # 3. Save results
    output_path = 'D:/seizures_analysis/excel_files'
    df_results.to_csv(output_path, index=False)
    print(f"Analysis complete. Results saved to {output_path}")

    return df_results

def seizure_num_to_raw_data(pat_num, seizure_index,seizures_list_table):
    """""
    Finds the compatible recording for seizure.
    
    Args:
        Patient id and its seizures list table, desired seizure
    
    Returns:
        Raw EEG data of the seizure
    """""
    # Base path to data
    data_path = Path("E:/Ben Gurion University Of Negev Dropbox/CPL lab members/epilepsy_data/Epilepsiea")

    # Extract the recording number for the specific seizure
    seizure_rec_num = seizures_list_table.loc[
        seizures_list_table['seizure_num'] == seizure_index,
        'file_seizure_ind'].iloc[0]

    # Load the recordings data table
    seizures_data_path = data_path / "tables" / f"pat_{pat_num}_surf30_file_list"
    seizures_data_table = pd.read_excel(seizures_data_path)

    #*********** NOT SURE ****************
    # Get the recording path
    seizure_recording_path = seizures_data_table.loc[seizure_rec_num, 'file_path']

    # Read the raw data
    raw = mne.io.read_raw_nicolet(seizure_recording_path)

    return raw

def copy_and_crop(raw, seizure_ind, seizures_list_table, sec_before=60, sec_after=60):
    """""
    Cut tha data +- desired sec to extract the seizure from in the recording

    Args:
        Raw data of recording with a seizure in it, list of seizures' info, seizure index 

    Returns:
        Raw EEG data crop of the seizure's time +- sec variant (default 60)
    """""
    # Create a copy to avoid modifying the original
    raw = raw.copy()

    # Get recording information in seconds
    recording_duration = raw.times[-1]  # or: len(raw.times) / raw.info['sfreq']
    recording_start = np.datetime64(raw.info['meas_date'])

    # Get the start & end time of desired seizure
    seizure_start = pd.to_datetime(seizures_list_table.loc[seizure_ind, 'onset'], format='%d/%m/%Y %H:%M').to_numpy()
    seizure_end = pd.to_datetime(seizures_list_table.loc[seizure_ind, 'offset'], format='%d/%m/%Y %H:%M').to_numpy()

    # Subtraction of start and end from recording start to get duration in sec
    seizure_start_from_tmin = (seizure_start - recording_start) / np.timedelta64(1, 's')
    seizure_end_from_tmin = (seizure_end - recording_start) / np.timedelta64(1, 's')

    # Calculate crop times in sec (5 minutes before and after (60s))
    crop_start = seizure_start_from_tmin - sec_before
    crop_end = seizure_end_from_tmin + sec_after
    print(crop_start)
    print(crop_end)

    # Make sure we're within recording borders
    if crop_start < 0:
        crop_start = 0
        print('Warning: crop_start adjusted to recording start (0)')

    if crop_end > recording_duration:
        crop_end = recording_duration
        print(f'Warning: crop_end adjusted to recording end ({recording_duration}s)')

    # Cropping the data
    raw_cropped = raw.copy().crop(tmin=crop_start, tmax=crop_end)
    # Maybe: raising assert for data exception
    return raw_cropped

def get_seizures_list(pat_num):
    # Base path to data
    data_path = Path("E:/Ben Gurion University Of Negev Dropbox/CPL lab members/epilepsy_data/Epilepsiea")

    # Load seizures table
    seizures_list_path = data_path / "tables" / "seizure_tables" / f"{pat_num}_surf30_seizures"
    seizures_list_table = pd.read_excel(seizures_list_path)

    return seizures_list_table

def main_analysis(pat_num, seizure_index):
    def main(pat_num, seizure_index, seizures_list_table):
        # Step 1: Find the raw EEG data for the given patient and seizure index
        raw_data = seizure_num_to_raw_data(pat_num, seizure_index, seizures_list_table)

        # Step 2: Crop the raw data around the seizure, with 60 seconds before and after the event
        raw_cropped = copy_and_crop(raw_data, seizure_index, seizures_list_table)

        # Step 3: Analyze the cropped data for delta power
        analysis_results = analyze_delta_power(raw_cropped)

        # Step 4: Return the analysis results (or save them, depending on requirements)
        return analysis_results

    # Example usage
    if __name__ == "__main__":
        pat_num_list = []
        for pat in pat_num_list:
            seizures_list_table = get_seizures_list(pat)
            seizures_list = seizures_list_table['seizure_num'].tolist()
            for seizure in seizures_list:
                results = main(pat, seizure,seizures_list_table)
                print(results)

    # WHAT'S LEFT: ADD it ALL to a single table