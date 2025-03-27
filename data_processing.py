import os
import pandas as pd
import numpy as np
import mne
import matplotlib as mpl
import sys
# from matplotlib import pyplot as plt

mpl.use('QtAgg')
from datetime import datetime, timedelta
from pathlib import Path

DATA_PATH = Path("E:/Ben Gurion University Of Negev Dropbox/CPL lab members/epilepsy_data/Epilepsiea")
adjusted_crop_count = 0

def get_pat_info(pat_num, surf):
    """Get original patient number with suffix"""
    suffix_mapping = {
        "surf30": "02",
        "surfCO": "00",
        "surfPA": "03",
        "CO": "00"
    }
    base_pat_num = pat_num[:-2] if pat_num.endswith(suffix_mapping.get(surf, '')) else pat_num
    full_pat_num = f"{base_pat_num}{suffix_mapping.get(surf, '')}"
    return base_pat_num, full_pat_num
def get_vigilance_code(vigilance):
    """Convert vigilance state to numeric code."""
    vigilance_mapping = {
        'awake': 1,
        'sleep stage 1': 2,
        'sleep stage 2': 3,
        'sleep stage 3': 4,
        'sleep stage 4': 5,
        'REM': 6,
        'unknown': 0
    }
    return vigilance_mapping.get(vigilance.lower(), 0)

def get_lobe_code(origin):
    """Convert lobe origin to numeric code."""
    lobe_mapping = {
        'temporal': 1,
        'frontal': 2,
        'parietal': 3,
        'occipital': 4,
        'central': 5,
        'unknown': 0
    }
    return lobe_mapping.get(origin.lower(), 0)

def fix_electrode_names(raw):
    """
    Rename old electrode names to new standard and keep only the 19 standard electrodes.
    """
    raw_processed = raw.copy()

    standard_electrodes = {'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                           'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                           'FZ', 'CZ', 'PZ'}

    # Normalize case for channel names to uppercase
    new_ch_names = {ch: ch.upper() for ch in raw_processed.ch_names}
    raw_processed.rename_channels(new_ch_names)

    channel_mapping = {
        'T3': 'T7',
        'T4': 'T8',
        'T5': 'P7',
        'T6': 'P8'
    }

    current_channels = raw_processed.ch_names
    channels_to_rename = {}

    for old_name, new_name in channel_mapping.items():
        if old_name in current_channels:
            channels_to_rename[old_name] = new_name
            print(f"Renaming channel {old_name} to {new_name}")

    if channels_to_rename:
        raw_processed.rename_channels(channels_to_rename)

    current_channels = raw_processed.ch_names
    channels_to_drop = [ch for ch in current_channels if ch not in standard_electrodes]

    if channels_to_drop:
        print(f"Dropping non-standard channels: {channels_to_drop}")
        raw_processed.drop_channels(channels_to_drop)

    final_channels = raw_processed.ch_names
    print(f"Final channels ({len(final_channels)}): {final_channels}")

    return raw_processed


def preprocess_eeg(raw):
    """
    Preprocess EEG data with standard pipeline (bandpass and avg reference).
    """
    raw_processed = raw.copy()
    raw_processed = fix_electrode_names(raw_processed)
    raw_processed.filter(l_freq=0.5, h_freq=40)
    print("Applied bandpass filter (0.5-40 Hz)")
    raw_processed.set_eeg_reference(ref_channels='average')
    print("Applied average reference")
    raw_processed._data *= 1e3
    print("Converted to milliVolts")
    return raw_processed


def compute_power_spectrum(raw_processed):
    """
    Compute normalized power spectrum for each frequency band for each channel.
    Returns a dictionary with channel_frequency band as keys.
    """
    freq_bands = {
        'delta': (0.5, 4),
        'sigma': (12, 16),
        'beta': (16, 30),
        'gamma': (30, 40)
    }

    channels = raw_processed.ch_names
    power_results = {}

    def get_freq_power(psd, freqs, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return float(np.sum(psd[:, idx]))  # Convert to float

    for channel in channels:
        spectrum = raw_processed.compute_psd(
            method='welch',
            picks=channel,
            fmin=0.5,
            fmax=40,
            n_fft=int(raw_processed.info['sfreq'] * 4),
            n_overlap=int(raw_processed.info['sfreq'] * 2)
        )

        psd = spectrum.get_data()
        freqs = spectrum.freqs

        # Get total power and ensure it's a scalar
        total_power = get_freq_power(psd, freqs, 0.5, 40)

        # Store results for each channel-band combination
        for band_name, (fmin, fmax) in freq_bands.items():
            band_power = get_freq_power(psd, freqs, fmin, fmax)
            normalized_power = band_power / total_power if total_power > 0 else 0
            power_results[f"{channel}_{band_name}"] = normalized_power  # No need for [0]

        # Store total power for each channel (multiplied by 10^6)
        power_results[f"{channel}_total_power"] = total_power

    # Calculate total brain power for control
    total_brain_power = sum(power_results[f"{ch}_total_power"] for ch in channels)
    power_results['total_brain_power'] = total_brain_power

    # Calculate delta/gamma, delta/sigma, sigma/beta ratios
    for channel in channels:
        delta_power = power_results[f"{channel}_delta"]
        sigma_power = power_results[f"{channel}_sigma"]
        beta_power = power_results[f"{channel}_beta"]
        gamma_power = power_results[f"{channel}_gamma"]
        delta_gamma_ratio = delta_power / gamma_power if gamma_power > 0 else 0
        delta_sigma_ratio = delta_power / sigma_power if gamma_power > 0 else 0
        sigma_beta_ratio = delta_power / beta_power if gamma_power > 0 else 0
        power_results[f"{channel}_delta_gamma_ratio"] = delta_gamma_ratio
        power_results[f"{channel}_delta_sigma_ratio"] = delta_gamma_ratio
        power_results[f"{channel}_sigma_beta_ratio"] = delta_gamma_ratio
    

    return power_results


def get_seizures_list(pat_num, surf, data_path=DATA_PATH):
    """
    Load seizures table for a specific patient and hospital.
    """
    seizures_list_path = data_path / "tables" / "seizure_tables" / f"{pat_num}_{surf}_seizures.csv"
    print(f"Attempting to load seizures list from: {seizures_list_path}")

    if not seizures_list_path.exists():
        raise FileNotFoundError(f"Seizures list file not found: {seizures_list_path}")

    seizures_list_table = pd.read_csv(seizures_list_path)
    return seizures_list_table


def seizure_num_to_raw_data(pat_num, seizure_index, seizures_list_table, surf):
    """
    Finds the compatible recording for seizure.
    """
    # Extract the recording number for the specific seizure
    seizure_rec_num = seizures_list_table.loc[
        seizures_list_table['seizure_num'] == seizure_index,
        'file_seizure_ind'].iloc[0]

    # Load the recordings data table using hospital-specific path
    seizures_data_path = DATA_PATH / "tables" / "pat_file_tables" / f"pat_{pat_num}_{surf}_file_list.csv"
    print(f"Looking for file list at: {seizures_data_path}")

    if not seizures_data_path.exists():
        raise FileNotFoundError(f"Patient file list not found: {seizures_data_path}")

    seizures_data_table = pd.read_csv(seizures_data_path)

    # Get the recording path and process it
    file_path = seizures_data_table.loc[seizure_rec_num, 'file_path']
    file_path = file_path.replace('head', 'data')

    if not file_path.endswith('.data'):
        file_path += '.data'

    # Process the path to get the correct subpath
    raw_data_index = file_path.find("raw_data")
    if raw_data_index == -1:
        raise ValueError(f"Could not find 'raw_data' in path: {file_path}")

    subpath = file_path[raw_data_index:]
    seizure_recording_path = DATA_PATH / subpath
    print(f"Attempting to read EEG data from: {seizure_recording_path}")

    if not seizure_recording_path.exists():
        raise FileNotFoundError(f"EEG data file not found: {seizure_recording_path}")

    raw = mne.io.read_raw_nicolet(seizure_recording_path, ch_type='eeg', preload=True)
    return raw


def copy_and_crop(raw, seizure_ind, seizures_list_table, sec_before=60, sec_after=0):
    """
    Cut the data +- desired sec to extract the seizure from in the recording
    """
    global adjusted_crop_count
    raw = raw.copy()
    
    print("Getting table index")
    matching_rows = seizures_list_table[seizures_list_table['seizure_num'] == seizure_ind]
    if matching_rows.empty:
        raise ValueError(f"No seizure found with index {seizure_ind}")

    table_index = matching_rows.index[0]
    recording_start = np.datetime64(raw.info['meas_date'])

    print("Getting onset/offset times")
    onset_str = seizures_list_table.loc[table_index, 'onset'].strip()
    print(f"Raw onset time string: '{onset_str}'")

    seizure_start = pd.to_datetime(onset_str, format='%Y-%m-%d %H:%M:%S')
    print(f"Successfully parsed onset time: {seizure_start}")

    seizure_start = np.datetime64(seizure_start)
    recording_start = recording_start.astype('datetime64[us]')
    seizure_start = seizure_start.astype('datetime64[us]')

    seizure_start_from_tmin = (seizure_start - recording_start) / np.timedelta64(1, 's')
    print(f"Seizure starts at {seizure_start_from_tmin} seconds from recording start")

    crop_start = seizure_start_from_tmin - sec_before
    crop_end = seizure_start_from_tmin

    if crop_start < 0:
        print(f"⚠️ crop_start ({crop_start:.2f}) < 0, adjusting to 0")
        crop_start = 0
        adjusted_crop_count += 1

    max_time = raw.times[-1]
    if crop_end > max_time:
        print(f"⚠️ crop_end ({crop_end:.2f}) > recording duration ({max_time:.2f}), adjusting.")
        crop_end = max_time
        adjusted_crop_count += 1

    raw_cropped = raw.copy().crop(tmin=crop_start, tmax=crop_end)
    return raw_cropped


def analyze_spectral_power(raw, base_pat_num, full_pat_num, seizure_info):
    """
    Analyze spectral power across all frequency bands for each electrode and return a single row of results.

    Parameters:
    -----------
    raw : mne.io.Raw
        The preprocessed EEG data
    base_pat_num : str
        The base patient number used for file operations
    full_pat_num : str
        The full patient number including suffix
    seizure_info : pd.Series
        Information about the seizure from the seizures list table
    """
    raw_processed = preprocess_eeg(raw)
    power_results = compute_power_spectrum(raw_processed)

    # Create base result dictionary with metadata
    result = {
        'pat_num': full_pat_num,  # Use full patient number with suffix
        'base_pat_num': base_pat_num,  # Store original number used for file lookup
        'seizure_num': seizure_info['seizure_num'],
        'classif.': seizure_info['classif.'],
        'onset': seizure_info['onset'],
        'offset': seizure_info['offset'],
        'vigilance': seizure_info['vigilance'],
        'vigilance_code': get_vigilance_code(seizure_info['vigilance']),
        'origin': seizure_info['origin'],
        'origin_code': get_lobe_code(seizure_info['origin']),
        'file_seizure_ind': seizure_info['file_seizure_ind']
    }

    # Add all the per-electrode, per-band results
    result.update(power_results)

    return result


def main_analysis(base_pat_num, full_pat_num, seizure_index, seizures_list_table, surf, data_path=DATA_PATH):
    """Process a single seizure and return its analysis results."""
    seizure_info = seizures_list_table[seizures_list_table['seizure_num'] == seizure_index].iloc[0]
    raw_data = seizure_num_to_raw_data(base_pat_num, seizure_index, seizures_list_table, surf)
    raw_cropped = copy_and_crop(raw_data, seizure_index, seizures_list_table)
    return analyze_spectral_power(raw_cropped, base_pat_num, full_pat_num, seizure_info)


def process_hospital(surf, surf_suffix_to_remove, data_path):
    """Process all patients and seizures for a single hospital."""
  
    hospital_path = data_path / "raw_data" / surf
    print(f"Checking hospital path: {hospital_path}")

    if not hospital_path.exists():
        raise FileNotFoundError(f"Hospital directory not found: {hospital_path}")

    pat_list = list(filter(lambda x: x.startswith("pat_"),
                            os.listdir(hospital_path)))

    pat_num_list = [pat.replace('pat_', '') for pat in pat_list]

    hospital_results = []

    for pat in pat_num_list:
        print(f"Processing patient {pat} from hospital {surf}")
        # Get base patient number first for file operations
        base_pat_num, full_pat_num = get_pat_info(pat, surf)

        # Use base_pat_num for getting seizures list
        seizures_list_table = get_seizures_list(base_pat_num, surf)
        seizures_list = seizures_list_table['seizure_num'].tolist()

        for seizure in seizures_list:
            print(f"Processing seizure {seizure}")
            result = main_analysis(base_pat_num, full_pat_num, seizure, seizures_list_table, surf)
            result['hospital'] = surf
            hospital_results.append(result)
            print(
                f"Completed analysis for patient {full_pat_num} (base: {base_pat_num}), seizure {seizure}")



    return hospital_results

if __name__ == "__main__":
    # Define parameters
    surf_list = ["surf30", "surfCO", "surfPA", "CO"]
    surf_suffix_mapping = {
        "surf30": "02",
        "surfCO": "00",
        "surfPA": "03",
        "CO": "00"
    }

    all_results = []

    for surf in surf_list:
        print(f"\nProcessing hospital: {surf}")
        surf_suffix = surf_suffix_mapping.get(surf, "")
        hospital_results = process_hospital(surf, surf_suffix, DATA_PATH)
        if hospital_results:  # Only extend if we got results
            all_results.extend(hospital_results)

        if not all_results:
            print("No results were collected. Check the error messages above.")
            sys.exit(1)

        final_df = pd.DataFrame(all_results)
        os.makedirs('D:/seizures_analysis/output/', exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f'D:/seizures_analysis/output/all_electrodes_all_host_seizures_analysis_{timestamp}.csv')

        # Get actual columns from the DataFrame
        available_columns = final_df.columns.tolist()

        # Define preferred order for known columns
        preferred_columns = [
            'hospital',
            'pat_num',
            'base_pat_num',
            'seizure_num',
            'classif.',
            'vigilance',
            'vigilance_code',
            'origin',
            'origin_code',
            'onset',
            'offset',
            'file_seizure_ind',
            'total_brain_power'
        ]

    # Create actual column order using available columns
    column_order = [col for col in preferred_columns if col in available_columns]

    # Add remaining columns alphabetically
    remaining_columns = [col for col in available_columns if col not in column_order]
    remaining_columns.sort()
    column_order.extend(remaining_columns)

    # Reorder columns and save
    final_df = final_df[column_order]
    final_df.to_csv(output_path, index=False)
    print(f"Analysis complete. All results saved to {output_path}")

    print("\nAnalysis Summary:")
    print(f"Total hospitals processed: {len(surf_list)}")
    print("Records per hospital:")
    print(final_df['hospital'].value_counts())
    print("\nTotal records:", len(final_df))
    print("\nColumns in output:", len(final_df.columns))
    print(f"Number of seizures with crop_end adjusted: {adjusted_crop_count}")
