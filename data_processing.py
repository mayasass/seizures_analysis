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
    """
    raw_processed = raw.copy()

    standard_electrodes = {'FP1', 'FP2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4',
                           'O1', 'O2', 'F7', 'F8', 'T7', 'T8', 'P7', 'P8',
                           'FZ', 'CZ', 'PZ'}

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
    return raw_processed


def compute_power_spectrum(raw_processed):
    """
    Compute normalized power spectrum for multiple frequency bands for each channel.
    """
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
    total_power = 0

    def get_freq_power(psds, freqs, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.sum(psds[:, idx])

    channel_count = len(channels)

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

        total_power = get_freq_power(psd, freqs, 0.5, 40)

        for band_name, (fmin, fmax) in freq_bands.items():
            band_power = get_freq_power(psd, freqs, fmin, fmax)
            normalized_power = band_power / total_power
            power_results[f"{band_name}_power"] += normalized_power

    # Average the frequency band powers
    for band_name in freq_bands.keys():
        power_results[f"{band_name}_power"] /= channel_count

    # Add the average total power as a separate metric
    power_results['total_power'] = total_power / channel_count
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

    try:
        raw = mne.io.read_raw_nicolet(seizure_recording_path, ch_type='eeg', preload=True)
        return raw
    except Exception as e:
        raise Exception(f"Error reading EEG file {seizure_recording_path}: {str(e)}")


def copy_and_crop(raw, seizure_ind, seizures_list_table, sec_before=60, sec_after=0):
    """
    Cut the data +- desired sec to extract the seizure from in the recording
    """
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

    try:
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
            print('Warning: crop_start adjusted to recording start (0)')
            crop_start = 0

        raw_cropped = raw.copy().crop(tmin=crop_start, tmax=crop_end)
        return raw_cropped

    except Exception as e:
        print(f"Error occurred while processing times: {str(e)}")
        print(f"Full row data: {seizures_list_table.loc[table_index]}")
        print(f"Data types: {seizures_list_table.loc[table_index].dtypes}")
        raise


def analyze_spectral_power(raw, pat_num, seizure_info):
    """
    Analyze spectral power across all frequency bands and return a single row of results.
    """
    raw_processed = preprocess_eeg(raw)
    power_results = compute_power_spectrum(raw_processed)

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

    result.update(power_results)
    return result


def main_analysis(pat_num, seizure_index, seizures_list_table, surf, data_path=DATA_PATH):
    """Process a single seizure and return its analysis results."""
    seizure_info = seizures_list_table[seizures_list_table['seizure_num'] == seizure_index].iloc[0]
    raw_data = seizure_num_to_raw_data(pat_num, seizure_index, seizures_list_table, surf)
    raw_cropped = copy_and_crop(raw_data, seizure_index, seizures_list_table)
    return analyze_spectral_power(raw_cropped, pat_num, seizure_info)


def process_hospital(surf, surf_suffix_to_remove, data_path):
    """Process all patients and seizures for a single hospital."""
    try:
        hospital_path = data_path / "raw_data" / surf
        print(f"Checking hospital path: {hospital_path}")

        if not hospital_path.exists():
            raise FileNotFoundError(f"Hospital directory not found: {hospital_path}")

        pat_list = list(filter(lambda x: x.startswith("pat_"),
                               os.listdir(hospital_path)))

        pat_num_list = [pat.replace('pat_', '') for pat in pat_list]
        pat_num_list = [num[:-2] if num.endswith(surf_suffix_to_remove) else num
                        for num in pat_num_list]

        hospital_results = []

        for pat in pat_num_list:
            print(f"Processing patient {pat} from hospital {surf}")
            try:
                seizures_list_table = get_seizures_list(pat, surf)
                seizures_list = seizures_list_table['seizure_num'].tolist()

                for seizure in seizures_list:
                    print(f"Processing seizure {seizure}")
                    try:
                        result = main_analysis(pat, seizure, seizures_list_table, surf)
                        result['hospital'] = surf
                        hospital_results.append(result)
                        print(f"Completed analysis for patient {pat}, seizure {seizure}")
                    except Exception as e:
                        print(f"Error processing seizure {seizure} for patient {pat}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing patient {pat}: {str(e)}")
                continue

        return hospital_results
    except Exception as e:
        print(f"Error processing hospital {surf}: {str(e)}")
        return []


if __name__ == "__main__":
    # Define parameters
    surf_list = ["surf30", "surfCO", "surfPA"]
    surf_suffix_mapping = {
        "surf30": "02",
        "surfCO": "00",
        "surfPA": "03"
    }

    all_results = []

    for surf in surf_list:
        print(f"\nProcessing hospital: {surf}")
        try:
            surf_suffix = surf_suffix_mapping.get(surf, "")
            hospital_results = process_hospital(surf, surf_suffix, DATA_PATH)
            all_results.extend(hospital_results)
        except Exception as e:
            print(f"Error processing hospital {surf}: {str(e)}")
            continue

    try:
        final_df = pd.DataFrame(all_results)
        os.makedirs('D:/seizures_analysis/output/', exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = Path(f'D:/seizures_analysis/output/all_spectrum_all_host_seizures_analysis_{timestamp}.csv')

        final_df.to_csv(output_path, index=False)
        print(f"Analysis complete. All results saved to {output_path}")

        print("\nAnalysis Summary:")
        print(f"Total hospitals processed: {len(surf_list)}")
        print("Records per hospital:")
        print(final_df['hospital'].value_counts())

    except Exception as e:
        print(f"Error saving final results: {str(e)}")