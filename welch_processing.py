import pandas as pd
import numpy as np
import mne
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('QtAgg')
from datetime import datetime, timedelta

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

    Args:
        raw (mne.io.Raw): Raw EEG data

    Returns:
        mne.io.Raw: Preprocessed EEG data
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
    Compute power spectrum and analyzed frequency bands for each channel.

    Args:
        raw_processed (mne.io.Raw): Preprocessed EEG data

    Returns:
        pd.DataFrame: Results containing power values for different frequency bands
    """
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
    """
    Main function to analyze delta power in EEG data.

    Args:
        raw (mne.io.Raw): Raw EEG data

    Returns:
        pd.DataFrame: Results of power analysis
    """
    # 1. Preprocess the data
    raw_processed = preprocess_eeg(raw)

    # 2. Compute power spectrum and analyze
    df_results = compute_power_spectrum(raw_processed)

    # 3. Save results
    output_path = '/Users/maya/Documents/backup_lab_project/exel_files/delta_power_analysis.csv_tests'
    df_results.to_csv(output_path, index=False)
    print(f"Analysis complete. Results saved to {output_path}")

    return df_results



if __name__ == "__main__":
    try:
        # Load your EEG data
        raw = mne.io.read_raw_nicolet('/Users/maya/Documents/backup_lab_project/data/100102_0075.data', ch_type='eeg', preload=True)

        # Run the analysis
        results = analyze_delta_power(raw)
        print("\nFinal Results:")
        print(results)

    except Exception as e:
        print(f"An error occurred: {e}")