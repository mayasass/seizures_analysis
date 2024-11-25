import pandas as pd
import numpy as np
import mne
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('QtAgg')
from datetime import datetime, timedelta

def plot_seizure(cropped_raw):
    fig = cropped_raw.plot(
        scalings='auto',
        title=('EEG Data Around The Seizure'),
        block=True)  # Added block=True
    plt.show(block=True)

def cut_seizure(raw, onset, offset, pre_window_sec: int = 60, post_window_sec: int = 60):
    recording_start = np.datetime64(raw.info['meas_date'])
    print(recording_start)
    """""
        seizure_start = np.datetime64(seizures_data.loc[i,'onset'])
        seizure_end = np.datetime64(seizures_data.loc[i, 'offset'])
    """""
    # 1) Dividing to get duration in sec
    seizure_start_from_tmin = (onset - recording_start) / np.timedelta64(1, 's')
    seizure_end_from_tmin = (offset - recording_start) / np.timedelta64(1, 's')

    # 2) Calculate crop times in sec
    crop_start = seizure_start_from_tmin - pre_window_sec
    crop_end = seizure_end_from_tmin + post_window_sec
    print('c(srop length ec) for seizure is: ', (crop_end-crop_start))

    # 3) Make sure we're not overflowing
    if crop_start < 0:
        crop_start = 0
    print('crop start time', crop_start)

    # 4) Cropping the data
    cropped_raw = raw.copy().crop(tmin=crop_start, tmax=crop_end)
    # Maybe: raising assert for data exception

    return cropped_raw

"""""
NOT SURE HOW TO GET DATA PATH

# 1) Loading data to a data frame
seizures_data = pd.read_csv('data/1_surf30_seizures.csv') # waiting for path
data_df = pd.read_csv('data/pat_1_surf30_file_list.csv') # waiting for path

# 2) Load EEG file (using MNE nicolet reader) and get the total recording start time
path = 'data/' + patient_id + '.data'
raw = mne.io.read_raw_nicolet(path, ch_type='eeg', preload=True)
recording_start = np.datetime64(raw.info['meas_date'])


seizures_data = pd.read_csv('data/1_surf30_seizures.csv')  # waiting for path
data_df = pd.read_csv('data/pat_1_surf30_file_list.csv')  # waiting for path
path = 'data/' + patient_id + '.data'
raw = mne.io.read_raw_nicolet(path, ch_type='eeg', preload=True)
"""""

def analyze_delta_power(cropped_raw):
    """
    Preprocess EEG data and analyze delta power for each channel.

    Args:
        cropped_raw: MNE Raw object containing the cropped EEG data
    """
    # 1) Preprocessing
    # Create a copy to avoid modifying the original
    raw_processed = cropped_raw.copy()

    # Define the channel name mapping in a dictionary
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

    # Rename channels if any need to be renamed
    if channels_to_rename: # if the dictionary is not empty
        # Use MNE's function to rename the channels:
        # keys are the old channel names
        # values are the new channel names
        raw_processed.rename_channels(channels_to_rename)

    # Apply bandpass filter (0.5-40 Hz)
    raw_processed.filter(l_freq=0.5, h_freq=40)

    # Set average reference
    raw_processed.set_eeg_reference(ref_channels='average')

    # 2) Power Analysis
    # Get channel names
    channels = raw_processed.ch_names

    # Initialize lists to store results
    results = []

    # Calculate power spectrum for each channel
    for channel in channels:
        # Extract data for this channel
        data, times = raw_processed[channel]

        # Calculate power spectrum using Welch's method
        # psds = power spectral density values
        psds, freqs = raw_processed.compute_psd(
            method='welch',
            picks=channel,
            fmin=0.5,
            fmax=40,
            n_fft=int(raw_processed.info['sfreq'] * 4),
            n_overlap=int(raw_processed.info['sfreq'] * 2)
            ).get_data(return_freqs=True)

        # Use it in your analysis loop:
        for channel in channels:
            psds, freqs = raw_processed.compute_psd(
            method='welch',
            picks=channel,
            fmin=0.5,
            fmax=40,
            n_fft=int(raw_processed.info['sfreq'] * 4),
            n_overlap=int(raw_processed.info['sfreq'] * 2)
            )
            plot_psd(psds, freqs, channel)

        psds, freqs = raw_processed.compute_spectrum(
            psds, freqs=raw_processed.compute_psd(
                method='welch',
                picks=channel,
                fmin=0.5,
                fmax=40,
                n_fft=int(raw_processed.info['sfreq'] * 4),
                n_overlap=int(raw_processed.info['sfreq'] * 2)
        ).get_data(return_freqs=True))
        inspect_spectrum(psds, freqs)

        # Calculate power in specific bands
        # Helper function to get power in specific frequency range
        def get_freq_power(psds, freqs, fmin, fmax):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            return np.mean(psds[:, idx], axis=1)[0]

        # Calculate powers
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

    # Save to CSV
    output_path = 'EEGTableAnalysis/delta_power_analysis.csv'
    df_results.to_csv(output_path, index=False)

    print(f"Analysis complete. Results saved to {output_path}")

    return df_results

"""""
code visualization
"""""
def plot_psd(psds, freqs, channel_name):
    plt.figure(figsize=(10, 6))
    plt.semilogy(freqs, psds[0])  # Log scale for better visualization
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (µV²/Hz)')
    plt.title(f'PSD for channel {channel_name}')
    plt.grid(True)
    plt.axvspan(0.5, 2, color='red', alpha=0.3, label='Delta 0.5-2 Hz')
    plt.axvspan(1, 4, color='blue', alpha=0.3, label='Delta 1-4 Hz')
    plt.legend()
    plt.show()


# Add this to your code to see what the data looks like
def inspect_spectrum(psds, freqs):
    print("First few frequencies (Hz):", freqs[:5])
    print("First few power values:", psds[0][:5])

    # Optional: Plot the spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, psds[0])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title(f'Spectrum for channel')
    plt.show()

def main():
    # Example usage
    try:
        # Get your raw EEG data and seizure times
        raw = mne.io.read_raw_nicolet('data/100102_0075.data', ch_type='eeg', preload=True)
        seizures_data = pd.read_csv('data/1_surf30_seizures.csv')
        onset = np.datetime64(seizures_data.loc[0, 'onset'])  # example
        offset = np.datetime64(seizures_data.loc[0, 'offset'])  # example

        # Cut the seizure
        cropped_raw = cut_seizure(raw, onset, offset)

        # Optional: Plot the cropped data
        #plot_seizure(cropped_raw)

        # Analyze delta power
        power_results = analyze_delta_power(cropped_raw)

        print("Power analysis results:")
        print(power_results)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
