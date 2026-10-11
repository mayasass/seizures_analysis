import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import mne

# Import functions from data_processing.py
from data_processing import (
    DATA_PATH,
    seizure_num_to_raw_data,
    copy_and_crop,
    get_seizures_list,
    preprocess_eeg
)

# Define electrode groups by lobe
LOBE_ELECTRODES = {
    'frontal': {'FP1', 'FP2', 'F3', 'F4', 'F7', 'F8', 'FZ'},
    'temporal': {'T7', 'T8', 'P7', 'P8'},
    'central': {'C3', 'C4', 'CZ'},
    'parietal': {'P3', 'P4', 'PZ'},
    'occipital': {'O1', 'O2'}
}


def compute_lobe_power_spectrum(raw_processed, lobe_channels):
    """
    Compute normalized power spectrum for specific lobe channels.

    Args:
        raw_processed (mne.io.Raw): Preprocessed EEG data
        lobe_channels (set): Set of channel names for specific lobe
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

    # Initialize power results
    power_results = {f"{band}_power": 0 for band in freq_bands.keys()}

    # Get available channels (intersection of lobe channels and data channels)
    available_channels = set(raw_processed.ch_names) & lobe_channels
    if not available_channels:
        raise ValueError(f"No channels available for lobe analysis. Expected: {lobe_channels}")

    channel_count = len(available_channels)
    print(f"Analyzing {channel_count} channels: {sorted(available_channels)}")

    def get_freq_power(psd, freqs, fmin, fmax):
        idx = np.logical_and(freqs >= fmin, freqs <= fmax)
        return np.sum(psd[:, idx])

    total_power = 0
    # Process each channel in the lobe
    for channel in available_channels:
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

        # Calculate total power for current channel
        channel_total_power = get_freq_power(psd, freqs, 0.5, 40)
        total_power += channel_total_power

        # Calculate normalized power for each band
        for band_name, (fmin, fmax) in freq_bands.items():
            band_power = get_freq_power(psd, freqs, fmin, fmax)
            normalized_power = band_power / channel_total_power
            power_results[f"{band_name}_power"] += normalized_power

    # Average the frequency band powers
    for band_name in freq_bands.keys():
        power_results[f"{band_name}_power"] /= channel_count

    # Add the average total power as a separate metric
    power_results['total_power'] = total_power / channel_count
    return power_results


def analyze_lobe_spectral_power(raw, pat_num, seizure_info, lobe_name, lobe_channels):
    """
    Analyze spectral power for a specific lobe.
    """
    raw_processed = preprocess_eeg(raw)
    power_results = compute_lobe_power_spectrum(raw_processed, lobe_channels)

    result = {
        'pat_num': pat_num,
        'seizure_num': seizure_info['seizure_num'],
        'classif.': seizure_info['classif.'],
        'onset': seizure_info['onset'],
        'offset': seizure_info['offset'],
        'vigilance': seizure_info['vigilance'],
        'origin': seizure_info['origin'],
        'file_seizure_ind': seizure_info['file_seizure_ind'],
        'lobe': lobe_name
    }

    result.update(power_results)
    return result


def process_hospital_by_lobe(surf, surf_suffix_to_remove, data_path):
    """Process all patients and seizures for a single hospital, separated by lobe."""
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

        # Initialize results dictionary for each lobe
        lobe_results = {lobe: [] for lobe in LOBE_ELECTRODES.keys()}

        for pat in pat_num_list:
            print(f"Processing patient {pat} from hospital {surf}")
            try:
                seizures_list_table = get_seizures_list(pat, surf)
                seizures_list = seizures_list_table['seizure_num'].tolist()

                for seizure in seizures_list:
                    print(f"Processing seizure {seizure}")
                    try:
                        # Get the raw data once for all lobes
                        raw_data = seizure_num_to_raw_data(pat, seizure, seizures_list_table, surf)
                        raw_cropped = copy_and_crop(raw_data, seizure, seizures_list_table)
                        seizure_info = seizures_list_table[seizures_list_table['seizure_num'] == seizure].iloc[0]

                        # Process each lobe
                        for lobe_name, lobe_channels in LOBE_ELECTRODES.items():
                            try:
                                result = analyze_lobe_spectral_power(
                                    raw_cropped, pat, seizure_info, lobe_name, lobe_channels
                                )
                                result['hospital'] = surf
                                lobe_results[lobe_name].append(result)
                            except Exception as e:
                                print(f"Error processing {lobe_name} lobe for seizure {seizure}: {str(e)}")
                                continue

                    except Exception as e:
                        print(f"Error processing seizure {seizure} for patient {pat}: {str(e)}")
                        continue

            except Exception as e:
                print(f"Error processing patient {pat}: {str(e)}")
                continue

        return lobe_results
    except Exception as e:
        print(f"Error processing hospital {surf}: {str(e)}")
        return {lobe: [] for lobe in LOBE_ELECTRODES.keys()}


if __name__ == "__main__":
    # Define parameters
    surf_list = ["surf30", "surfCO", "surfPA"]
    surf_suffix_mapping = {
        "surf30": "02",
        "surfCO": "00",
        "surfPA": "03"
    }

    # Initialize results dictionary for each lobe
    all_lobe_results = {lobe: [] for lobe in LOBE_ELECTRODES.keys()}

    # Process each hospital
    for surf in surf_list:
        print(f"\nProcessing hospital: {surf}")
        try:
            surf_suffix = surf_suffix_mapping.get(surf, "")
            hospital_results = process_hospital_by_lobe(surf, surf_suffix, DATA_PATH)

            # Combine results for each lobe
            for lobe in LOBE_ELECTRODES.keys():
                all_lobe_results[lobe].extend(hospital_results[lobe])

        except Exception as e:
            print(f"Error processing hospital {surf}: {str(e)}")
            continue

    # Save results for each lobe
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path('D:/seizures_analysis/output/lobe_analysis')
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create and save DataFrame for each lobe
        for lobe, results in all_lobe_results.items():
            if results:  # Only create file if there are results
                lobe_df = pd.DataFrame(results)
                output_path = output_dir / f'spectrum_analysis_{lobe}_{timestamp}.csv'
                lobe_df.to_csv(output_path, index=False)
                print(f"\nAnalysis for {lobe} lobe complete. Results saved to {output_path}")
                print(f"Total records for {lobe}: {len(results)}")
                if len(results) > 0:
                    print("Records per hospital:")
                    print(lobe_df['hospital'].value_counts())
            else:
                print(f"\nNo results for {lobe} lobe")

    except Exception as e:
        print(f"Error saving final results: {str(e)}")