import pandas as pd
import numpy as np
import mne
import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('QtAgg')
from datetime import datetime, timedelta

# 1) Load data to dataframes
seizures_data = pd.read_csv('data/1_surf30_seizures.csv')
big_data = pd.read_csv('data/pat_1_surf30_file_list.csv')

# 2) Load EEG file (using MNE nicolet reader) and get the total recording start time
raw = mne.io.read_raw_nicolet('data/100102_0075.data',ch_type='eeg', preload=True)
recording_start = np.datetime64(raw.info['meas_date'])
print(recording_start)

# 3) Setting analyzing time before and after a seizure:
sec_before = 60
sec_after = 60
# Get the start time of first seizure
seizure_start = np.datetime64(seizures_data.loc[0, 'onset'])
print(seizure_start)

# 4) Get the end time of first seizure
seizure_end = np.datetime64(seizures_data.loc[0, 'offset'])
print(seizure_end)

# 5) Subtraction of start and end from recording start to get duration in sec
seizure_start_from_tmin = (seizure_start - recording_start) / np.timedelta64(1,'s')
print(seizure_start_from_tmin)
seizure_end_from_tmin = (seizure_end - recording_start) / np.timedelta64(1,'s')
print(seizure_end_from_tmin)

# 6) Calculate crop times in sec (5 minutes before and after (60s))
crop_start = seizure_start_from_tmin - sec_before
crop_end = seizure_end_from_tmin + sec_after
print(crop_start)
print(crop_end)
# Make sure we're reaching out in recording borders
if crop_start < 0:
    crop_start = 0
print('crop start time', crop_start)

# 7) Cropping the data
raw_cropped = raw.copy().crop(tmin=crop_start,tmax=crop_end)
# Maybe: raising assert for data exception

# 8) Plot the cropped data
#raw_cropped.plot(scalings='auto', title='EEG Data Around First Seizure')
fig = raw_cropped.plot(scalings='auto', title='EEG Data Around First Seizure', block=True)  # Added block=True
plt.show(block=True)