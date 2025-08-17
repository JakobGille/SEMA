# -*- coding: utf-8 -*-
"""
SEMA: Statistic Electroacoustic Music Analyzer (Modified Version)

This script is a customized version of the original SEMA project, modified with the help of Gemini Ai to
enhance usability and focus on direct data visualization. The core audio feature
extraction is based on the original, but the workflow and output have been
altered.

Key Modifications:
- Gemini AI Removal: All integration with the Google Gemini API for automated
  musicological analysis has been removed. The script operates entirely
  locally and offline.
- GUI File Selector: Replaced the command-line argument with a graphical file
  dialog (using Tkinter), allowing users to select an audio file directly.
- Enhanced Plotting:
    - Trend Lines: Added a smoothed trend line (moving average) to each
      primary plot for improved readability of musical contours.
    - MM:SS Time Format: The X-axis is now formatted as Minutes:Seconds for
      intuitive interpretation.
  - Streamlined Output: The script generates a comprehensive CSV file and a
  multi-plot PNG image in a single, user-friendly execution.
"""

import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import tkinter as tk
from tkinter import filedialog
from matplotlib.ticker import FuncFormatter

# --- 1. File Dialog for Audio File Selection ---
print("Opening file dialog to select a .wav file...")
root = tk.Tk()
root.withdraw()

audio_file_path = filedialog.askopenfilename(
    title="Select a WAV audio file for analysis",
    filetypes=(("WAV Files", "*.wav"), ("All Files", "*.*"))
)

if not audio_file_path:
    print("No file selected. Exiting script.")
    sys.exit()

file_name = os.path.basename(audio_file_path)

# --- 2. Load Audio ---
try:
    print(f"Loading audio from: {audio_file_path}")
    y, sr = librosa.load(audio_file_path, sr=22050)
    print(f"Audio loaded. Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")
except Exception as e:
    print(f"An error occurred while loading the audio file: {e}")
    sys.exit()

# --- 3. Feature Extraction ---
print("Extracting features...")
frame_length = 2048
hop_length = 512
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]
onset_sf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# --- 4. Organize, Round, and Smooth Data ---
print("Organizing, rounding, and smoothing data...")
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

df = pd.DataFrame({
    'Time_Seconds': times, 'RMS_Energy': rms, 'Spectral_Centroid': cent,
    'ZCR': zcr, 'Novelty_Curve': onset_sf,
})
df = df.round({
    'Time_Seconds': 3, 'RMS_Energy': 4, 'Spectral_Centroid': 2,
    'ZCR': 4, 'Novelty_Curve': 4,
})

window_size = 10
for col in ['RMS_Energy', 'Spectral_Centroid', 'ZCR', 'Novelty_Curve']:
    df[f'{col}_smooth'] = df[col].rolling(window=window_size, center=True, min_periods=1).mean()

# --- 5. Save to CSV ---
output_csv_path = f'{os.path.splitext(file_name)[0]}_rounded.csv'
df.to_csv(output_csv_path, index=False)
print(f"Data successfully saved to {output_csv_path}")

# --- 6. Create and Save Plots ---
print("Creating plots...")

# Function to format seconds into MM:SS
def format_time(x, pos):
    minutes = int(x // 60)
    seconds = int(x % 60)
    return f'{minutes:02d}:{seconds:02d}'

time_formatter = FuncFormatter(format_time)

plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle(f'Analysis of: {file_name}', fontsize=16)

# Plot 1: RMS Energy
axes[0].plot(df['Time_Seconds'], df['RMS_Energy'], color='lightblue', alpha=0.6, label='Original')
axes[0].plot(df['Time_Seconds'], df['RMS_Energy_smooth'], color='darkblue', label=f'Trend (Window={window_size})')
axes[0].set_title('RMS Energy: Overall loudness/intensity.')
axes[0].set_ylabel('RMS')
axes[0].legend(loc='upper right')

# Plot 2: Novelty Curve
axes[1].plot(df['Time_Seconds'], df['Novelty_Curve'], color='lightgreen', alpha=0.6, label='Original')
axes[1].plot(df['Time_Seconds'], df['Novelty_Curve_smooth'], color='darkgreen', label='Trend')
axes[1].set_title('Novelty Curve: Rate of significant sonic changes/onsets (peaks indicate \'newness\' or activity).')
axes[1].set_ylabel('Novelty Value')
axes[1].legend(loc='upper right')

# Plot 3: Spectral Centroid
axes[2].plot(df['Time_Seconds'], df['Spectral_Centroid'], color='#ffcccb', alpha=0.6, label='Original')
axes[2].plot(df['Time_Seconds'], df['Spectral_Centroid_smooth'], color='darkred', label='Trend')
axes[2].set_title('Spectral Centroid: Perceived brightness (higher value = brighter).')
axes[2].set_ylabel('Frequency (Hz)')
axes[2].legend(loc='upper right')

# Plot 4: Zero-Crossing Rate
axes[3].plot(df['Time_Seconds'], df['ZCR'], color='mediumpurple', alpha=0.6, label='Original')
axes[3].plot(df['Time_Seconds'], df['ZCR_smooth'], color='purple', label='Trend')
axes[3].set_title('Zero-Crossing Rate (ZCR): Degree of noisiness vs. tonality.')
axes[3].set_ylabel('ZCR')
axes[3].set_xlabel('Time (Minutes:Seconds)')
axes[3].legend(loc='upper right')

# Apply the custom time formatter to the X-axis
axes[3].xaxis.set_major_formatter(time_formatter)

for ax in axes:
    ax.grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
output_plot_path = f'{os.path.splitext(file_name)[0]}_Feature_Plots.png'
plt.savefig(output_plot_path)
print(f"Plots successfully saved to {output_plot_path}")

print("\nAnalysis complete! ✅")