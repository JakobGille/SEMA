import pandas as pd
import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import linregress
import io
import sys
import datetime
import google.genai as genai  # Import the Gemini library
import config


file_name = sys.argv[1]

# --- Configuration ---
audio_file_path = file_name 
sr_target = 22050 # Target sample rate (LibROSA default, good for most features)
frame_length = 2048 # Number of samples per analysis window (approx 93 ms at 22050 Hz)
hop_length = 512    # Number of samples to advance for next window (approx 23 ms at 22050 Hz)
                    # This means features will be calculated every ~23ms

# --- Load Audio ---
print(f"Loading audio from: {audio_file_path}")
y, sr = librosa.load(audio_file_path, sr=sr_target)
print(f"Audio loaded. Sample rate: {sr}, Duration: {librosa.get_duration(y=y, sr=sr):.2f} seconds")

# --- Feature Extraction ---
print("Extracting features...")

# 1. RMS Energy
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# 2. Spectral Centroid
cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

# 3. Spectral Roll-off
roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=frame_length, hop_length=hop_length)[0]

# 4. Novelty Curve (Onset Strength Function)
# get the continuous novelty curve
onset_sf = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)


# 5. Zero-Crossing Rate
zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]

# 6. MFCCs (Mel-Frequency Cepstral Coefficients)
# We'll extract 13 MFCCs, which is a common number.
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop_length)

# --- Create Time Vector ---
# The number of feature frames might differ slightly based on librosa's padding.
# We'll base the time vector on the length of the RMS feature.
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

# --- Organize and Save Data ---
print("Organizing and saving data...")
data = {
    'Time_Seconds': times,
    'RMS_Energy': rms,
    'Spectral_Centroid': cent,
    'Spectral_rolloff': roll,
    'ZCR': zcr,
    'Novelty_Curve': onset_sf, # Assign the actual novelty curve here
}

# Add MFCCs as separate columns
for i in range(mfccs.shape[0]):
    data[f'MFCC_{i+1}'] = mfccs[i]

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
output_csv_path = f'{file_name}.csv' # <--- Output file name
df.to_csv(output_csv_path, index=False)
print(f"Features saved to {output_csv_path}")

# --- Display a snippet (for verification) ---
print("\nFirst 5 rows of extracted features:")
print(df.head())


input_csv_path = f'{file_name}.csv'  # Assumes feature extraction saved here
output_report_name = f'{file_name}_Analysis_Report_Gemini_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'



client = genai.Client(api_key=config.API_KEY)
model = "gemini-2.5-flash"  

# --- Custom Print Function for File Output ---
output_buffer = io.StringIO()


def custom_print(*args, **kwargs):
    """Prints to console and also writes to a string buffer for file output."""
    print(*args, **kwargs)
    print(*args, file=output_buffer, **kwargs)


# --- Gemini Helper Function ---
def get_gemini_interpretation(prompt_text):
    """Sends a prompt to Gemini and returns the interpreted text."""
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt_text,
        )
        #response = client.models.generate_content(prompt_text)
        return response.text.strip()
    except Exception as e:
        return f"[[GEMINI API ERROR: {e} - Could not generate interpretation for this section.]]"


# --- Common Prompt Components for Musicological Context ---
MUSICOLOGICAL_PROMPT_HEADER = f"""
You are an expert musicologist specializing in electroacoustic music analysis.
Your task is to interpret provided audio feature data for the piece {file_name}.
Focus on understanding the piece's formal structure and timbral evolution.
Use formal, academic, and scientific musicological language. Avoid conversational tone.
Connect quantitative data to qualitative musical observations and concepts.
Define the features and provide their meaning in an electroacoustic context:
- RMS Energy: Overall loudness/intensity.
- Spectral Centroid: Perceived brightness (higher value = brighter).
- Zero-Crossing Rate (ZCR): Degree of noisiness vs. tonality (higher value = noisier/more percussive, lower = more tonal/sustained).
- Novelty Curve: Rate of significant sonic changes/onsets (peaks indicate 'newness' or activity).
- MFCCs: Mel-Frequency Cepstral Coefficients, a compact representation of timbral color/texture.
"""

# --- Data Loading ---
custom_print(f"Loading features from: {input_csv_path}")
try:
    df = pd.read_csv(input_csv_path)
    custom_print("CSV loaded successfully.")
    custom_print("First 5 rows of data:")
    custom_print(df.head())
except FileNotFoundError:
    custom_print(f"Error: The file '{input_csv_path}' was not found. Please ensure it's in the same directory.")
    sys.exit()

# --- Store Global Statistics in a Dictionary (NEW AND IMPROVED) ---
global_stats = {
    'RMS_Energy': {
        'mean': df['RMS_Energy'].mean(),
        'std': df['RMS_Energy'].std()
    },
    'Spectral_Centroid': {
        'mean': df['Spectral_Centroid'].mean(),
        'std': df['Spectral_Centroid'].std()
    },
    'ZCR': {
        'mean': df['ZCR'].mean(),
        'std': df['ZCR'].std()
    },
    'Novelty_Curve': {
        'mean': df['Novelty_Curve'].mean(),
        'std': df['Novelty_Curve'].std()
    }
}


# --- Narrative Interpretation Functions (Now using global_stats dictionary) ---

def describe_feature_with_gemini(feature_name, dataframe, context_prefix="Overall", feature_unit=""):
    """Generates a narrative for a single feature using Gemini."""
    values = dataframe[feature_name]
    times = dataframe['Time_Seconds']

    # Statistics for the segment/overall
    mean_val = values.mean()
    std_val = values.std()
    min_val = values.min()
    max_val = values.max()

    slope = 0
    if len(times) > 1:
        slope, _, _, _, _ = linregress(times, values)

    # Global context for comparison - ACCESSING FROM DICTIONARY
    global_mean_val = global_stats[feature_name]['mean']
    global_std_val = global_stats[feature_name]['std']

    prompt_data = f"""
    Context: {context_prefix} analysis of '{feature_name}'.
    Segment Statistics:
    - Mean: {mean_val:.4f}{feature_unit}
    - Standard Deviation (Volatility): {std_val:.4f}{feature_unit}
    - Min: {min_val:.4f}{feature_unit}
    - Max: {max_val:.4f}{feature_unit}
    - Linear Trend (Slope): {slope:.4f}{feature_unit}/s

    Global Statistics (for contextual comparison):
    - Global Mean: {global_mean_val:.4f}{feature_unit}
    - Global Standard Deviation: {global_std_val:.4f}{feature_unit}

    Based on this data, provide a concise musicological interpretation of the {feature_name} profile.
    """

    full_prompt = MUSICOLOGICAL_PROMPT_HEADER + prompt_data
    return get_gemini_interpretation(full_prompt)


def analyze_segment_narrative_with_gemini(dataframe, start_s, end_s, title="Detailed Segment Analysis"):
    """Analyzes a specific segment and generates a comprehensive narrative using Gemini."""
    segment_df = dataframe[(dataframe['Time_Seconds'] >= start_s) & (dataframe['Time_Seconds'] <= end_s)]

    if segment_df.empty:
        return "No data found for the specified segment. Check your time range."

    # Prepare data for the prompt
    segment_stats = {}
    for feature in ['RMS_Energy', 'Spectral_Centroid', 'ZCR', 'Novelty_Curve']:
        stats = {
            'mean': segment_df[feature].mean(),
            'std': segment_df[feature].std(),
            'min': segment_df[feature].min(),
            'max': segment_df[feature].max()
        }
        if len(segment_df['Time_Seconds']) > 1:
            slope, _, _, _, _ = linregress(segment_df['Time_Seconds'], segment_df[feature])
            stats['slope'] = slope
        else:
            stats['slope'] = 0
        segment_stats[feature] = stats

    mfcc_columns = [col for col in segment_df.columns if 'MFCC_' in col]
    mfcc_avg = {}
    if mfcc_columns:
        avg_mfccs = segment_df[mfcc_columns].mean()
        for mfcc_name, value in avg_mfccs.items():
            mfcc_avg[mfcc_name] = value

    prompt_data = f"""
    Segment Time Range: {start_s:.2f}s to {end_s:.2f}s.
    Below are the statistics for this segment compared to the overall piece.

    RMS Energy:
    - Segment Mean: {segment_stats['RMS_Energy']['mean']:.4f} (Global Mean: {global_stats['RMS_Energy']['mean']:.4f})
    - Segment Std Dev: {segment_stats['RMS_Energy']['std']:.4f} (Global Std Dev: {global_stats['RMS_Energy']['std']:.4f})
    - Segment Trend (Slope): {segment_stats['RMS_Energy']['slope']:.4f}/s

    Spectral Centroid:
    - Segment Mean: {segment_stats['Spectral_Centroid']['mean']:.2f} Hz (Global Mean: {global_stats['Spectral_Centroid']['mean']:.2f} Hz)
    - Segment Std Dev: {segment_stats['Spectral_Centroid']['std']:.2f} Hz (Global Std Dev: {global_stats['Spectral_Centroid']['std']:.2f} Hz)
    - Segment Trend (Slope): {segment_stats['Spectral_Centroid']['slope']:.2f} Hz/s

    Zero-Crossing Rate:
    - Segment Mean: {segment_stats['ZCR']['mean']:.4f} (Global Mean: {global_stats['ZCR']['mean']:.4f})
    - Segment Std Dev: {segment_stats['ZCR']['std']:.4f} (Global Std Dev: {global_stats['ZCR']['std']:.4f})
    - Segment Trend (Slope): {segment_stats['ZCR']['slope']:.5f}/s

    Novelty Curve:
    - Segment Mean: {segment_stats['Novelty_Curve']['mean']:.4f} (Global Mean: {global_stats['Novelty_Curve']['mean']:.4f})
    - Segment Std Dev: {segment_stats['Novelty_Curve']['std']:.4f} (Global Std Dev: {global_stats['Novelty_Curve']['std']:.4f})
    - Segment Trend (Slope): {segment_stats['Novelty_Curve']['slope']:.4f}/s

    Average MFCCs for this segment: {', '.join([f'{k}: {v:.4f}' for k, v in mfcc_avg.items()])}.

    Based on these combined statistics, provide a comprehensive musicological interpretation of this segment's dynamic and timbral characteristics, and its potential role in the piece's form. Highlight any significant deviations from the global trends.
    """

    full_prompt = MUSICOLOGICAL_PROMPT_HEADER + prompt_data
    return f"\n### {title} ({start_s:.2f}s - {end_s:.2f}s)\n" + get_gemini_interpretation(full_prompt)


# ... (rest of the Main Analysis Script, Visualizations, Formal Boundary Detection, etc. remains the same) ...

# The rest of the script from the previous message is unchanged below this point.
# It includes plot generation, report content generation, saving the report,
# and the interactive segment analysis loop.


# --- Main Analysis Script ---

# 1. Generate Plots
custom_print("\n--- Generating Plots ---")
plt.style.use('seaborn-v0_8-darkgrid')
fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle(f'Quantitative Analysis of - {file_name}', fontsize=16)

axes[0].plot(df['Time_Seconds'], df['RMS_Energy'], color='darkblue', alpha=0.8)
axes[0].set_title('RMS Energy (Loudness Profile)', fontsize=12)
axes[0].set_ylabel('RMS (Normalized)', fontsize=10)
axes[0].grid(True)

axes[1].plot(df['Time_Seconds'], df['Novelty_Curve'], color='darkgreen', alpha=0.8)
axes[1].set_title('Novelty Curve (Onset Strength Function)', fontsize=12)
axes[1].set_ylabel('Novelty Value', fontsize=10)
axes[1].grid(True)

axes[2].plot(df['Time_Seconds'], df['Spectral_Centroid'], color='darkred', alpha=0.8)
axes[2].set_title('Spectral Centroid (Brightness)', fontsize=12)
axes[2].set_ylabel('Frequency (Hz)', fontsize=10)
axes[2].grid(True)

axes[3].plot(df['Time_Seconds'], df['ZCR'], color='purple', alpha=0.8)
axes[3].set_title('Zero-Crossing Rate (Noisiness/Percussiveness)', fontsize=12)
axes[3].set_ylabel('ZCR', fontsize=10)
axes[3].set_xlabel('Time (Seconds)', fontsize=12)
axes[3].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.96])
plt.savefig(f'{file_name}_Feature_Plots.png')
custom_print(f"Plots saved as {file_name}_Feature_Plots.png")
plt.show()

# 2. Start Report Generation
report_content = io.StringIO()
report_content.write(f"Quantitative Musicological Analysis Report\n")
report_content.write(f"Piece: {file_name}\n")
report_content.write(f"Duration: {df['Time_Seconds'].max():.2f} seconds\n")
report_content.write(f"Date of Analysis: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
report_content.write("--- Introduction ---\n")
report_content.write(
    f"This report presents a quantitative analysis of piece '{file_name}', focusing on its formal structure and timbral evolution. Utilizing audio feature extraction from the {file_name}_features.csv dataset, this analysis employs computational methods to identify patterns in dynamic, timbral, and event-based characteristics over time. These objective metrics, combined with advanced AI interpretation, serve as a foundation for a more informed musicological understanding.\n\n")

report_content.write("--- Methodology ---\n")
report_content.write(
    "Audio features (RMS Energy, Spectral Centroid, Zero-Crossing Rate, Novelty Curve, and Mel-Frequency Cepstral Coefficients) were extracted using the LibROSA Python library. Time-series analysis and descriptive statistics were applied. The collected data was then fed to the Google Gemini Pro large language model, prompted as an expert musicologist, to generate narrative interpretations of global trends and specific segments. Visualizations are provided to illustrate temporal trajectories of these features.\n\n")

report_content.write("--- Global Feature Analysis (AI Interpreted) ---\n")

report_content.write("### Dynamic Profile (RMS Energy)\n")
report_content.write(describe_feature_with_gemini('RMS_Energy', df, context_prefix="Overall") + "\n\n")

report_content.write("### Timbral Brightness (Spectral Centroid)\n")
report_content.write(
    describe_feature_with_gemini('Spectral_Centroid', df, context_prefix="Overall", feature_unit=" Hz") + "\n\n")

report_content.write("### Timbral Noisiness/Tonality (Zero-Crossing Rate)\n")
report_content.write(describe_feature_with_gemini('ZCR', df, context_prefix="Overall") + "\n\n")

report_content.write("### Event Activity (Novelty Curve)\n")
report_content.write(describe_feature_with_gemini('Novelty_Curve', df, context_prefix="Overall") + "\n\n")

report_content.write("--- Indications of Formal Boundaries ---\n")
novelty_peaks_indices, _ = find_peaks(df['Novelty_Curve'],
                                      height=np.mean(df['Novelty_Curve']) + np.std(df['Novelty_Curve']) * 1.0,
                                      prominence=0.1)
novelty_peak_times = df['Time_Seconds'].iloc[novelty_peaks_indices]
report_content.write("### Potential Major Onsets/Changes (Novelty Curve Peaks)\n")
if not novelty_peak_times.empty:
    report_content.write("  Moments of significant spectral change or 'newness' are indicated at:\n")
    for t in novelty_peak_times:
        report_content.write(f"  - {t:.2f} seconds\n")
else:
    report_content.write(
        "  No significant peaks found above the current threshold, indicating a continuously evolving or highly homogeneous texture.\n")

rms_diff = df['RMS_Energy'].diff().abs()
rms_diff_threshold = np.mean(rms_diff.dropna()) + np.std(rms_diff.dropna()) * 2.0
significant_rms_changes_indices = rms_diff[rms_diff > rms_diff_threshold].index
significant_rms_change_times = df['Time_Seconds'].iloc[significant_rms_changes_indices].tolist()
filtered_rms_change_times = []
if significant_rms_change_times:
    filtered_rms_change_times.append(significant_rms_change_times[0])
    for i in range(1, len(significant_rms_change_times)):
        if significant_rms_change_times[i] - filtered_rms_change_times[-1] > 2:
            filtered_rms_change_times.append(significant_rms_change_times[i])

report_content.write("\n### Potential Major Dynamic Shifts (RMS Energy Changes)\n")
if filtered_rms_change_times:
    report_content.write("  Abrupt and significant changes in loudness are detected at:\n")
    for t in filtered_rms_change_times:
        report_content.write(f"  - {t:.2f} seconds\n")
else:
    report_content.write(
        "  No abrupt major dynamic shifts detected above the current threshold, suggesting a more gradual dynamic progression.\n")

report_content.write(
    "\nThese time markers are computationally derived indicators of structural shifts. Their validity as formal boundaries requires careful aural verification and musical interpretation.\n\n")

# Automated analysis of first segment (e.g., first 30 seconds)
first_segment_end = min(30.0, df['Time_Seconds'].max())
report_content.write(
    analyze_segment_narrative_with_gemini(df, 0.0, first_segment_end, title=f"AI Interpreted Opening Segment"))
report_content.write("\n\n")

report_content.write("--- Limitations and Future Directions ---\n")
report_content.write(
    "This analysis integrates quantitative feature extraction with AI-driven interpretation. While the AI generates sophisticated narratives based on the provided data, it is crucial to acknowledge inherent limitations. The AI lacks true subjective human perception, aural experience, or contextual understanding beyond its training data and explicit prompting. Therefore, the interpretations should be considered as highly informed hypotheses and a robust starting point for deeper musicological inquiry.\n")
report_content.write(
    "Factors such as precise spatialization (unless multi-channel data is processed for specific spatial features), symbolic meaning, and the composer's nuanced intent require human expertise. The quality of AI interpretation is directly dependent on the accuracy and richness of the input features.\n")
report_content.write("Future work could involve:\n")
report_content.write(
    "- Advanced timbral clustering (e.g., t-SNE, UMAP on MFCCs) to visualize timbral 'spaces' and trajectories.\n")
report_content.write("- Integration of psychoacoustic models for more perceptually informed feature extraction.\n")
report_content.write(
    "- Manual segmentation and annotation by human experts to validate and refine algorithmic formal detection.\n")
report_content.write(
    "- Applying AI to identify and describe specific sound types or gestures, potentially using transfer learning from annotated datasets.\n\n")

report_content.write("--- Conclusion ---\n")
report_content.write(
    f"The quantitative analysis of '{file_name}', enhanced by AI-driven narrative interpretation, offers a powerful, data-informed perspective on its dynamic and timbral evolution and potential formal divisions. By diligently combining these objective metrics and AI-generated insights with rigorous aural analysis and theoretical frameworks, a profound and comprehensive understanding of the piece's intricate structure and sonic identity can be achieved.\n\n")

# --- Save Report ---
with open(output_report_name, 'w', encoding='utf-8') as f:
    f.write(report_content.getvalue())
custom_print(f"\nAnalysis report saved to {output_report_name}")

# --- Interactive Segment Analysis (for continued exploration) ---
custom_print("\n--- Interactive Segment Analysis (Console Only - AI Interpreted) ---")
custom_print("You can continue to analyze specific time ranges here, with AI interpretation.")
custom_print("Enter start_time end_time (e.g., '0.0 30.0') or 'q' to quit.")

while True:
    time_input = input("Enter start_time end_time (or 'q' to quit): ")
    if time_input.lower() == 'q':
        break
    try:
        start_time_str, end_time_str = time_input.split()
        start_time = float(start_time_str)
        end_time = float(end_time_str)

        if start_time >= end_time:
            custom_print("End time must be greater than start time.")
            continue
        if start_time < 0 or end_time > df['Time_Seconds'].max() + 0.1:
            custom_print(f"Time range must be within 0.0 and {df['Time_Seconds'].max():.2f} seconds.")
            continue

        custom_print(analyze_segment_narrative_with_gemini(df, start_time, end_time,
                                                           title=f"AI Interpreted Segment ({start_time:.2f}s - {end_time:.2f}s)"))

    except ValueError:
        custom_print("Invalid input. Please enter two numbers separated by a space, or 'q'.")
    except Exception as e:
        custom_print(f"An unexpected error occurred: {e}")

custom_print("\n--- Interactive Analysis Session Concluded ---")
