# SEMA (Customized Version)

> **Note:** This is a modified fork of the original SEMA project by **@milanmuz**. This version has been altered to focus on a user-friendly, local-only analysis workflow.
>
> The original repository can be found here: **https://github.com/milanmuz/SEMA**

---

## Summary of Changes in this Version

The primary goal of this fork was to streamline the user experience and enhance visualization capabilities, removing the need for API keys and command-line arguments.

* **GUI File Selector:** Replaced the command-line interface with a graphical file dialog (`tkinter`) for easy audio file selection.
* **Removed Gemini AI Integration:** All features related to the Google Gemini API were removed. The script now operates completely offline.
* **Enhanced Visualizations:**
    * **Trend Lines:** Added smoothed moving average lines to all primary plots for better readability of musical contours.
    * **MM:SS Time Format:** The time axis is now displayed in an intuitive `Minutes:Seconds` format.
    * **MFCC Heatmap:** Integrated MFCC extraction and visualization as a heatmap for detailed timbral analysis.

---
*(The original README content from the upstream repository follows below.)*

# SEMA
Statistic Electroacoustic Music Analyzer

It extracts data from audio file into csv file.
Focuses on understanding the piece's formal structure and timbral evolution using Gemini.
Connects quantitative data to qualitative musical observations and concepts.
Defines the features and provide their meaning in an electroacoustic context:
- RMS Energy: Overall loudness/intensity.
- Spectral Centroid: Perceived brightness (higher value = brighter).
- Zero-Crossing Rate (ZCR): Degree of noisiness vs. tonality (higher value = noisier/more percussive, lower = more tonal/sustained).
- Novelty Curve: Rate of significant sonic changes/onsets (peaks indicate 'newness' or activity).
- MFCCs: Mel-Frequency Cepstral Coefficients, a compact representation of timbral color/texture.




