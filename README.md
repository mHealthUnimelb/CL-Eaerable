# CL-Eaerable
# README: Energy-Based Audio Feature Processing and Analysis

## Overview
This repository contains scripts and notebooks designed for processing and analyzing audio recordings for energy-based features using different bandpass filtering configurations. It supports datasets focusing on specific frequency bands (1kHz, 2kHz, 3kHz) and integrates these features with Wav2Vec2 for dependent and independent setups. Additionally, EEG-based energy features are included for advanced audio-energy correlation studies.

---

## Repository Structure

### Scripts
- **`dataset_nobandpass_1k2k3k_extra_feature.py`**  
  Processes audio recordings without bandpass filters for all tones (1kHz, 2kHz, 3kHz) while extracting energy-based features.

  You can also use this script to generate the dataset with bandpass filters through change sf.write(segment) to filter_segment. Detail can be found in the script.
- **`dataset_nobandpass_3k_extra_feature.py`**  
  Processes audio recordings without bandpass filters for the 3kHz tone while extracting energy-based features. (For 3kHz tone, the bandpass filter is not used.)

- **`dataset_2k3k_extra_feature.py`**  
  Processes audio recordings without bandpass filters for the 2kHz and 3kHz tones while extracting energy-based features.
---

### Notebooks
- **`EEGwithenergyfeatures_onedimension_1k2k3k_table3 OAES Final.ipynb`**  
  Analyzes EEG-based features combined with audio energy features for tones 1kHz, 2kHz, and 3kHz.

- **`EEGwithenergyfeatures_onedimension_2k3k_table3 OAES Final.ipynb`**  
  Focuses on EEG and energy feature analysis for 2kHz and 3kHz tones.

- **`EEGwithenergyfeatures_onedimension_3k_table3 OAES Final.ipynb`**  
  Dedicated to EEG-energy feature analysis for the 3kHz tone.

- **`EEG_featureOnly.ipynb`**  
  Processes and analyzes EEG features without integrating audio data.

- **`nobandpass_withoutfeatures 1k2k3k.ipynb`**  
  Processes audio recordings for 1kHz, 2kHz, and 3kHz tones without extracting features.

- **`Wav2Vec2 + Energy-based 1k2k3k Dep.ipynb`**  
  Combines Wav2Vec2 embeddings and energy-based features for dependent setups using 1kHz, 2kHz, and 3kHz tones.

- **`Wav2Vec2 + Energy-based 1k2k3k InDep.ipynb`**  
  Uses Wav2Vec2 embeddings and energy-based features for independent setups with 1kHz, 2kHz, and 3kHz tones.

- **`Wav2Vec2 + Energy-based 2k3k Dep.ipynb`**  
  Focuses on Wav2Vec2 and energy features for 2kHz and 3kHz tones in dependent setups.

- **`Wav2Vec2 + Energy-based 2k3k InDep.ipynb`**  
  Integrates Wav2Vec2 and energy features for independent setups with 2kHz and 3kHz tones.

- **`Wav2Vec2 + Energy-based 3k Dep Exp2.ipynb`**  
  Performs dependent experiments with Wav2Vec2 and energy features for 3kHz tone.

- **`Wav2Vec2 + Energy-based 3k InDep Exp2.ipynb`**  
  Runs independent experiments with Wav2Vec2 embeddings and energy features for 3kHz tone.

---

## Environment Setup
To replicate the results and run the scripts or notebooks, use the provided `environment.yml` file to set up the Python environment.

### Steps:
1. Create a new Conda environment:
   ```bash
   conda env create -f environment.yml
