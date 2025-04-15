# voice-cognition
# Cognitive Decline Detection from Voice Data - Proof of Concept

## Overview

This project is a proof-of-concept (POC) for analyzing voice data to detect potential indicators of cognitive stress or decline. It uses simulated voice samples, extracts relevant features, and applies a simple unsupervised machine learning approach to identify potentially abnormal samples.

## Features

* **Simulated Voice Data Generation:** Generates simulated voice data with variations to mimic cognitive decline indicators.
* **Feature Extraction:** Extracts the following features from the audio and/or transcribed text:
    * Pauses per sentence
    * Hesitation markers (uh, um, etc.)
    * Word recall issues (simulated)
    * Speech rate, pitch variability
    * Naming & Word-Association Tasks (simulated)
    * Sentence Completion (simulated)
* **Unsupervised Modeling:** Calculates similarity scores between voice samples using cosine similarity and identifies potentially abnormal samples based on a similarity threshold.
* **Reporting:** Generates a report summarizing the analysis, including insights into the most informative features.
* **Visualization**: Generates visualization of feature trends and similarity matrix.
* **API-Ready Function (Optional):** Provides a function to analyze a single voice clip and return a risk score.

## Dependencies

The project uses the following Python libraries:

* librosa
* soundfile
* numpy
* scikit-learn
* matplotlib
* pandas

## Installation

1.  Clone the repository.
2.  Install the required libraries using pip:

    ```bash
    pip install librosa soundfile scikit-learn matplotlib pandas
    ```

## Usage

1.  Run the Python notebook (cognitive\_decline\_detection\_poc.ipynb).  The notebook will:
    * Generate simulated voice data.
    * Extract features from the simulated data.
    * Calculate similarity scores.
    * Identify potentially abnormal samples.
    * Generate a report summarizing the analysis.
    * Generate visualizations.
    * (Optionally) test the API-ready function.

2.  (Optional) To analyze a single voice clip using the `analyze_voice_clip` function:
    * Ensure you have a WAV audio file.
    * Modify the `if __name__ == "__main__":` block to call the  `analyze_voice_clip`  function with the path to your audio file.
    * Run the script.

## Code Description

### 1. Data Acquisition and Preprocessing

* `generate_simulated_voice_data(num_samples=5, base_text="The quick brown fox jumps over the lazy dog.")`:
    * Generates simulated voice data with cognitive decline indicators.
    * Returns a dictionary containing audio data, transcriptions, and sample IDs.
* `insert_pauses(text, pause_duration_ms=200)`: Inserts pauses into the text.
* `insert_hesitations(text, hesitation_markers=["uh", "um", "er"])`: Inserts hesitation markers.
* `substitute_words(text, substitution_prob=0.2)`: Substitutes words with similar-sounding words.
* `incomplete_sentence(text, removal_prob=0.4)`: Simulates incomplete sentences.

### 2. Feature Extraction

* `extract_features(audio_data, transcriptions)`:
    * Extracts features from the audio data and transcriptions.
    * Returns a dictionary where keys are sample IDs and values are dictionaries of extracted features.

### 3. Modeling

* `calculate_similarity(features)`:
    * Calculates the similarity between voice samples using cosine similarity.
    * Returns a Pandas DataFrame containing the similarity matrix.
* `identify_abnormal_samples(similarity_df, threshold=0.8)`:
    * Identifies potentially abnormal samples based on their similarity to others.
    * Returns a list of sample IDs considered abnormal.

### 4. Reporting

* `generate_report(features, similarity_df, abnormal_samples)`:
    * Generates a short report summarizing the analysis.
    * Returns the report text.
* `generate_visualization(similarity_df, features)`:
    * Generates visualizations of feature trends and the similarity matrix.
    * Returns a list of base64 encoded PNG images.

### 5. API-Ready Function (Optional)

* `analyze_voice_clip(audio_file_path, threshold=0.8)`:
    * Analyzes a single voice clip (WAV file) and returns a risk score.
    * Returns a dictionary containing the risk score (0-1) and a message.

## Potential Next Steps

The report suggests several next steps to make this approach clinically robust, including:

* Larger and more realistic datasets
* Advanced feature engineering
* Improved transcription
* Clinically validated features
* Supervised machine learning
* Longitudinal analysis
* Integration with other data
* Rigorous evaluation
* API Integration

## Disclaimer

This is a proof-of-concept and is not intended for clinical use. Further research and development are needed to create a reliable and clinically useful tool.
