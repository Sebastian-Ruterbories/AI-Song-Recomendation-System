Result-Oriented Deliverable: 
  A trained, version-controlled model and CLI tool that:
  1. Takes one or more audio files (WAV/FLAC/MP3).
  2. Computes a kick-hardness score for each track.
  3. Renames or copies each file with the score prepended, e.g. 0.73__originalTitle.wav.

Acceptance Criteria (all must pass):
  Accuracy – Spearman ρ ≥ 0.85 vs. expert rankings on a 100-track blind test.
  Filename output – For every processed file:  
    • Tool prefixes the score rounded to two decimals (0.00–1.00). 
  Reproducibility – Running python train_kick_hardness.py --seed 42 recreates model within ±0.02 ρ of published result. 
  Documentation – kick_hardness_report.md explains feature extraction (e.g., librosa), model architecture, hyper-parameters, and exact CLI usage examples.

Notes:
  Artifacts to hand off: 
    • kick_hardness_model.pkl (or equivalent)
    • train_kick_hardness.py – fully reproducible training script
    • score_and_rename.py – CLI: python score_and_rename.py /path/to/folder [--copy-dir outdir]
    • requirements.txt or environment.yml
    • kick_hardness_report.md – data, features, model, evaluation



AI Sample Process: 
# Example usage and function calls:

# 1. Create analyzer instance
# analyzer = KickHardnessAnalyzer("models/kick_hardness_model.pkl")

# 2. Analyze a single file
# score = analyzer.analyze_file("path/to/song.wav")
# print(f"Kick hardness score: {score}")

# 3. CLI usage examples:
# python src/cli.py song1.wav song2.mp3 song3.flac
# python src/cli.py /path/to/music/folder --recursive --copy --output-dir /path/to/output
# python src/cli.py *.wav --model models/custom_model.pkl

# 4. Extract features directly (for training/debugging)
# audio_data, sr = librosa.load("song.wav", sr=22050)
# features = analyzer.extract_features(audio_data, sr)
# print(f"Feature vector shape: {features.shape}")

# 5. Train your own model (run once to create initial model)
# python train_model.py
