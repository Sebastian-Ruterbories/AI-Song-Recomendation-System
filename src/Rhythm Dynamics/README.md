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
