import librosa
import numpy as np
import pickle
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

class KickHardnessAnalyzer:
    def __init__(self, model_path="models/kick_hardness_model.pkl"):
        """
        Initialize the analyzer with a trained model
        
        Args:
            model_path: Path to the trained model file
        """
        self.model = self.load_model(model_path)
        self.sample_rate = 22050  # Standard sample rate for analysis
    
    def load_model(self, model_path):
        """
        Load a trained machine learning model from disk
        
        Args:
            model_path: Path to the pickled model file
            
        Returns:
            Loaded sklearn model
        """
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return model
        except FileNotFoundError:
            print(f"Warning: Model file {model_path} not found. Using dummy model.")
            # Return a simple dummy model for demonstration
            return DummyModel()
    
    def analyze_file(self, audio_path):
        """
        Analyze a single audio file and return kick hardness score
        
        Args:
            audio_path: Path to audio file (WAV/FLAC/MP3)
            
        Returns:
            float: Kick hardness score between 0.0 and 1.0
        """
        try:
            # Load audio file
            audio_data, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract features
            features = self.extract_features(audio_data, sr)
            
            # Predict kick hardness using trained model
            score = self.model.predict([features])[0]
            
            # Ensure score is between 0 and 1
            score = np.clip(score, 0.0, 1.0)
            
            return float(score)
            
        except Exception as e:
            print(f"Error analyzing {audio_path}: {str(e)}")
            return 0.0
    
    def extract_features(self, audio_data, sr):
        """
        Extract audio features relevant to kick hardness
        
        Args:
            audio_data: numpy array of audio samples
            sr: sample rate
            
        Returns:
            numpy array: Feature vector for ML model
        """
        features = []
        
        # 1. Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0]
        features.append(np.mean(spectral_centroids))  # Average brightness
        features.append(np.std(spectral_centroids))   # Brightness variation
        
        # 2. Zero crossing rate (indicates noisiness/distortion)
        zcr = librosa.feature.zero_crossing_rate(audio_data)[0]
        features.append(np.mean(zcr))
        
        # 3. MFCC coefficients (timbral characteristics)
        mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
        features.extend(np.mean(mfccs, axis=1))  # 13 MFCC coefficients
        
        # 4. Kick-specific frequency analysis (20-200Hz)
        stft = librosa.stft(audio_data)
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Find indices for kick frequency range
        kick_freq_mask = (freqs >= 20) & (freqs <= 200)
        kick_energy = np.mean(np.abs(stft[kick_freq_mask]))
        features.append(kick_energy)
        
        # 5. Onset strength (how pronounced beats are)
        onset_strength = librosa.onset.onset_strength(y=audio_data, sr=sr)
        features.append(np.mean(onset_strength))
        features.append(np.max(onset_strength))
        
        # 6. Tempo and rhythm features
        tempo, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
        features.append(tempo)
        
        # 7. Dynamic range (difference between loud and quiet parts)
        rms = librosa.feature.rms(y=audio_data)[0]
        features.append(np.max(rms) - np.min(rms))
        
        return np.array(features)

class DummyModel:
    """Dummy model for when trained model isn't available"""
    def predict(self, X):
        # Simple heuristic based on spectral centroid and energy
        features = X[0]
        # Higher spectral centroid + higher energy = harder kick
        score = min(1.0, (features[0] / 2000.0) * (features[-1] / 0.1))
        return [score]

