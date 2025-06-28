import argparse
import json
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from src.kick_detector import KickHardnessAnalyzer

class ProductionModelTrainer:
    """
    Production-ready model trainer for kick hardness prediction
    Ensures reproducibility and meets accuracy requirements
    """
    
    def __init__(self, seed=42):
        """Initialize trainer with fixed seed for reproducibility"""
        self.seed = seed
        np.random.seed(seed)
        self.analyzer = KickHardnessAnalyzer()
        self.model = None
        self.training_results = {}
        
    def load_expert_labels(self, labels_file):
        """
        Load expert-labeled training data
        
        Expected format:
        {
            "song1.wav": 0.85,
            "song2.mp3": 0.23,
            ...
        }
        
        Args:
            labels_file: Path to JSON file with expert labels
            
        Returns:
            dict: Filename -> kick hardness score mapping
        """
        with open(labels_file, 'r') as f:
            labels = json.load(f)
        
        print(f"Loaded {len(labels)} expert labels")
        
        # Validate label format
        for filename, score in labels.items():
            if not isinstance(score, (int, float)) or not (0.0 <= score <= 1.0):
                raise ValueError(f"Invalid score {score} for {filename}. Must be float 0.0-1.0")
        
        return labels
    
    def extract_training_features(self, audio_dir, labels):
        """
        Extract features from all labeled audio files
        
        Args:
            audio_dir: Directory containing training audio files
            labels: Dict mapping filenames to kick hardness scores
            
        Returns:
            tuple: (X, y, filenames) - features, labels, filenames
        """
        audio_dir = Path(audio_dir)
        X, y, filenames = [], [], []
        
        print(f"Extracting features from {len(labels)} files...")
        
        for filename, score in labels.items():
            audio_path = audio_dir / filename
            
            if not audio_path.exists():
                print(f"Warning: Audio file not found: {audio_path}")
                continue
            
            try:
                print(f"Processing: {filename}")
                _, features = self.analyzer.analyze_file(audio_path, return_features=True)
                
                X.append(features)
                y.append(score)
                filenames.append(filename)
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"Successfully extracted features from {len(X)} files")
        print(f"Feature matrix shape: {X.shape}")
        
        return X, y, filenames
    
    def optimize_hyperparameters(self, X, y):
        """
        Find optimal hyperparameters using grid search
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            dict: Best hyperparameters
        """
        print("Optimizing hyperparameters...")
        
        # Define parameter grid
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Use smaller grid for faster training if dataset is small
        if len(X) < 50:
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 15],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', None]
            }
        
        rf = RandomForestRegressor(random_state=self.seed)
        
        # Use 3-fold CV for small datasets, 5-fold for larger
        cv_folds = 3 if len(X) < 100 else 5
        
        grid_search = GridSearchCV(
            rf, param_grid, 
            cv=cv_folds, 
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
            random_state=self.seed
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_params_
    
    def train_model(self, X, y, hyperparams=None):
        """
        Train the kick hardness model
        
        Args:
            X: Feature matrix
            y: Target labels
            hyperparams: Optional hyperparameters dict
            
        Returns:
            dict: Training results and metrics
        """
        print(f"Training model with {len(X)} samples and {X.shape[1]} features...")
        
        # Split data with fixed seed for reproducibility
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.seed, stratify=None
        )
        
        # Use optimized hyperparameters or defaults
        if hyperparams is None:
            hyperparams = {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'max_features': 'sqrt',
                'random_state': self.seed
            }
        else:
            hyperparams['random_state'] = self.seed
        
        # Train model
        self.model = RandomForestRegressor(**hyperparams)
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)
        
        # Calculate metrics
        train_r2 = r2_score(y_train, train_pred)
        test_r2 = r2_score(y_test, test_pred)
        train_mae = mean_absolute_error(y_train, train_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Spearman correlation (key requirement)
        train_spearman, _ = spearmanr(y_train, train_pred)
        test_spearman, _ = spearmanr(y_test, test_pred)
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=5, random_state=self.seed)
        cv_spearman_scores = []
        
        # Calculate Spearman correlation for each CV fold
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=5, shuffle=True, random_state=self.seed)
        for train_idx, val_idx in kf.split(X):
            fold_model = RandomForestRegressor(**hyperparams)
            fold_model.fit(X[train_idx], y[train_idx])
            fold_pred = fold_model.predict(X[val_idx])
            fold_spearman, _ = spearmanr(y[val_idx], fold_pred)
            cv_spearman_scores.append(fold_spearman)
        
        cv_spearman_mean = np.mean(cv_spearman_scores)
        cv_spearman_std = np.std(cv_spearman_scores)
        
        # Store results
        results = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_spearman': train_spearman,
            'test_spearman': test_spearman,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'cv_spearman_mean': cv_spearman_mean,
            'cv_spearman_std': cv_spearman_std,
            'hyperparameters': hyperparams,
            'feature_count': X.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'seed': self.seed
        }
        
        self.training_results = results
        
        # Print results
        print("\n=== TRAINING RESULTS ===")
        print(f"Training R²: {train_r2:.4f}")
        print(f"Test R²: {test_r2:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Test MAE: {test_mae:.4f}")
        print(f"Training Spearman ρ: {train_spearman:.4f}")
        print(f"Test Spearman ρ: {test_spearman:.4f}")
        print(f"CV Spearman ρ: {cv_spearman_mean:.4f} ± {cv_spearman_std:.4f}")
        
        # Check if model meets requirements
        if cv_spearman_mean >= 0.85:
            print("✅ Model meets Spearman ρ ≥ 0.85 requirement!")
        else:
            print(f"⚠️  Model does not meet Spearman ρ ≥ 0.85 requirement (got {cv_spearman_mean:.4f})")
            print("Consider:")
            print("- Adding more diverse training data")
            print("- Improving label quality")
            print("- Feature engineering")
        
        return results
    
    def generate_plots(self, X, y):
        """Generate analysis plots"""
        if self.model is None:
            print("No model trained yet")
            return
        
        # Predict on full dataset for plotting
        y_pred = self.model.predict(X)
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Prediction vs Actual
        axes[0, 0].scatter(y, y_pred, alpha=0.6)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Kick Hardness')
        axes[0, 0].set_ylabel('Predicted Kick Hardness')
        axes[0, 0].set_title('Predictions vs Actual Values')
        
        # Add correlation info
        spearman_corr, _ = spearmanr(y, y_pred)
        axes[0, 0].text(0.05, 0.95, f'Spearman ρ = {spearman_corr:.3f}', 
                       transform=axes[0, 0].transAxes, fontsize=12)
        
        # 2. Feature importance
        feature_names = self.analyzer.feature_names
        importances = self.model.feature_importances_
        indices = np.argsort(importances)[::-1][:15]  # Top 15 features
        
        axes[0, 1].bar(range(15), importances[indices])
        axes[0, 1].set_title('Top 15 Feature Importances')
        axes[0, 1].set_xticks(range(15))
        axes[0, 1].set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right')
        
        # 3. Residuals plot
        residuals = y - y_pred
        axes[1, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Values')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residual Plot')
        
        # 4. Distribution of predictions
        axes[1, 1].hist(y, bins=20, alpha=0.5, label='Actual', density=True)
        axes[1, 1].hist(y_pred, bins=20, alpha=0.5, label='Predicted', density=True)
        axes[1, 1].set_xlabel('Kick Hardness Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].set_title('Distribution Comparison')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('kick_hardness_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Analysis plots saved as 'kick_hardness_analysis.png'")
    
    def save_model(self, filepath='models/kick_hardness_model.pkl'):
        """Save the trained model and metadata"""
        if self.model is None:
            print("No model to save")
            return
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        with open(filepath,
