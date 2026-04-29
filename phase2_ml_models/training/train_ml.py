"""
Glacier Melting Detection — ML Models Training Script
=====================================================
Trains traditional machine learning models (Linear Regression, Random Forest, etc.).
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.ml_models import train_ml_models

def run_ml_models():
    """Train all traditional ML models and return their results."""
    print("\n" + "═"*60)
    print("  PHASE 2: Traditional Machine Learning Models")
    print("═"*60)
    
    # Train both regression and classification variants
    results = train_ml_models(run_regression=True, run_classification=True)
    return results

if __name__ == "__main__":
    run_ml_models()
