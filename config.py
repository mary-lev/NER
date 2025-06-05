"""
Configuration settings for NER model evaluation project.
"""
from pathlib import Path
from typing import List


class Config:
    """Centralized configuration for NER evaluation project."""
    
    # Project paths
    PROJECT_ROOT = Path(__file__).parent
    DATA_DIR = PROJECT_ROOT / "data"
    OUTPUT_DIR = PROJECT_ROOT
    
    # Data file paths
    DEFAULT_DATASET = DATA_DIR / "dataset_with_models_1000_03_07_2024.csv"
    ALTERNATIVE_DATASET = DATA_DIR / "dataset_with_models_1000_31_05_2024.csv"
    
    # Output file paths
    RESULTS_JSON = OUTPUT_DIR / "model_comparison_results.json"
    PRECISION_PLOT = OUTPUT_DIR / "precision_comparison.png"
    RECALL_PLOT = OUTPUT_DIR / "recall_comparison.png"
    F1_PLOT = OUTPUT_DIR / "f1_comparison.png"
    
    # Model evaluation parameters
    BOUNDARY_TOLERANCE = 2
    BOOTSTRAP_SAMPLES = 1000
    CONFIDENCE_LEVEL = 0.95
    CONFIDENCE_INTERVAL = [2.5, 97.5]  # For 95% CI
    
    # Entity types
    PERSON_ENTITY_TYPES = ['PERSON', 'PER']
    
    # Model columns (order matters for some operations)
    MODEL_COLUMNS = [
        'mult_model',
        'rus_ner_model', 
        'roberta_large',
        'spacy',
        'gpt3.5-31-05-2024',
        'gpt4',
        'gpt4o',
        'gpt4o_json',
        'gpt-4.1-2025-04-14',
        'mistral'
    ]
    
    # Statistical significance test pairs
    SIGNIFICANCE_TEST_PAIRS = [
        ('gpt4o', 'gpt-4.1-2025-04-14'),
        # ('gpt4o_json', 'gpt-4.1-2025-04-14')
    ]
    
    # Metrics to evaluate
    METRICS = ['precision', 'recall', 'f1']
    
    
    # Visualization parameters
    PLOT_FIGSIZE = (12, 6)
    PLOT_DPI = 300
    PLOT_Y_LIMIT = 1.1
    TABLE_FONT_SIZE = 10
    
    # Statistical testing parameters
    MIN_BOOTSTRAP_SAMPLES = 20
    BOOTSTRAP_RELIABILITY_THRESHOLD = 0.8
    
    # Logging configuration
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    @classmethod
    def get_dataset_path(cls, prefer_latest: bool = True) -> Path:
        """Get the appropriate dataset path."""
        if prefer_latest and cls.DEFAULT_DATASET.exists():
            return cls.DEFAULT_DATASET
        elif cls.ALTERNATIVE_DATASET.exists():
            return cls.ALTERNATIVE_DATASET
        else:
            raise FileNotFoundError("No valid dataset file found")
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_model_columns(cls, df) -> List[str]:
        """Get available model columns from DataFrame."""
        available_columns = [col for col in cls.MODEL_COLUMNS if col in df.columns]
        if not available_columns:
            # Fallback: detect model columns automatically
            available_columns = [col for col in df.columns 
                               if col not in ['id', 'text', 'labels']]
        return available_columns


class NotebookConfig(Config):
    """Configuration specific to Jupyter notebook environment."""
    
    # Google Colab paths (if running in Colab)
    COLAB_DRIVE_PATH = "/content/drive/My Drive/data/litgid/"
    
    @classmethod
    def is_colab_environment(cls) -> bool:
        """Check if running in Google Colab."""
        try:
            import google.colab
            return True
        except ImportError:
            return False
    
    @classmethod
    def get_base_path(cls) -> str:
        """Get appropriate base path depending on environment."""
        if cls.is_colab_environment():
            return cls.COLAB_DRIVE_PATH
        else:
            return str(cls.DATA_DIR)


# Environment-specific configurations
def get_config() -> Config:
    """Get appropriate configuration based on environment."""
    try:
        import google.colab
        return NotebookConfig()
    except ImportError:
        return Config()


# Global configuration instance
config = get_config()