"""
Data processing utilities for NER evaluation.
"""
import pandas as pd
import logging
from pathlib import Path
from typing import List, Any
from config import config
from .common import safe_eval_list

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data loading and preprocessing operations."""
    
    def __init__(self):
        """Initialize data processor."""
        config.ensure_directories()
    
    def safe_eval(self, x: Any) -> List:
        """Safely evaluate string representation of list or return empty list."""
        return safe_eval_list(x, "data")
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load and preprocess the dataset.
        
        Args:
            file_path: Path to dataset file (uses config default if None)
            
        Returns:
            Loaded and preprocessed DataFrame
            
        Raises:
            FileNotFoundError: If dataset file doesn't exist
            ValueError: If dataset is empty or invalid
        """
        if file_path is None:
            file_path = config.get_dataset_path()
        else:
            file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded {len(df)} rows from {file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError(f"Empty data file: {file_path}")
        except Exception as e:
            raise ValueError(f"Error reading data file {file_path}: {e}")
        
        if df.empty:
            raise ValueError("Loaded DataFrame is empty")
        
        return self.preprocess_data(df)
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the loaded data.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing data...")
        
        # Convert string representations of lists to actual lists
        model_columns = config.get_model_columns(df)
        
        for col in model_columns + ['labels']:
            if col in df.columns:
                logger.info(f"Processing column: {col}")
                df[col] = df[col].apply(self.safe_eval)
        
        # Validate essential columns
        required_columns = ['labels']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        logger.info(f"Preprocessing complete. Available model columns: {model_columns}")
        return df
    
    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        Validate the loaded data structure.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            True if data is valid
            
        Raises:
            ValueError: If data validation fails
        """
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        if 'labels' not in df.columns:
            raise ValueError("Ground truth 'labels' column missing")
        
        model_columns = config.get_model_columns(df)
        if not model_columns:
            raise ValueError("No model columns found")
        
        # Check for reasonable data distribution
        non_empty_labels = df['labels'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
        if non_empty_labels == 0:
            raise ValueError("No ground truth labels found")
        
        logger.info(f"Data validation passed: {len(df)} rows, {len(model_columns)} models, {non_empty_labels} labeled rows")
        return True
    
    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics for the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with summary statistics
        """
        model_columns = config.get_model_columns(df)
        
        summary = {
            'total_rows': len(df),
            'model_count': len(model_columns),
            'models': model_columns,
            'columns': list(df.columns)
        }
        
        # Count non-empty labels
        if 'labels' in df.columns:
            non_empty_labels = df['labels'].apply(lambda x: len(x) > 0 if isinstance(x, list) else False).sum()
            summary['labeled_rows'] = non_empty_labels
            summary['unlabeled_rows'] = len(df) - non_empty_labels
        
        # Count entities per model
        entity_counts = {}
        for model in model_columns:
            if model in df.columns:
                total_entities = df[model].apply(lambda x: len(x) if isinstance(x, list) else 0).sum()
                entity_counts[model] = total_entities
        
        summary['entity_counts'] = entity_counts
        
        return summary
    
    def save_results(self, results: dict, output_file: str = None) -> None:
        """
        Save evaluation results to JSON file.
        
        Args:
            results: Results dictionary to save
            output_file: Output file path (uses config default if None)
        """
        import json
        
        if output_file is None:
            output_file = config.RESULTS_JSON
        else:
            output_file = Path(output_file)
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)
            logger.info(f"Results saved to {output_file}")
        except Exception as e:
            logger.error(f"Error saving results to {output_file}: {e}")
            raise