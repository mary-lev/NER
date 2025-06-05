"""
NER model evaluation utilities.
"""
import pandas as pd
import logging
from typing import List, Tuple, Dict, Any, Set
from config import config
from .common import safe_eval_list

logger = logging.getLogger(__name__)


class NEREvaluator:
    """Handles NER model evaluation with configurable parameters."""
    
    def __init__(self, tolerance: int = None):
        """
        Initialize evaluator.
        
        Args:
            tolerance: Boundary tolerance for entity matching (default from config)
        """
        self.tolerance = tolerance or config.BOUNDARY_TOLERANCE
        self.person_types = config.PERSON_ENTITY_TYPES
    
    def safe_eval(self, x: Any) -> List:
        """Safely evaluate string representation of list or return empty list."""
        return safe_eval_list(x, "entity list")
    
    def normalize_entities(self, entities: List[Tuple], entity_type: str = 'PERSON') -> List[Tuple]:
        """
        Normalize entities to consistent format and filter by type.
        
        Args:
            entities: List of (start, end, type) tuples
            entity_type: Entity type to filter for
            
        Returns:
            Normalized and filtered entity list
        """
        if not entities:
            return []
        
        normalized = []
        for entity in entities:
            if len(entity) >= 3:
                start, end, ent_type = entity[:3]
                if ent_type in self.person_types:
                    normalized.append((start, end, entity_type))
            else:
                logger.warning(f"Malformed entity: {entity}")
        
        return sorted(normalized, key=lambda x: (x[0], x[1]))
    
    def entities_match(self, gt_entity: Tuple, pred_entity: Tuple) -> bool:
        """
        Check if two entities match within tolerance.
        
        Args:
            gt_entity: Ground truth entity (start, end, type)
            pred_entity: Predicted entity (start, end, type)
            
        Returns:
            True if entities match within tolerance
        """
        gt_start, gt_end, gt_type = gt_entity
        pred_start, pred_end, pred_type = pred_entity
        
        return (gt_type == pred_type and 
                abs(gt_start - pred_start) <= self.tolerance and 
                abs(gt_end - pred_end) <= self.tolerance)
    
    def find_entity_matches(self, gt_entities: List[Tuple], pred_entities: List[Tuple]) -> Tuple[Set, Set]:
        """
        Find matching entities between ground truth and predictions.
        
        Args:
            gt_entities: Ground truth entities
            pred_entities: Predicted entities
            
        Returns:
            Tuple of (matched_ground_truth, matched_predictions) sets
        """
        matched_gt = set()
        matched_pred = set()
        
        for gt in gt_entities:
            for pred in pred_entities:
                if self.entities_match(gt, pred):
                    matched_gt.add(gt)
                    matched_pred.add(pred)
                    break  # Each ground truth entity matches at most one prediction
        
        return matched_gt, matched_pred
    
    def calculate_metrics(self, true_positives: int, false_positives: int, false_negatives: int) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score.
        
        Args:
            true_positives: Number of true positive matches
            false_positives: Number of false positive predictions
            false_negatives: Number of false negative misses
            
        Returns:
            Dictionary with precision, recall, and f1 scores
        """
        precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def evaluate_model_performance(self, df: pd.DataFrame, ground_truth_col: str, model_col: str, quiet: bool = False) -> Dict[str, float]:
        """
        Evaluate NER model performance using precision, recall, and F1 score.
        
        Args:
            df: DataFrame containing evaluation data
            ground_truth_col: Column name for ground truth labels
            model_col: Column name for model predictions
            quiet: If True, suppress verbose logging (for bootstrap sampling)
            
        Returns:
            Dictionary with evaluation metrics
            
        Raises:
            ValueError: If required columns are missing
            KeyError: If specified columns don't exist in DataFrame
        """
        if ground_truth_col not in df.columns:
            raise KeyError(f"Ground truth column '{ground_truth_col}' not found")
        if model_col not in df.columns:
            raise KeyError(f"Model column '{model_col}' not found")
        
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        processed_rows = 0
        
        for _, row in df.iterrows():
            try:
                # Get and normalize entities
                ground_truth = self.safe_eval(row[ground_truth_col])
                predictions = self.safe_eval(row[model_col])
                
                gt_entities = self.normalize_entities(ground_truth)
                pred_entities = self.normalize_entities(predictions)
                
                # Find matches
                matched_gt, matched_pred = self.find_entity_matches(gt_entities, pred_entities)
                
                # Update counts
                true_positives += len(matched_gt)
                false_positives += len(pred_entities) - len(matched_pred)
                false_negatives += len(gt_entities) - len(matched_gt)
                
                processed_rows += 1
                
            except Exception as e:
                logger.warning(f"Error processing row {row.name}: {str(e)}")
                continue
        
        if processed_rows == 0:
            raise ValueError("No valid rows processed")
        
        if not quiet:
            logger.info(f"Processed {processed_rows} rows for {model_col}")
            logger.info(f"TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
        
        return self.calculate_metrics(true_positives, false_positives, false_negatives)
    
    def evaluate_all_models(self, df: pd.DataFrame, ground_truth_col: str = 'labels') -> Dict[str, Dict[str, float]]:
        """
        Evaluate all available models in the DataFrame.
        
        Args:
            df: DataFrame containing evaluation data
            ground_truth_col: Column name for ground truth labels
            
        Returns:
            Dictionary mapping model names to their evaluation metrics
        """
        model_columns = config.get_model_columns(df)
        results = {}
        
        logger.info(f"Evaluating {len(model_columns)} models: {model_columns}")
        
        for model in model_columns:
            try:
                logger.info(f"Evaluating {model}...")
                metrics = self.evaluate_model_performance(df, ground_truth_col, model)
                results[model] = metrics
                logger.info(f"{model}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1']:.3f}")
            except Exception as e:
                logger.error(f"Error evaluating {model}: {str(e)}")
                continue
        
        return results