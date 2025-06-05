"""
Statistical testing utilities for NER evaluation.
"""
import numpy as np
import pandas as pd
import logging
from typing import Tuple, List
from config import config
from .evaluation import NEREvaluator

logger = logging.getLogger(__name__)


class StatisticalTester:
    """Handles statistical significance testing for model comparisons."""
    
    def __init__(self, n_samples: int = None):
        """
        Initialize statistical tester.
        
        Args:
            n_samples: Number of bootstrap samples (default from config)
        """
        self.n_samples = n_samples or config.BOOTSTRAP_SAMPLES
        self.evaluator = NEREvaluator()
        self.confidence_level = config.CONFIDENCE_LEVEL
        
    def bootstrap_significance_test(self, df: pd.DataFrame, model1: str, model2: str, 
                                  metric: str) -> Tuple[float, Tuple[float, float], float]:
        """
        Perform paired bootstrap resampling for statistical significance testing.
        
        Uses bootstrap resampling to estimate the sampling distribution of the
        difference in performance metrics between two NER models.
        
        Args:
            df: DataFrame containing evaluation data
            model1: First model column name
            model2: Second model column name  
            metric: Performance metric ('precision', 'recall', 'f1')
            
        Returns:
            Tuple containing:
            - mean_diff: Mean difference in performance (model2 - model1)
            - confidence_interval: 95% confidence interval as (lower, upper)
            - p_value: One-sided p-value for H0: model2 <= model1
            
        Raises:
            KeyError: If specified models or metric not found in data
            ValueError: If insufficient data for reliable bootstrap estimation
        """
        if model1 not in df.columns:
            raise KeyError(f"Model '{model1}' not found in DataFrame")
        if model2 not in df.columns:
            raise KeyError(f"Model '{model2}' not found in DataFrame")
        if metric not in config.METRICS:
            raise ValueError(f"Invalid metric: {metric}. Must be one of {config.METRICS}")
        
        if len(df) < 10:
            raise ValueError("Insufficient data for reliable bootstrap estimation (need at least 10 samples)")
        
        logger.info(f"Running bootstrap test: {model2} vs {model1} on {metric} ({self.n_samples} samples)")
        
        differences = []
        metric_index = {'precision': 0, 'recall': 1, 'f1': 2}[metric]
        
        for i in range(self.n_samples):
            # Sample with replacement
            sample_indices = np.random.choice(len(df), size=len(df), replace=True)
            sample_df = df.iloc[sample_indices].copy()
            
            try:
                # Compute metric for both models on the sample (quiet mode for bootstrap)
                result1 = self.evaluator.evaluate_model_performance(sample_df, 'labels', model1, quiet=True)
                result2 = self.evaluator.evaluate_model_performance(sample_df, 'labels', model2, quiet=True)
                
                metric1 = result1[metric]
                metric2 = result2[metric]
                
                differences.append(metric2 - metric1)
                
            except Exception as e:
                logger.warning(f"Error in bootstrap sample {i}: {e}")
                continue
            
            # Progress logging (every 25%)
            if (i + 1) % (self.n_samples // 4) == 0:
                logger.info(f"Bootstrap progress: {i + 1}/{self.n_samples} ({((i + 1) / self.n_samples * 100):.0f}%)")
        
        if len(differences) < self.n_samples * 0.8:  # Less than 80% success rate
            logger.warning(f"Only {len(differences)}/{self.n_samples} bootstrap samples succeeded")
        
        if not differences:
            raise ValueError("All bootstrap samples failed")
        
        differences = np.array(differences)
        
        # Calculate statistics
        mean_diff = np.mean(differences)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = (alpha/2) * 100
        ci_upper = (1 - alpha/2) * 100
        ci_low, ci_high = np.percentile(differences, [ci_lower, ci_upper])
        
        # One-sided p-value (H0: model2 <= model1, H1: model2 > model1)
        p_value = np.mean(differences <= 0)
        
        logger.info(f"Bootstrap test results: mean_diff={mean_diff:.4f}, "
                   f"CI=[{ci_low:.4f}, {ci_high:.4f}], p={p_value:.4f}")
        
        return mean_diff, (ci_low, ci_high), p_value
    
    def run_configured_significance_tests(self, df: pd.DataFrame) -> dict:
        """
        Run significance tests for configured model pairs.
        
        Args:
            df: DataFrame containing evaluation data
            
        Returns:
            Dictionary with test results for each pair and metric
        """
        results = {}
        
        logger.info(f"Running significance tests for {len(config.SIGNIFICANCE_TEST_PAIRS)} model pairs")
        
        for model1, model2 in config.SIGNIFICANCE_TEST_PAIRS:
            if model1 not in df.columns or model2 not in df.columns:
                logger.warning(f"Skipping test for {model1} vs {model2}: models not found")
                continue
            
            pair_key = f"{model1}_vs_{model2}"
            results[pair_key] = {}
            
            for metric in config.METRICS:
                try:
                    mean_diff, (ci_low, ci_high), p_value = self.bootstrap_significance_test(
                        df, model1, model2, metric
                    )
                    
                    results[pair_key][metric] = {
                        'mean_difference': mean_diff,
                        'confidence_interval': [ci_low, ci_high],
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'confidence_level': self.confidence_level
                    }
                    
                except Exception as e:
                    logger.error(f"Error in significance test for {pair_key} on {metric}: {e}")
                    results[pair_key][metric] = {
                        'error': str(e)
                    }
        
        return results
    
    def print_significance_results(self, results: dict) -> None:
        """
        Print formatted significance test results.
        
        Args:
            results: Results from run_configured_significance_tests
        """
        if not results:
            logger.warning("No significance test results to display")
            return
        
        print("\n" + "="*80)
        print("STATISTICAL SIGNIFICANCE TEST RESULTS")
        print("="*80)
        
        for pair_key, pair_results in results.items():
            print(f"\n{pair_key.replace('_vs_', ' vs ').upper()}:")
            print("-" * 50)
            
            for metric, metric_results in pair_results.items():
                if 'error' in metric_results:
                    print(f"{metric.capitalize()}: ERROR - {metric_results['error']}")
                    continue
                
                mean_diff = metric_results['mean_difference']
                ci_low, ci_high = metric_results['confidence_interval']
                p_value = metric_results['p_value']
                significant = metric_results['significant']
                
                significance_text = "SIGNIFICANT" if significant else "not significant"
                
                print(f"{metric.capitalize()}: "
                      f"Mean difference = {mean_diff:.4f}, "
                      f"95% CI = [{ci_low:.4f}, {ci_high:.4f}], "
                      f"p-value = {p_value:.4f} ({significance_text})")
        
        print("="*80)
    
    def compare_two_models(self, df: pd.DataFrame, model1: str, model2: str) -> dict:
        """
        Compare two specific models across all metrics.
        
        Args:
            df: DataFrame containing evaluation data
            model1: First model name
            model2: Second model name
            
        Returns:
            Dictionary with comparison results
        """
        results = {}
        
        for metric in config.METRICS:
            try:
                mean_diff, ci, p_value = self.bootstrap_significance_test(df, model1, model2, metric)
                results[metric] = {
                    'mean_difference': mean_diff,
                    'confidence_interval': ci,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            except Exception as e:
                logger.error(f"Error comparing {model1} vs {model2} on {metric}: {e}")
                results[metric] = {'error': str(e)}
        
        return results