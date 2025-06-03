import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any
import json
import ast

def safe_eval(x: Any) -> List:
    """Safely evaluate string representation of list or return empty list."""
    if pd.isna(x):
        return []
    if isinstance(x, list):
        return x
    try:
        return ast.literal_eval(x)
    except (ValueError, SyntaxError):
        return []

def load_data(file_path: str) -> pd.DataFrame:
    """Load and preprocess the dataset."""
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} rows")
    # Convert string representations of lists to actual lists
    for col in df.columns:
        if col != 'text':  # Skip the text column
            df[col] = df[col].apply(safe_eval)
    return df

def evaluate_ner_performance(df: pd.DataFrame, ground_truth_col: str, model_col: str) -> Tuple[float, float, float]:
    """Evaluate NER model performance using precision, recall, and F1 score."""
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for _, row in df.iterrows():
        try:
            # Get ground truth and predictions
            ground_truth = row[ground_truth_col]
            predictions = row[model_col]

            # Filter for 'PERSON' entities and standardize to 'PERSON' type
            ground_truth = [(start, end, 'PERSON') for start, end, typ in ground_truth if typ in ['PERSON', 'PER']]
            predictions = [(start, end, 'PERSON') for start, end, typ in predictions if typ in ['PERSON', 'PER']]

            # Sort by the start and then end positions
            gt_set = sorted(ground_truth, key=lambda x: (x[0], x[1]))
            pred_set = sorted(predictions, key=lambda x: (x[0], x[1]))

            # Calculate matches allowing for slight boundary mismatches
            matched_gt = set()
            matched_pred = set()

            for gt in gt_set:
                for pred in pred_set:
                    # Check if entities match and boundaries are within tolerance
                    if gt[2] == pred[2] and abs(gt[0] - pred[0]) <= 2 and abs(gt[1] - pred[1]) <= 2:
                        matched_gt.add(gt)
                        matched_pred.add(pred)

            # Update counts
            true_positives += len(matched_gt)
            false_positives += len(pred_set) - len(matched_pred)
            false_negatives += len(gt_set) - len(matched_gt)
        except Exception as e:
            print(f"Warning: Error processing row {row.name}: {str(e)}")
            continue

    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def evaluate_all_models(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Evaluate all models and return their metrics."""
    model_columns = [col for col in df.columns if col not in ['id', 'text', 'labels']]
    results = {}

    for model in model_columns:
        try:
            print(f"\nEvaluating {model}...")
            precision, recall, f1 = evaluate_ner_performance(df, 'labels', model)
            results[model] = {
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            print(f"Precision: {precision:.2f}, Recall: {recall:.2f}, F1 Score: {f1:.2f}")
        except Exception as e:
            print(f"Error evaluating {model}: {str(e)}")
            continue

    return results

def create_comparison_plot(results: Dict[str, Dict[str, float]], metric: str):
    """Create a bar plot comparing models based on the specified metric."""
    models = list(results.keys())
    values = [results[model][metric] for model in models]

    plt.figure(figsize=(12, 6))
    sns.barplot(x=models, y=values)
    plt.title(f'Model Comparison - {metric.capitalize()}')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel(metric.capitalize())
    plt.tight_layout()
    plt.savefig(f'{metric}_comparison.png')
    plt.close()

def save_results(results: Dict[str, Dict[str, float]], output_file: str):
    """Save the evaluation results to a JSON file."""
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def bootstrap_significance_test(df: pd.DataFrame, model1: str, model2: str, metric: str, n_samples: int = 1000) -> Tuple[float, float, float]:
    """
    Perform paired bootstrap resampling to test statistical significance between two models.
    Returns the mean difference, 95% confidence interval, and p-value.
    """
    differences = []
    for _ in range(n_samples):
        # Sample with replacement
        sample_indices = np.random.choice(len(df), size=len(df), replace=True)
        sample_df = df.iloc[sample_indices]
        
        # Compute metric for both models on the sample
        metric1 = evaluate_ner_performance(sample_df, 'labels', model1)[{'precision': 0, 'recall': 1, 'f1': 2}[metric]]
        metric2 = evaluate_ner_performance(sample_df, 'labels', model2)[{'precision': 0, 'recall': 1, 'f1': 2}[metric]]
        
        differences.append(metric2 - metric1)
    
    differences = np.array(differences)
    mean_diff = np.mean(differences)
    ci_low, ci_high = np.percentile(differences, [2.5, 97.5])
    p_value = np.mean(differences <= 0)  # One-sided p-value
    
    return mean_diff, (ci_low, ci_high), p_value

def main():
    try:
        # Load data
        print("Loading data...")
        df = load_data('data/dataset_with_models_1000_03_07_2024.csv')
        print(f"Loaded {len(df)} rows")

        # Evaluate all models
        print("\nEvaluating models...")
        results = evaluate_all_models(df)
        
        # Create comparison plots
        print("\nCreating comparison plots...")
        for metric in ['precision', 'recall', 'f1']:
            create_comparison_plot(results, metric)
        
        # Save results
        print("\nSaving results...")
        save_results(results, 'model_comparison_results.json')
        
        # Print summary table
        print("\nSummary Table:")
        print("Model\t\tPrecision\tRecall\t\tF1 Score")
        print("-" * 50)
        for model, metrics in results.items():
            print(f"{model}\t{metrics['precision']:.2f}\t\t{metrics['recall']:.2f}\t\t{metrics['f1']:.2f}")

        # Statistical significance test between GPT-4o and GPT-4.1
        print("\nStatistical Significance Test (GPT-4o vs GPT-4.1):")
        for metric in ['precision', 'recall', 'f1']:
            mean_diff, (ci_low, ci_high), p_value = bootstrap_significance_test(df, 'gpt4o', 'gpt-4.1-2025-04-14', metric)
            print(f"{metric.capitalize()}: Mean difference = {mean_diff:.4f}, 95% CI = [{ci_low:.4f}, {ci_high:.4f}], p-value = {p_value:.4f}")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")

if __name__ == "__main__":
    main() 