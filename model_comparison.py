import logging

# Import our new modular utilities
from config import config
from utils import NEREvaluator, DataProcessor, PlotGenerator, StatisticalTester

# Configure logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)







def main() -> None:
    """Main execution function using modular architecture."""
    try:
        # Initialize components
        data_processor = DataProcessor()
        evaluator = NEREvaluator()
        plot_generator = PlotGenerator()
        statistical_tester = StatisticalTester()
        
        logger.info("Starting NER model evaluation...")
        
        # Load and validate data
        logger.info("Loading data...")
        df = data_processor.load_data()
        data_processor.validate_data(df)
        
        # Print data summary
        summary = data_processor.get_data_summary(df)
        logger.info(f"Data summary: {summary['total_rows']} rows, {summary['model_count']} models")
        
        # Evaluate all models
        logger.info("Evaluating models...")
        results = evaluator.evaluate_all_models(df)
        
        if not results:
            logger.error("No evaluation results obtained")
            return
        
        # Create visualizations
        logger.info("Creating visualizations...")
        plot_generator.create_all_comparison_plots(results)
        plot_generator.create_summary_table_plot(results)
        plot_generator.create_heatmap(results)
        
        # Save results
        logger.info("Saving results...")
        data_processor.save_results(results)
        
        # Print summary
        plot_generator.print_summary_table(results)
        
        # Statistical significance tests
        logger.info("Running statistical significance tests...")
        significance_results = statistical_tester.run_configured_significance_tests(df)
        statistical_tester.print_significance_results(significance_results)
        
        # Save significance test results
        if significance_results:
            import json
            
            significance_output = config.OUTPUT_DIR / "significance_test_results.json"
            
            # Convert numpy types to Python types for JSON serialization
            def convert_numpy_types(obj):
                if hasattr(obj, 'item'):  # numpy scalar
                    return obj.item()
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(v) for v in obj]
                else:
                    return obj
            
            serializable_results = convert_numpy_types(significance_results)
            
            with open(significance_output, 'w') as f:
                json.dump(serializable_results, f, indent=4)
            logger.info(f"Significance test results saved to {significance_output}")
        
        logger.info("Evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 