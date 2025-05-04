from error_simulation import ErrorScenarios
from error_analysis import ErrorAnalysis
if __name__ == "__main__":
    # Create instances
    error_scenarios = ErrorScenarios()
    error_analyzer = ErrorAnalysis()
    
    # Simulate high bias scenario
    high_bias_results = error_scenarios.simulate_high_bias()
    
    # Analyze the results
    analysis = error_analyzer.analyze_high_bias(high_bias_results)