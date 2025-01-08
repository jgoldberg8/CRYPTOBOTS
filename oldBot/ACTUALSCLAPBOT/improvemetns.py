import json
import logging
from collections import defaultdict

class ModelPerformanceAnalyzer:
    def __init__(self):
        self.misclassifications = defaultdict(list)
        self.error_patterns = {
            'confidence_distribution': defaultdict(list),
            'classification_confusion': defaultdict(int)
        }

    def analyze_misclassifications(self, predictions_file, final_analysis_file):
        # Similar to previous comparison script, but with deeper analysis
        predictions = self._read_predictions(predictions_file)
        final_analyses = self._read_final_analysis(final_analysis_file)
        
        for mint, prediction in predictions.items():
            matching_analysis = next(
                (analysis for analysis in final_analyses 
                 if analysis.get('tokenAddress') == mint), 
                None
            )
            
            if not matching_analysis:
                continue
            
            predicted_class = prediction.get('prediction', '').lower()
            actual_class = matching_analysis.get('buy_classification', 'neutral').lower()
            confidence = prediction.get('confidence', 0)
            
            if predicted_class != actual_class:
                # Record misclassification details
                misclassification_details = {
                    'mint': mint,
                    'predicted_class': predicted_class,
                    'actual_class': actual_class,
                    'confidence': confidence,
                    'coin_name': matching_analysis.get('coin_name', 'Unknown'),
                    'market_cap': matching_analysis.get('market_cap'),
                    'volume': matching_analysis.get('volume'),
                    'transaction_count': matching_analysis.get('transaction_count')
                }
                
                # Store misclassification
                self.misclassifications[f"{predicted_class}_to_{actual_class}"].append(misclassification_details)
                
                # Analyze confidence distribution
                self.error_patterns['confidence_distribution'][actual_class].append(confidence)
                
                # Track classification confusion
                self.error_patterns['classification_confusion'][(predicted_class, actual_class)] += 1

    def generate_improvement_recommendations(self):
        """Generate model improvement recommendations based on error analysis."""
        recommendations = []
        
        # Confidence threshold recommendations
        for cls, confidences in self.error_patterns['confidence_distribution'].items():
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            recommendations.append(f"Low confidence in {cls} classification. Average confidence: {avg_confidence:.4f}")
        
        # Classification confusion analysis
        for (predicted, actual), count in self.error_patterns['classification_confusion'].items():
            if count > 0:
                recommendations.append(f"High confusion between {predicted} and {actual}. Occurrences: {count}")
        
        return recommendations

    def _read_predictions(self, filepath):
        # Similar to previous implementation
        predictions = {}
        with open(filepath, 'r') as file:
            for line in file:
                entry = json.loads(line[line.find('{'):])
                if 'mint' in entry:
                    predictions[entry['mint']] = entry
        return predictions

    def _read_final_analysis(self, filepath):
        # Similar to previous implementation
        final_analyses = []
        with open(filepath, 'r') as file:
            for line in file:
                if '[FINAL-ANALYSIS]' in line:
                    entry = json.loads(line[line.find('{'):])
                    final_analyses.append(entry)
        return final_analyses

def main():
    analyzer = ModelPerformanceAnalyzer()
    analyzer.analyze_misclassifications(
        'predictions/predictions_2025-01-05.log', 
        'logs/final-analysis-2025-01-06.log'
    )
    
    # Generate and print recommendations
    recommendations = analyzer.generate_improvement_recommendations()
    for rec in recommendations:
        print(rec)

if __name__ == "__main__":
    main()