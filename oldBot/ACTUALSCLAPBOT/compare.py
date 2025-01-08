import json
import logging
import re
from typing import Dict, List, Any
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s: %(message)s',
    filename='token_classification_comparison.log'
)

def parse_log_entry(line: str) -> Dict[str, Any]:
    try:
        json_start = line.find('{')
        if json_start == -1:
            return None
        entry = json.loads(line[json_start:])
        return entry
    except Exception as e:
        logging.error(f"Error parsing log entry: {e}")
        return None

def read_predictions(filepath: str) -> Dict[str, Dict[str, Any]]:
    predictions = {}
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                entry = parse_log_entry(line)
                if entry and 'mint' in entry:
                    predictions[entry['mint']] = entry
        return predictions
    except Exception as e:
        logging.error(f"Error reading predictions file: {e}")
        return {}

def read_final_analysis(filepath: str) -> List[Dict[str, Any]]:
    final_analyses = []
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                json_match = re.search(r'\[FINAL-ANALYSIS\]:\s*(\{.*\})', line)
                if json_match:
                    try:
                        entry = json.loads(json_match.group(1))
                        final_analyses.append(entry)
                    except Exception as e:
                        logging.error(f"Error parsing final analysis entry: {e}")
        return final_analyses
    except Exception as e:
        logging.error(f"Error reading final analysis file: {e}")
        return []

def analyze_rug_time_predictions(predictions: Dict[str, Dict[str, Any]], 
                               final_analyses: List[Dict[str, Any]], 
                               final_analysis_lookup: Dict[str, Dict[str, Any]]):
    """Analyze the accuracy of rug time predictions"""
    logging.info("\nRug Time Prediction Analysis")
    logging.info("===========================")

    actual_times = []
    predicted_times = []
    prediction_errors = []
    prediction_details = []

    for mint, prediction in predictions.items():
        if prediction.get('prediction') != 'good':
            continue

        matching_analysis = final_analysis_lookup.get(mint)
        if not matching_analysis:
            continue

        rug_time_prediction = prediction.get('rug_time_prediction')
        actual_rug_time = matching_analysis.get('time_to_rug')

        if rug_time_prediction is None or actual_rug_time is None:
            continue

        # Store actual and predicted times
        actual_times.append(float(actual_rug_time))
        predicted_times.append(float(rug_time_prediction))

        # Calculate error
        error = abs(float(actual_rug_time) - float(rug_time_prediction))
        prediction_errors.append(error)

        # Store detailed prediction information
        prediction_details.append({
            'mint': mint,
            'coin_name': matching_analysis.get('coin_name', 'Unknown'),
            'predicted_time': float(rug_time_prediction),
            'actual_time': float(actual_rug_time),
            'error': error,
            'error_percentage': (error / float(actual_rug_time)) * 100 if float(actual_rug_time) != 0 else float('inf')
        })

    if actual_times and predicted_times:
        # Calculate metrics
        mse = mean_squared_error(actual_times, predicted_times)
        mae = mean_absolute_error(actual_times, predicted_times)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_times, predicted_times)

        # Log overall metrics
        logging.info(f"\nRug Time Prediction Metrics:")
        logging.info(f"Total Predictions Analyzed: {len(actual_times)}")
        logging.info(f"Mean Squared Error: {mse:.2f}")
        logging.info(f"Root Mean Squared Error: {rmse:.2f}")
        logging.info(f"Mean Absolute Error: {mae:.2f}")
        logging.info(f"R² Score: {r2:.4f}")

        # Calculate and log additional statistics
        mean_error = np.mean(prediction_errors)
        median_error = np.median(prediction_errors)
        std_error = np.std(prediction_errors)

        logging.info(f"\nError Statistics:")
        logging.info(f"Mean Error: {mean_error:.2f} seconds")
        logging.info(f"Median Error: {median_error:.2f} seconds")
        logging.info(f"Standard Deviation of Error: {std_error:.2f} seconds")

        # Sort prediction details by error and log worst predictions
        sorted_predictions = sorted(prediction_details, key=lambda x: x['error'], reverse=True)
        logging.info("\nWorst Predictions (Top 10):")
        for pred in sorted_predictions[:10]:
            logging.info(f"Mint: {pred['mint']}")
            logging.info(f"Coin Name: {pred['coin_name']}")
            logging.info(f"Predicted Time: {pred['predicted_time']:.2f} seconds")
            logging.info(f"Actual Time: {pred['actual_time']:.2f} seconds")
            logging.info(f"Absolute Error: {pred['error']:.2f} seconds")
            logging.info(f"Error Percentage: {pred['error_percentage']:.2f}%")
            logging.info("---")

        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mean_error': mean_error,
            'median_error': median_error,
            'std_error': std_error,
            'prediction_details': prediction_details
        }
    else:
        logging.warning("No valid rug time predictions found for analysis")
        return None

def compare_token_classifications(predictions_file: str, final_analysis_file: str):
    predictions = read_predictions(predictions_file)
    final_analyses = read_final_analysis(final_analysis_file)

    final_analysis_lookup = {analysis.get('tokenAddress', ''): analysis for analysis in final_analyses}

    logging.info("Token Classification Comparison Report")
    logging.info("======================================")

    total_tokens = 0
    correct_classifications = 0
    classification_details = {
        'neutral': {'total': 0, 'correct': 0, 'time_since_first_trade': []},
        'bad': {'total': 0, 'correct': 0, 'time_since_first_trade': []},
        'good': {'total': 0, 'correct': 0, 'time_since_first_trade': [], 'false_positives': []}
    }

    for mint, prediction in predictions.items():
        matching_analysis = final_analysis_lookup.get(mint)

        if not matching_analysis:
            logging.warning(f"No token analysis found for prediction: {mint}")
            continue

        total_tokens += 1

        final_classification = matching_analysis.get('buy_classification', 'neutral').lower()
        predicted_class = prediction.get('prediction', '').lower()
        prediction_confidence = prediction.get('confidence', 0)
        coin_name = matching_analysis.get('coin_name', 'Unknown')
        time_since_first_trade = prediction.get('time_since_first_trade', None)

        classification_details[final_classification]['total'] += 1
        
        if time_since_first_trade is not None:
            classification_details[final_classification]['time_since_first_trade'].append(time_since_first_trade)

        if predicted_class == 'good' and final_classification != 'good':
            false_positive_details = {
                'mint': mint,
                'coin_name': coin_name,
                'actual_classification': final_classification,
                'prediction_confidence': prediction_confidence,
                'time_since_first_trade': time_since_first_trade,
                'probabilities': prediction.get('probabilities', {})
            }
            classification_details['good']['false_positives'].append(false_positive_details)
            
            logging.warning("FALSE POSITIVE (Predicted 'Good'):")
            logging.warning(f"  Mint: {mint}")
            logging.warning(f"  Coin Name: {coin_name}")
            logging.warning(f"  Actual Classification: {final_classification}")
            logging.warning(f"  Prediction Confidence: {prediction_confidence:.4f}")
            logging.warning(f"  Time since first trade: {time_since_first_trade:.4f}")
            logging.warning(f"  Probabilities: {prediction.get('probabilities', {})}")
            logging.warning("---")

        if predicted_class == final_classification:
            correct_classifications += 1
            classification_details[final_classification]['correct'] += 1

    # Log classification results
    accuracy = (correct_classifications / total_tokens * 100) if total_tokens > 0 else 0

    logging.info("\nOverall Analysis:")
    logging.info(f"Total Tokens Analyzed: {total_tokens}")
    logging.info(f"Correctly Classified Tokens: {correct_classifications}")
    logging.info(f"Overall Classification Accuracy: {accuracy:.2f}%")

    logging.info("\nClassification Breakdown:")
    for cls, stats in classification_details.items():
        cls_accuracy = (stats['correct'] / stats['total'] * 100) if stats['total'] > 0 else 0
        avg_time_since_first_trade = sum(stats['time_since_first_trade']) / len(stats['time_since_first_trade']) if stats['time_since_first_trade'] else 0

        logging.info(f"\n{cls.upper()} Classification:")
        logging.info(f"  Total Tokens: {stats['total']}")
        logging.info(f"  Correct Classifications: {stats['correct']}")
        logging.info(f"  Classification Accuracy: {cls_accuracy:.2f}%")
        logging.info(f"  Average Time Since First Trade: {avg_time_since_first_trade:.2f}")

    # Analyze rug time predictions
    rug_time_analysis = analyze_rug_time_predictions(predictions, final_analyses, final_analysis_lookup)

    return classification_details, rug_time_analysis

def main():
    predictions_file = 'predictions/predictions_2025-01-07.log'
    final_analysis_file = 'logs/new-data.log'

    classification_details, rug_time_analysis = compare_token_classifications(predictions_file, final_analysis_file)
    print("Analysis complete. Check token_classification_comparison.log for detailed results.")

if __name__ == "__main__":
    main()