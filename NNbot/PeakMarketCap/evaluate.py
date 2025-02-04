import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

from PeakMarketCap.models.peak_market_cap_model import TokenPricePredictor

def evaluate_token_price_model(data_paths):
    """
    Evaluate token price prediction model showing classification metrics and visualizations.
    """
    # Load and preprocess data
    dfs = []
    for data_path in data_paths:
        df = pd.read_csv(data_path)
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Initialize and train model
    predictor = TokenPricePredictor()
    metrics, importance = predictor.train(train_df)

    # Make predictions on test set
    test_predictions = predictor.predict(test_df)
    
    # Get mask for valid samples
    valid_mask = (
        (test_df['hit_peak_before_30'].astype(str).str.lower() == "false") & 
        (test_df['percent_increase'] > 0)
    )
    
    # Get actual categories for valid samples
    true_values = test_df.loc[valid_mask, 'percent_increase'].values
    true_categories = np.array([predictor._get_increase_category(val) for val in true_values])
    pred_categories = test_predictions.loc[valid_mask, 'predicted_category'].values
    probabilities = test_predictions.loc[valid_mask, ['probability_range_0', 'probability_range_1', 
                                                     'probability_range_2', 'probability_range_3']].values

    # Create figure with subplots
    plt.figure(figsize=(15, 10))
    gs = GridSpec(2, 2)

    # 1. Confusion Matrix
    ax1 = plt.subplot(gs[0, 0])
    cm = confusion_matrix(true_categories, pred_categories)
    sns.heatmap(cm, annot=True, fmt='d', ax=ax1, cmap='Blues')
    ax1.set_title('Confusion Matrix')
    ax1.set_xlabel('Predicted Category')
    ax1.set_ylabel('True Category')

    # 2. Probability Distribution Violin Plot
    ax2 = plt.subplot(gs[0, 1])
    prob_data = []
    labels = []
    for i in range(4):
        prob_data.extend(probabilities[:, i])
        labels.extend([f'Range {i}'] * len(probabilities))
    
    sns.violinplot(x=labels, y=prob_data, ax=ax2)
    ax2.set_title('Prediction Probability Distributions')
    ax2.set_xlabel('Range Category')
    ax2.set_ylabel('Predicted Probability')

    # 3. True vs Predicted Category Distribution
    ax3 = plt.subplot(gs[1, 0])
    df_counts = pd.DataFrame({
        'True': pd.Series(true_categories).value_counts(),
        'Predicted': pd.Series(pred_categories).value_counts()
    }).fillna(0)
    df_counts.plot(kind='bar', ax=ax3)
    ax3.set_title('Category Distribution Comparison')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Count')
    plt.xticks(rotation=45)

    # 4. Most Important Features
    ax4 = plt.subplot(gs[1, 1])
    importance.head(10).plot(x='feature', y='importance', kind='barh', ax=ax4)
    ax4.set_title('Top 10 Most Important Features')
    ax4.set_xlabel('Importance')

    # Create visualizations directory if it doesn't exist
    current_dir = os.path.dirname(os.path.abspath(__file__))
    visualizations_dir = os.path.join(current_dir, 'Visualizations')
    os.makedirs(visualizations_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(visualizations_dir, 'token_price_prediction_performance.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

    # Print evaluation results
    print("\n=== Token Price Classification Model Performance ===")
    print("\nTraining Metrics:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    
    print("\nClassification Report:")
    # Convert classification report dict to readable format
    for label, metrics_dict in metrics['classification_report'].items():
        if isinstance(metrics_dict, dict):  # Skip 'accuracy' and other non-class metrics
            print(f"\nRange {label}:")
            print(f"Precision: {metrics_dict['precision']:.4f}")
            print(f"Recall: {metrics_dict['recall']:.4f}")
            print(f"F1-score: {metrics_dict['f1-score']:.4f}")
            print(f"Support: {metrics_dict['support']}")

    print("\nConfusion Matrix:")
    print(cm)

    print("\nTop 10 Most Important Features:")
    print(importance.head(10))

    return {
        'metrics': metrics,
        'predictions': {
            'true_categories': true_categories,
            'predicted_categories': pred_categories,
            'probabilities': probabilities
        },
        'dataframes': {
            'train_df': train_df,
            'test_df': test_df
        },
        'model': predictor,
        'importance': importance,
        'confusion_matrix': cm
    }

if __name__ == "__main__":
    results = evaluate_token_price_model(['data/new-token-data.csv'])
    print(results)