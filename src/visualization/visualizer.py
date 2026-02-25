"""
Visualization Module
Creates comprehensive visualizations including confusion matrix, ROC curves, scatter plots
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score
)
import cv2
from pathlib import Path
import json

class WildfireVisualizer:
    """
    Comprehensive visualization tools for wildfire detection and prediction
    """
    
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names, save_name="confusion_matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', 
                   xticklabels=class_names, yticklabels=class_names,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix - Wildfire Detection', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        
        # Add text annotations
        accuracy = np.trace(cm) / np.sum(cm)
        plt.text(0.5, -0.15, f'Overall Accuracy: {accuracy*100:.2f}%', 
                ha='center', va='center', transform=plt.gca().transAxes,
                fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved: {save_path}")
        plt.close()
        
        return cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_name="roc_curve"):
        """Plot ROC curve"""
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('Receiver Operating Characteristic (ROC) Curve', 
                 fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"✅ ROC curve saved: {save_path}")
        plt.close()
        
        return roc_auc
    
    def plot_precision_recall_curve(self, y_true, y_pred_proba, save_name="precision_recall"):
        """Plot Precision-Recall curve"""
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label='PR curve')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower left", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"✅ Precision-Recall curve saved: {save_path}")
        plt.close()
    
    def plot_prediction_scatter(self, predictions, ground_truth, save_name="prediction_scatter"):
        """Scatter plot of predictions vs ground truth"""
        plt.figure(figsize=(10, 8))
        
        colors = ['green' if p == g else 'red' 
                 for p, g in zip(predictions, ground_truth)]
        
        plt.scatter(range(len(predictions)), predictions, 
                   c=colors, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        plt.scatter(range(len(ground_truth)), ground_truth, 
                   c='blue', alpha=0.3, s=30, marker='x', label='Ground Truth')
        
        plt.xlabel('Sample Index', fontsize=12)
        plt.ylabel('Class', fontsize=12)
        plt.title('Prediction Scatter Plot', fontsize=16, fontweight='bold')
        plt.yticks([0, 1], ['No Fire', 'Fire'])
        plt.legend(['Correct Prediction', 'Ground Truth'], fontsize=10)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"✅ Scatter plot saved: {save_path}")
        plt.close()
    
    def plot_class_distribution(self, y_true, class_names, save_name="class_distribution"):
        """Plot class distribution"""
        unique, counts = np.unique(y_true, return_counts=True)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar([class_names[i] for i in unique], counts, 
                      color=['#ff6b6b', '#4ecdc4'], edgecolor='black', linewidth=1.5)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('Class Distribution', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"✅ Class distribution plot saved: {save_path}")
        plt.close()
    
    def plot_model_comparison(self, results_dict, save_name="model_comparison"):
        """Compare multiple models"""
        models = list(results_dict.keys())
        metrics = ['test_accuracy', 'test_precision', 'test_recall', 'test_auc']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'AUC']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        for idx, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
            ax = axes[idx // 2, idx % 2]
            values = [results_dict[model].get(metric, 0) for model in models]
            
            bars = ax.bar(models, values, color=plt.cm.viridis(np.linspace(0, 1, len(models))),
                         edgecolor='black', linewidth=1.5)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(f'{metric_name} Comparison', fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.1])
            ax.grid(True, alpha=0.3, axis='y')
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"✅ Model comparison plot saved: {save_path}")
        plt.close()
    
    def visualize_prediction(self, image, prediction, confidence, ground_truth=None, 
                           save_name="prediction_visualization"):
        """Visualize single prediction"""
        fig, axes = plt.subplots(1, 2 if ground_truth is None else 3, 
                                figsize=(15, 5))
        
        # Original image
        if isinstance(image, str):
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = image
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
        
        axes[0].imshow(img)
        axes[0].set_title('Input Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Prediction
        pred_text = "FIRE DETECTED" if prediction == 1 else "NO FIRE"
        pred_color = 'red' if prediction == 1 else 'green'
        
        axes[1].text(0.5, 0.5, pred_text, 
                    ha='center', va='center', fontsize=24, fontweight='bold',
                    color=pred_color, transform=axes[1].transAxes)
        axes[1].text(0.5, 0.3, f'Confidence: {confidence*100:.2f}%',
                    ha='center', va='center', fontsize=16,
                    transform=axes[1].transAxes)
        axes[1].set_title('Prediction', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Ground truth (if available)
        if ground_truth is not None:
            gt_text = "FIRE" if ground_truth == 1 else "NO FIRE"
            gt_color = 'red' if ground_truth == 1 else 'green'
            correct = "✓" if prediction == ground_truth else "✗"
            
            axes[2].text(0.5, 0.5, gt_text,
                        ha='center', va='center', fontsize=24, fontweight='bold',
                        color=gt_color, transform=axes[2].transAxes)
            axes[2].text(0.5, 0.3, f'Match: {correct}',
                        ha='center', va='center', fontsize=20,
                        transform=axes[2].transAxes)
            axes[2].set_title('Ground Truth', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"✅ Prediction visualization saved: {save_path}")
        plt.close()
    
    def visualize_spread_prediction(self, input_image, spread_mask, analysis, 
                                   save_name="spread_prediction"):
        """Visualize fire spread prediction"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original image
        if isinstance(input_image, str):
            img = cv2.imread(input_image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = input_image
            if img.max() <= 1:
                img = (img * 255).astype(np.uint8)
        
        axes[0].imshow(img)
        axes[0].set_title('Current State', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Spread prediction heatmap
        axes[1].imshow(img, alpha=0.5)
        heatmap = axes[1].imshow(spread_mask, cmap='hot', alpha=0.7)
        plt.colorbar(heatmap, ax=axes[1], label='Spread Probability')
        axes[1].set_title('Predicted Spread', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Analysis
        analysis_text = f"""
Risk Level: {analysis['risk_level']}

Spread Area: {analysis['spread_percentage']:.1f}%
Max Intensity: {analysis['max_intensity']:.2f}
Avg Intensity: {analysis['avg_intensity']:.2f}

Critical Zones: {len(analysis['critical_zones'])}

Recommendation:
{analysis['recommended_action']}
        """
        
        axes[2].text(0.1, 0.5, analysis_text,
                    ha='left', va='center', fontsize=11,
                    transform=axes[2].transAxes,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].set_title('Analysis', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = self.output_dir / f"{save_name}.png"
        plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
        print(f"✅ Spread prediction visualization saved: {save_path}")
        plt.close()
    
    def create_comprehensive_report(self, y_true, y_pred, y_pred_proba, class_names):
        """Create comprehensive evaluation report with all visualizations"""
        print("\n" + "="*80)
        print("📊 CREATING COMPREHENSIVE VISUALIZATION REPORT")
        print("="*80 + "\n")
        
        # Confusion Matrix
        cm = self.plot_confusion_matrix(y_true, y_pred, class_names)
        
        # ROC Curve
        roc_auc = self.plot_roc_curve(y_true, y_pred_proba)
        
        # Precision-Recall Curve
        self.plot_precision_recall_curve(y_true, y_pred_proba)
        
        # Scatter Plot
        self.plot_prediction_scatter(y_pred, y_true)
        
        # Class Distribution
        self.plot_class_distribution(y_true, class_names)
        
        # Classification Report
        report = classification_report(y_true, y_pred, 
                                       target_names=class_names,
                                       output_dict=True)
        
        report_path = self.output_dir / "classification_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4)
        print(f"✅ Classification report saved: {report_path}")
        
        print("\n" + "="*80)
        print("✅ COMPREHENSIVE REPORT COMPLETE")
        print(f"   All visualizations saved to: {self.output_dir}")
        print("="*80 + "\n")
        
        return report
