"""
Task 3: Predictive Analytics for Resource Allocation
Using Breast Cancer Dataset to Predict Issue Priority Levels
"""

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PredictiveResourceAllocator:
    """
    AI-Powered Predictive Analytics System for Resource Allocation
    
    This class demonstrates how machine learning can be used to predict
    issue priorities and optimize resource allocation in software development
    or project management contexts.
    """
    
    def __init__(self):
        """Initialize the Predictive Resource Allocator"""
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        self.target_names = ['Low Priority', 'Medium Priority', 'High Priority']
        
    def load_and_prepare_data(self):
        """
        Load and prepare the dataset for predictive modeling
        
        Note: We're adapting the breast cancer dataset to simulate
        software issue priority prediction. In reality, you would use
        actual project management data with features like:
        - Bug severity scores
        - Customer impact ratings
        - Development complexity
        - Business value metrics
        - Time constraints
        - Resource availability
        
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        print("Loading and preparing dataset...")
        
        # Load the breast cancer dataset
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        
        # Create simulated priority labels based on feature combinations
        # This simulates how issue priority might be determined in real scenarios
        y_binary = data.target  # Original binary classification
        
        # Transform binary to multi-class priority system
        # High priority: malignant cases with high severity scores
        # Medium priority: borderline cases
        # Low priority: benign cases with low complexity
        
        # Calculate composite scores for priority assignment
        severity_score = (X['mean radius'] + X['mean texture'] + X['mean perimeter']) / 3
        complexity_score = (X['mean compactness'] + X['mean concavity'] + X['mean symmetry']) / 3
        
        # Create priority labels based on medical severity (adapted for issue priority)
        y_priority = []
        for i in range(len(y_binary)):
            if y_binary[i] == 1:  # Malignant (High Priority Issues)
                if severity_score.iloc[i] > severity_score.quantile(0.7):
                    y_priority.append(2)  # High Priority
                else:
                    y_priority.append(1)  # Medium Priority
            else:  # Benign (Lower Priority Issues)
                if complexity_score.iloc[i] > complexity_score.quantile(0.6):
                    y_priority.append(1)  # Medium Priority
                else:
                    y_priority.append(0)  # Low Priority
        
        y = np.array(y_priority)
        
        # Store feature names for later analysis
        self.feature_names = X.columns.tolist()
        
        # Print dataset information
        print(f"Dataset shape: {X.shape}")
        print(f"Feature columns: {len(X.columns)}")
        print(f"Target distribution:")
        unique, counts = np.unique(y, return_counts=True)
        for priority, count in zip(unique, counts):
            print(f"  {self.target_names[priority]}: {count} ({count/len(y)*100:.1f}%)")
        
        return X, y
    
    def preprocess_data(self, X, y):
        """
        Comprehensive data preprocessing pipeline
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Preprocessed X and y
        """
        print("\nPreprocessing data...")
        
        # Check for missing values
        missing_values = X.isnull().sum()
        if missing_values.any():
            print(f"Found {missing_values.sum()} missing values")
            # Impute missing values
            imputer = SimpleImputer(strategy='median')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        else:
            print("No missing values found")
        
        # Check for duplicates
        duplicates = X.duplicated().sum()
        if duplicates > 0:
            print(f"Found {duplicates} duplicate rows - removing...")
            X = X.drop_duplicates()
            y = y[X.index]
        else:
            print("No duplicate rows found")
        
        # Feature scaling
        print("Applying feature scaling...")
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Print preprocessing summary
        print(f"Final dataset shape: {X_scaled.shape}")
        print("Preprocessing completed successfully")
        
        return X_scaled, y
    
    def exploratory_data_analysis(self, X, y):
        """
        Perform comprehensive exploratory data analysis
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        print("\nPerforming Exploratory Data Analysis...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Exploratory Data Analysis - Issue Priority Prediction', fontsize=16)
        
        # 1. Target distribution
        priority_counts = pd.Series(y).value_counts().sort_index()
        priority_labels = [self.target_names[i] for i in priority_counts.index]
        
        axes[0, 0].pie(priority_counts.values, labels=priority_labels, autopct='%1.1f%%')
        axes[0, 0].set_title('Priority Distribution')
        
        # 2. Feature correlation heatmap (top 10 features)
        top_features = X.columns[:10]  # First 10 features for visualization
        correlation_matrix = X[top_features].corr()
        
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   ax=axes[0, 1], fmt='.2f', cbar_kws={'shrink': 0.8})
        axes[0, 1].set_title('Feature Correlation Matrix (Top 10)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].tick_params(axis='y', rotation=0)
        
        # 3. Box plot of key features by priority
        key_feature = X.columns[0]  # Use first feature as example
        priority_data = []
        for priority in range(3):
            mask = y == priority
            priority_data.append(X[key_feature][mask])
        
        axes[0, 2].boxplot(priority_data, labels=[self.target_names[i] for i in range(3)])
        axes[0, 2].set_title(f'{key_feature} by Priority Level')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Feature importance preview (using simple correlation)
        feature_importance = []
        for feature in X.columns[:15]:  # Top 15 features
            correlation = abs(np.corrcoef(X[feature], y)[0, 1])
            feature_importance.append(correlation)
        
        top_indices = np.argsort(feature_importance)[-10:]  # Top 10
        top_features_imp = [X.columns[i] for i in top_indices]
        top_scores = [feature_importance[i] for i in top_indices]
        
        axes[1, 0].barh(range(len(top_features_imp)), top_scores)
        axes[1, 0].set_yticks(range(len(top_features_imp)))
        axes[1, 0].set_yticklabels([f.split()[-1] for f in top_features_imp])  # Shortened names
        axes[1, 0].set_title('Top 10 Features by Correlation with Priority')
        axes[1, 0].set_xlabel('Absolute Correlation')
        
        # 5. Distribution of first few features
        X.iloc[:, :4].hist(ax=axes[1, 1], bins=20, alpha=0.7)
        axes[1, 1].set_title('Distribution of Key Features')
        
        # 6. Priority vs Feature scatter plot
        axes[1, 2].scatter(X.iloc[:, 0], X.iloc[:, 1], c=y, cmap='viridis', alpha=0.6)
        axes[1, 2].set_xlabel(X.columns[0])
        axes[1, 2].set_ylabel(X.columns[1])
        axes[1, 2].set_title('Priority Distribution in Feature Space')
        
        plt.tight_layout()
        plt.savefig('eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistical summary
        print("\nDataset Statistical Summary:")
        print(f"Total samples: {len(X)}")
        print(f"Number of features: {len(X.columns)}")
        print(f"Priority distribution: {dict(zip(self.target_names, np.bincount(y)))}")
        
    def train_model(self, X_train, y_train):
        """
        Train Random Forest model with hyperparameter optimization
        
        Args:
            X_train: Training features
            y_train: Training targets
        """
        print("\nTraining Random Forest model...")
        
        # Define hyperparameter grid for optimization
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None]
        }
        
        # Initialize Random Forest
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Perform Grid Search with Cross Validation
        print("Performing hyperparameter optimization...")
        grid_search = GridSearchCV(
            rf, param_grid, cv=5, scoring='f1_weighted', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Use best model
        self.model = grid_search.best_estimator_
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best cross-validation F1 score: {grid_search.best_score_:.4f}")
        
        return self.model
    
    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model performance...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1_weighted = f1_score(y_test, y_pred, average='weighted')
        f1_macro = f1_score(y_test, y_pred, average='macro')
        
        # Detailed classification report
        class_report = classification_report(y_test, y_pred, 
                                           target_names=self.target_names,
                                           output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba
        }
        
        # Print results
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Weighted): {f1_weighted:.4f}")
        print(f"F1-Score (Macro): {f1_macro:.4f}")
        
        return results
    
    def visualize_results(self, results, X_test, y_test):
        """
        Create comprehensive visualizations of model results
        
        Args:
            results: Model evaluation results
            X_test: Test features
            y_test: Test targets
        """
        print("\nGenerating result visualizations...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Model Performance Analysis', fontsize=16)
        
        # 1. Confusion Matrix
        sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', 
                   xticklabels=self.target_names, yticklabels=self.target_names,
                   ax=axes[0, 0], cmap='Blues')
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. Feature Importance (Top 15)
        top_features = results['feature_importance'].head(15)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels([f.split()[-1] for f in top_features['feature']])
        axes[0, 1].set_title('Top 15 Feature Importance')
        axes[0, 1].set_xlabel('Importance Score')
        
        # 3. Class-wise F1 Scores
        f1_scores = [results['classification_report'][name]['f1-score'] 
                    for name in self.target_names]
        axes[0, 2].bar(self.target_names, f1_scores, color=['red', 'orange', 'green'])
        axes[0, 2].set_title('F1-Score by Priority Class')
        axes[0, 2].set_ylabel('F1-Score')
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # 4. Prediction Confidence Distribution
        max_proba = np.max(results['prediction_probabilities'], axis=1)
        axes[1, 0].hist(max_proba, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 0].set_title('Prediction Confidence Distribution')
        axes[1, 0].set_xlabel('Maximum Probability')
        axes[1, 0].set_ylabel('Frequency')
        
        # 5. Precision-Recall by Class
        precision_scores = [results['classification_report'][name]['precision'] 
                           for name in self.target_names]
        recall_scores = [results['classification_report'][name]['recall'] 
                        for name in self.target_names]
        
        x_pos = np.arange(len(self.target_names))
        width = 0.35
        
        axes[1, 1].bar(x_pos - width/2, precision_scores, width, label='Precision', alpha=0.8)
        axes[1, 1].bar(x_pos + width/2, recall_scores, width, label='Recall', alpha=0.8)
        axes[1, 1].set_xlabel('Priority Class')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_title('Precision vs Recall by Class')
        axes[1, 1].set_xticks(x_pos)
        axes[1, 1].set_xticklabels(self.target_names)
        axes[1, 1].legend()
        
        # 6. Actual vs Predicted scatter plot
        axes[1, 2].scatter(y_test, results['predictions'], alpha=0.6)
        axes[1, 2].plot([0, 2], [0, 2], 'r--', lw=2)
        axes[1, 2].set_xlabel('Actual Priority')
        axes[1, 2].set_ylabel('Predicted Priority')
        axes[1, 2].set_title('Actual vs Predicted Priority')
        axes[1, 2].set_xticks([0, 1, 2])
        axes[1, 2].set_xticklabels(['Low', 'Med', 'High'])
        axes[1, 2].set_yticks([0, 1, 2])
        axes[1, 2].set_yticklabels(['Low', 'Med', 'High'])
        
        plt.tight_layout()
        plt.savefig('model_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_insights(self, results):
        """
        Generate business insights from the model
        
        Args:
            results: Model evaluation results
        """
        print("\n" + "="*60)
        print("BUSINESS INSIGHTS AND RECOMMENDATIONS")
        print("="*60)
        
        # Top predictive features
        top_5_features = results['feature_importance'].head(5)
        print("Top 5 Most Predictive Features for Priority Assignment:")
        for idx, (_, row) in enumerate(top_5_features.iterrows(), 1):
            print(f"  {idx}. {row['feature']}: {row['importance']:.4f}")
        
        # Model performance summary
        print(f"\nModel Performance Summary:")
        print(f"  Overall Accuracy: {results['accuracy']:.1%}")
        print(f"  Weighted F1-Score: {results['f1_weighted']:.4f}")
        print(f"  Macro F1-Score: {results['f1_macro']:.4f}")
        
        # Class-specific insights
        print(f"\nClass-Specific Performance:")
        for class_name in self.target_names:
            if class_name in results['classification_report']:
                report = results['classification_report'][class_name]
                print(f"  {class_name}:")
                print(f"    Precision: {report['precision']:.3f}")
                print(f"    Recall: {report['recall']:.3f}")
                print(f"    F1-Score: {report['f1-score']:.3f}")
        
        # Resource allocation recommendations
        print(f"\nResource Allocation Recommendations:")
        print(f"  1. Focus on features: {', '.join(top_5_features['feature'].head(3).tolist())}")
        print(f"  2. Model confidence: {np.mean(np.max(results['prediction_probabilities'], axis=1)):.1%}")
        print(f"  3. Recommended for production: {'Yes' if results['accuracy'] > 0.8 else 'No'}")
        
    def predict_new_issues(self, new_data):
        """
        Predict priority for new issues
        
        Args:
            new_data: New issue data (same format as training data)
            
        Returns:
            Predictions and probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model first.")
        
        # Scale the new data
        new_data_scaled = self.scaler.transform(new_data)
        
        # Make predictions
        predictions = self.model.predict(new_data_scaled)
        probabilities = self.model.predict_proba(new_data_scaled)
        
        # Convert predictions to readable format
        predicted_priorities = [self.target_names[pred] for pred in predictions]
        
        return predicted_priorities, probabilities


def main():
    """
    Main execution function that runs the complete predictive analytics pipeline
    """
    print("="*60)
    print("PREDICTIVE ANALYTICS FOR RESOURCE ALLOCATION")
    print("="*60)
    
    # Initialize the predictor
    predictor = PredictiveResourceAllocator()
    
    # Step 1: Load and prepare data
    X, y = predictor.load_and_prepare_data()
    
    # Step 2: Preprocess data
    X_processed, y_processed = predictor.preprocess_data(X, y)
    
    # Step 3: Exploratory Data Analysis
    predictor.exploratory_data_analysis(X_processed, y_processed)
    
    # Step 4: Split data
    print("\nSplitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y_processed, test_size=0.2, random_state=42, stratify=y_processed
    )
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Step 5: Train model
    model = predictor.train_model(X_train, y_train)
    
    # Step 6: Evaluate model
    results = predictor.evaluate_model(X_test, y_test)
    
    # Step 7: Visualize results
    predictor.visualize_results(results, X_test, y_test)
    
    # Step 8: Generate insights
    predictor.generate_insights(results)
    
    # Step 9: Demonstrate prediction on new data
    print("\n" + "="*60)
    print("DEMONSTRATION: PREDICTING NEW ISSUE PRIORITIES")
    print("="*60)
    
    # Use first 3 samples from test set as "new" data for demonstration
    sample_new_data = X_test.head(3)
    actual_priorities = [predictor.target_names[priority] for priority in y_test[:3]]
    
    predicted_priorities, probabilities = predictor.predict_new_issues(sample_new_data)
    
    print("Sample Predictions:")
    for i in range(len(predicted_priorities)):
        print(f"\nIssue {i+1}:")
        print(f"  Predicted Priority: {predicted_priorities[i]}")
        print(f"  Actual Priority: {actual_priorities[i]}")
        print(f"  Confidence Scores:")
        for j, priority_name in enumerate(predictor.target_names):
            print(f"    {priority_name}: {probabilities[i][j]:.3f}")
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE - Check generated visualizations!")
    print("="*60)


if __name__ == "__main__":
    # Run the complete predictive analytics pipeline
    main()