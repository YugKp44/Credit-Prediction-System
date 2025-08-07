# STEP 2: MODEL TRAINING AND EVALUATION
print("STEP 2: MODEL TRAINING AND EVALUATION")
print("="*60)

# Initialize models with different algorithms
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10, min_samples_split=5),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=6),
    'SVM': SVC(random_state=42, probability=True, gamma='scale')
}

# Training results storage
model_results = {}

print("Training multiple models...")
print("-" * 40)

for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Train the model
    model.fit(X_train_balanced, y_train_balanced)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    # Store results
    model_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'auc': auc,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  AUC: {auc:.4f}")

# Create results summary
print("\n" + "="*60)
print("MODEL PERFORMANCE SUMMARY")
print("="*60)

results_df = pd.DataFrame({
    'Model': list(model_results.keys()),
    'Accuracy': [results['accuracy'] for results in model_results.values()],
    'Precision': [results['precision'] for results in model_results.values()],
    'Recall': [results['recall'] for results in model_results.values()],
    'F1-Score': [results['f1_score'] for results in model_results.values()],
    'AUC': [results['auc'] for results in model_results.values()]
})

# Sort by AUC score (descending)
results_df = results_df.sort_values('AUC', ascending=False).reset_index(drop=True)
print(results_df)

# Find best model
best_model_name = results_df.iloc[0]['Model']
best_model = model_results[best_model_name]['model']

print(f"\nBest performing model: {best_model_name}")
print(f"Best AUC Score: {results_df.iloc[0]['AUC']:.4f}")

# Detailed evaluation of best model
print(f"\nDetailed Evaluation of {best_model_name}:")
print("-" * 40)

y_pred_best = model_results[best_model_name]['predictions']
cm = confusion_matrix(y_test, y_pred_best)

print("Confusion Matrix:")
print(f"True Negatives: {cm[0,0]}")
print(f"False Positives: {cm[0,1]}")
print(f"False Negatives: {cm[1,0]}")
print(f"True Positives: {cm[1,1]}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_best, target_names=['No Default', 'Default']))

# Save results to CSV
results_df.to_csv('model_performance_results.csv', index=False)
print(f"\nResults saved to 'model_performance_results.csv'")

# Feature importance for tree-based models
if 'Random Forest' in best_model_name or 'Gradient Boosting' in best_model_name or 'Decision Tree' in best_model_name:
    print(f"\nFeature Importance Analysis ({best_model_name}):")
    print("-" * 40)
    
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Save feature importance
    feature_importance.to_csv('feature_importance.csv', index=False)
    print(f"\nFeature importance saved to 'feature_importance.csv'")