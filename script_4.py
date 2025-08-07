# Let's train models with a more efficient approach and focus on key algorithms
print("STEP 2: MODEL TRAINING AND EVALUATION (Optimized)")
print("="*60)

# Train a few key models efficiently
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=50, max_depth=8),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=50, max_depth=4)
}

model_results = {}

for name, model in models.items():
    print(f"Training {name}...")
    
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
    
    print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, AUC: {auc:.4f}")

print("\n" + "="*60)
print("MODEL PERFORMANCE COMPARISON")
print("="*60)

# Create and display results
results_comparison = []
for name, results in model_results.items():
    results_comparison.append({
        'Model': name,
        'Accuracy': f"{results['accuracy']:.4f}",
        'Precision': f"{results['precision']:.4f}",
        'Recall': f"{results['recall']:.4f}",
        'F1-Score': f"{results['f1_score']:.4f}",
        'AUC': f"{results['auc']:.4f}"
    })

results_df = pd.DataFrame(results_comparison)
print(results_df.to_string(index=False))

# Find best model by AUC
best_auc = 0
best_model_name = None
for name, results in model_results.items():
    if results['auc'] > best_auc:
        best_auc = results['auc']
        best_model_name = name

print(f"\nBest Model: {best_model_name} (AUC: {best_auc:.4f})")

# Feature importance for the best model
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    best_model = model_results[best_model_name]['model']
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(f"\nTop 10 Feature Importances ({best_model_name}):")
    print("-" * 50)
    for idx, row in feature_importance.head(10).iterrows():
        print(f"{row['Feature']:<30}: {row['Importance']:.4f}")

# Business impact analysis
print(f"\nBUSINESS IMPACT ANALYSIS")
print("="*60)

best_predictions = model_results[best_model_name]['predictions']
cm = confusion_matrix(y_test, best_predictions)

tn, fp, fn, tp = cm.ravel()

print(f"Confusion Matrix Analysis:")
print(f"  True Negatives (Correctly identified non-defaults): {tn}")
print(f"  False Positives (Incorrectly flagged as default): {fp}")
print(f"  False Negatives (Missed defaults): {fn}")
print(f"  True Positives (Correctly identified defaults): {tp}")

# Calculate business metrics
total_loans = len(y_test)
default_rate = y_test.mean()
model_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
model_recall = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"\nBusiness Metrics:")
print(f"  Total loan applications evaluated: {total_loans}")
print(f"  Actual default rate: {default_rate:.2%}")
print(f"  Model identifies {model_recall:.1%} of actual defaults")
print(f"  {model_precision:.1%} of flagged applications are actual defaults")
print(f"  Potential defaults missed: {fn} ({fn/total_loans:.1%} of all applications)")

print("\nModel training and evaluation completed successfully!")