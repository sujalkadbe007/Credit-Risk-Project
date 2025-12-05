# Feature importance (for Random Forest - use preprocessor to get feature names)
# Get feature names after one-hot encoding
ohe = best_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
ohe_features = ohe.get_feature_names_out(categorical_features)
features_names = numerical_features + list(ohe_features)

importances = best_rf.named_steps['clf'].feature_importances_
fi_df = pd.DataFrame({'feature': feature_names, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=fi_df.head(20))
plt.title('Top 20 Feature Importances - Random Forest')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_importance.png'))
plt.close()

# Save the best model
model_path = os.path.join(OUTPUT_DIR, 'best_model_rf.joblib')
joblib.dump(best_rf, model_path)
print("Saved best model to:", moodel_path)

# Same evaluation summary to CSV
summary = {
  'model': ['LogisticRegression', 'RandomForest', 'RandomForest_GridSearch' + ('_XGBoost' if xgb_available else '')],
    'notes': ['balanced logistic', 'baseline RF', 'tuned rf']
}
pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR, 'model_summary.csv'), index=False)

print("Project completed. Check outputs folder for plots and model.")
