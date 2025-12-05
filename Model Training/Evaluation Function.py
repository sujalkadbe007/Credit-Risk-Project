def eval_model(y_true, y_pred, y_proba, name):
  acc = accuracy_score(y_true, y_pred)
  prec = precision_score(y_true, y_pred)
  rec = recall_score(y_true, y_pred)
  f1 = f1_score(y_true, y_pred)
  roc = roc_auc_score(y_true, y_proba)
  print(f"\n---{name} ---")
  print("Accuracy:", acc)
  print("Precision:", prec)
  print("Recall:", rec)
  print("F1-score", f1)
  print("ROC AUC:", roc)
  print("\nClassification Report:\n", classification_report(y_true, y_pred))
  
  # Confusion matrix
  cm = confusion_matrix(y_true, y_pred)
  plt.figure(figsize=(5,4))
  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
  ply.title(f'Confusion Matrix - {name}')
  plt.xlable('Predicted')
  plt.ylable('Actual')
  plt.savefig(os.path.join(OUTPUT_DIR, f'confusion_matrix_{name}.png'))
  plt.close()
  
  # ROC Curve
  fpr, tpr, _ = roc_curve(y_true, y_proba)
  plt.figure(figsize=(6,4))
  plt.plot(fpr, tpr, label=f'{name} (AUC = {roc:.3f})')
  plt.plot([0,1],[0,1], linestyle='--', color='grey')
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  plt.title(f'ROC Curve - {name}')
  plt.legend()
  plt.savefig(os.path.join(OUTPUT_DIR, f'roc_{name}.png'))
  plt.close()
  
  # Evaluate models
  eval_model(y_test, lr_pred, lr_proba, 'LogisticRegression')
  eval_model(y_test, rf_pred, rf_proba, 'RandomForest')
  if xgb_availabe:
    eval_model(y_test, xgb_pred, xgb_proba, 'XGBoost')
