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
