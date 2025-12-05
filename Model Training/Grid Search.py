# Hyperparameter tuning for Random Forest (Grid Search)
param_grid = {
  'clf__n_estimators': [100, 200],
  'clf__max_depth': [None, 6, 12],
  'clf__min_samples_split': [2, 5]
}

grid_search = GridSearchCV(rf-pipeline, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)
print("\nStarting Grid Search for Random Forest")
grid_search.fit(X_train, y_train)

print("Best params:", grid_search.best_params_)
best_rf = grid_search.best_estimator_

# Evaluate best_rf
best_pred = best_rf.predict(X_test)
best_proba = best_rf.predict_proba(X_test)[:,1]
eval_model(y_test, best_pred, best_proba, 'RandomForest_GridSearch')
