# Logistic Regression Pipeline
lr_pipeline = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
])
#Random forest Pipeline
rf_pipeline = Pipeline(steps=[
  ('preprocessor', preprocessor),
  ('clf', RandomForestClassification(n_jobs=-1, random_state=42, class_weight='balance'))
])

#Train logistic regression
print("\nTraining Logistic Regression :")
lr_pipeline.fit(x_train, y_train)
lr_pred = lr_pipeline.predict(x_test)
lr_proba = lr_pipeline.predict_proba(x_test)[:,1]

#Train Random Forest
print("\nTraining Random Forest :")
rf_pipeline.fit(x_train, y_train)
rf_pred = rf_pipeline.predict(x_test)
rf_proba = rf_pipeline.predict_proba(x_test)[:,1]

# Optional: XGBoost if available
try:
  from xgboost import XBGClassifier
  xgb_pipeline = Pipeline(steps[
      ('preprocessor', preprocessor),
      ('clf', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42))                   
  ])
  print("Training XGBoost :")
  xgb_pipeline.fit(x_train, y_train)
  xgb_pred = xgb_pipeline.predict(x_test)
  xgb_proba = xgb_pipeline.predict_proba(x_test)[:,1]
  xgb_available = True
except Exception as e:
  print("XGboost not available or failed to import. Skipping XGBoost. Error:",e)
  xgb_available = False























