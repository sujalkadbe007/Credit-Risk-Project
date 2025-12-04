# define feature sets
categorical_features = ['loan_purpose', 'home_ownership', 'education', 'marital_status']
numerical_features = [c for c in df.columns if c not in categorical_feature + ['risk']]

# Pipelines
numeric_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(transformers=[
   ('num', numeric_transformer, numerical_features),
   ('cat', categorical_transformer, categorical_features)
])

# Train- Test split
X = df.drop('risk', axis= 1)
y = df['risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
