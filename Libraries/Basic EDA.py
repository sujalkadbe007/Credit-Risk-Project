// Basic EDA
print("\n--- Value counts for target ---")
print(df['risk'].value_counts(normalize=False))
print(df.describe())

// Plot target distribution
plt.figure(figsize=(6,4))
sns.countplot(x='risk', data=df)
plt.title('Risk Distribution (0 = Good / Low Risk, 1 = Bad / High Risk)')
plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'))
plt.close()

// Correlations for numeric columns
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove('risk')
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols + ['risk']].corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'))
plt.close()
