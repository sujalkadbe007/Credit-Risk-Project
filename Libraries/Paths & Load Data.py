# Paths

DATA_PATH = "Credit_risk_dataset_large.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load Data
df = pd.read_csv(DATA_PATH)
print("Dataset shape:", df.shape)
print(df.head())
