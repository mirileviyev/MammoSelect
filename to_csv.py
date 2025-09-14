from ucimlrepo import fetch_ucirepo
import pandas as pd

# =======================
# Dataset 1: Breast Cancer Wisconsin (Diagnostic)
# =======================

# Step 1: Fetch dataset
breast_cancer_wisconsin = fetch_ucirepo(id=17)

# Step 2: Extract features and labels
X1 = breast_cancer_wisconsin.data.features
y1 = breast_cancer_wisconsin.data.targets

# Step 3: Combine into a single DataFrame
df1 = pd.concat([X1, y1], axis=1)

# Step 4: Save to CSV
df1.to_csv('breast_cancer_wisconsin.csv', index=False)
print("✓ 'breast_cancer_wisconsin.csv' saved successfully!")
print(df1.head())

# =======================
# Dataset 2: Breast Cancer Coimbra
# =======================

# Step 1: Fetch dataset
breast_cancer_coimbra = fetch_ucirepo(id=451)

# Step 2: Extract features and labels
X2 = breast_cancer_coimbra.data.features
y2 = breast_cancer_coimbra.data.targets

# Step 3: Combine into a single DataFrame
df2 = pd.concat([X2, y2], axis=1)

# Step 4: Save to CSV
df2.to_csv('breast_cancer_coimbra.csv', index=False)
print("✓ 'breast_cancer_coimbra.csv' saved successfully!")
print(df2.head())
