import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv('../data/heart_disease_uci.csv')

# Identify categorical and numerical columns
categorical_cols = ['sex', 'cp', 'restecg', 'slope', 'thal']
numerical_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'ca']

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Apply the transformations
df_preprocessed = preprocessor.fit_transform(df)

# Split the data into train and test sets
X_train, X_test = train_test_split(df_preprocessed, test_size=0.2, random_state=42)

# Save preprocessed data (optional)
pd.DataFrame(X_train).to_csv('../data/train_preprocessed.csv', index=False)
pd.DataFrame(X_test).to_csv('../data/test_preprocessed.csv', index=False)
