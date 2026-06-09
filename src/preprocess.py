import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop(columns=['customerID'], inplace=True)
    return df

def encode_features(df: pd.DataFrame):
    df = df.copy()
    binary_cols = [c for c in df.select_dtypes('object').columns
                   if c != 'Churn' and df[c].nunique() == 2]
    multi_cols = [c for c in df.select_dtypes('object').columns
                  if c != 'Churn' and df[c].nunique() > 2]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col])
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    df['Churn'] = (df['Churn'] == 'Yes').astype(int)
    return df

def split(df: pd.DataFrame, target='Churn', test_size=0.2, seed=42):
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)