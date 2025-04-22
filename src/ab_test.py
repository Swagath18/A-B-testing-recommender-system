import numpy as np

def assign_groups(df, ratio=0.5):
    df['group'] = np.random.choice(['A', 'B'], size=len(df), p=[1-ratio, ratio])
    return df

def simulate_recommendations(df, baseline_func, content_func):
    df['recommended'] = df.apply(
        lambda row: baseline_func(df) if row['group'] == 'A' else content_func(row['user_id']), axis=1)
    return df

def simulate_clicks(df):
    df['clicked'] = df['group'].apply(lambda g: 1 if np.random.rand() < (0.08 if g == 'A' else 0.097) else 0)
    return df