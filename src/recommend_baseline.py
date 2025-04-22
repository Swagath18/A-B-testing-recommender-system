def recommend_popular(df, N=5):
    return df['item_id'].value_counts().head(N).index.tolist()