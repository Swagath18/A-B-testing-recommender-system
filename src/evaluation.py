def evaluate_ctr(df):
    ctr = df.groupby('group')['clicked'].mean()
    print("CTR A:", round(ctr['A'], 3))
    print("CTR B:", round(ctr['B'], 3))
    print("Lift: ", round((ctr['B'] - ctr['A']) / ctr['A'] * 100, 2), "%")
    return ctr