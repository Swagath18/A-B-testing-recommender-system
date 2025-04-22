import pandas as pd
from src.recommend_baseline import recommend_popular
from src.recommend_new import recommend_content_based
from src.ab_test import assign_groups, simulate_recommendations, simulate_clicks
from src.evaluation import evaluate_ctr
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt

# Load interaction data
df = pd.read_csv("data/interactions.csv")

# Assign A/B groups
df = assign_groups(df)

# Define baseline and content-based recommenders
baseline = lambda df: recommend_popular(df)
content = lambda uid: recommend_content_based(df[df['user_id'] == uid]['item_id'].tolist())

# Simulate recommendations and clicks
df = simulate_recommendations(df, baseline, content)
df = simulate_clicks(df)

# Evaluate CTR
ctr = evaluate_ctr(df)

# Statistical significance test
clicks = df.groupby("group")["clicked"].agg(["sum", "count"])
z_stat, p_val = proportions_ztest(count=clicks["sum"], nobs=clicks["count"])
print("\n--- Statistical Significance Test ---")
print(f"Z-statistic = {z_stat:.2f}")
print(f"P-value     = {p_val:.4f}")
if p_val < 0.05:
    print("✅ Result: Statistically significant improvement!")
else:
    print("❌ Result: Not statistically significant.")

# Plot CTR by group
ctr.plot(kind="bar", title="CTR by Group", ylabel="Click-Through Rate", color=["skyblue", "salmon"])
plt.xticks(rotation=0)
plt.ylim(0, max(ctr) + 0.02)
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()