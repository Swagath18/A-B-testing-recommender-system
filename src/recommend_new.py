import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd

item_metadata = pd.read_csv("data/item_metadata_tfidf.csv")

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(item_metadata['description'])
item_ids = item_metadata['item_id'].values
item_index_lookup = {item_id: idx for idx, item_id in enumerate(item_ids)}

def recommend_content_based(user_history, top_n=5):
    valid_items = [item_index_lookup[item] for item in user_history if item in item_index_lookup]
    if not valid_items:
        return [int(i) for i in item_ids[:top_n].tolist()]

    # Convert matrix to array properly 
    user_profile = tfidf_matrix[valid_items].mean(axis=0)
    user_profile_array = np.asarray(user_profile).reshape(1, -1)

    similarities = cosine_similarity(user_profile_array, tfidf_matrix)[0]
    recommended_indices = similarities.argsort()[::-1]

    seen_indices = set(valid_items)
    recommended = [item_ids[i] for i in recommended_indices if i not in seen_indices]
    # Convert all to standard Python int
    return [int(item) for item in recommended[:top_n]]
