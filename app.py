import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Load Data
books = pd.read_csv("books.csv")
ratings = pd.read_csv("ratings.csv")  # Add your ratings.csv here

# --- Collaborative Filtering Preprocessing ---
user_item_matrix = ratings.pivot_table(index='User_ID', columns='Book_ID', values='Rating').fillna(0)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

st.header("👤 Collaborative Filtering (User-Based)")
user_ids = user_item_matrix.index.tolist()
selected_user = st.selectbox("Choose a User ID", user_ids)

if st.button("Recommend", key="cf_recommend"):
    # Get ratings from similar users
    sim_users = user_similarity_df.loc[selected_user].sort_values(ascending=False)[1:4].index
        sim_ratings = user_item_matrix.loc[sim_users].mean().sort_values(ascending=False)

        # Recommend unrated books
        user_rated_books = user_item_matrix.loc[selected_user][user_item_matrix.loc[selected_user] > 0].index
        recommendations = sim_ratings.drop(user_rated_books).head(3)

        st.write("### Recommended Books:")
        for book_id in recommendations.index:
            title = books[books['Book_ID'] == book_id]['Title'].values[0]
            st.write("- " + title)
