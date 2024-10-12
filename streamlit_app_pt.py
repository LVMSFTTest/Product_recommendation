import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
import streamlit as st

# Load data (replace this with your own dataset)
@st.cache_data
def load_data():
    return pd.read_csv("user_product_ratings.csv", names=['user_id', 'product_id', 'rating', 'timestamp'])

# Load the dataset
data = load_data()

# Assuming your image file is named 'background_image.png' and is in the same directory as your script
background_image_path = "background_image.png"  # Relative path to the image file

# Debug: Display the background image
#st.image(background_image_path, caption='Background Image Preview', use_column_width=True)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("{background_image_path}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)



# Streamlit app title and description
st.title("Amazon Product Recommendation System")
st.markdown("""
    Welcome to the Amazon Product Recommendation System! 
    Here, you can explore personalized product recommendations based on user preferences.
""")

# Display a relevant image
st.image("C:\\Users\\Vetri GP\\OneDrive\\Desktop\\BIA\\SRM\\Recom.png", caption="Explore recommended products tailored just for you!", use_column_width=True)

# Display data in Streamlit
st.write("Dataset Preview", data.head())
st.write("Data shape", data.shape)

# Sample 10% of the data
data_sample = data.sample(frac=0.0005, random_state=42)
st.write("Sampled Data shape", data_sample.shape)

# Check for and clean duplicates and missing data
def check_and_clean_data(data):
    if data.duplicated(subset=['user_id', 'product_id']).any():
        st.write("Duplicate entries found, removing them...")
        data = data.drop_duplicates(subset=['user_id', 'product_id'])

    if data[['user_id', 'product_id', 'rating']].isnull().any().any():
        st.write("Missing values found, removing them...")
        data = data.dropna(subset=['user_id', 'product_id', 'rating'])
        
    return data

# Clean the data sample
data_sample = check_and_clean_data(data_sample)

# Map user_id and product_id to integer indices
data_sample['user_id_code'], user_mapping = pd.factorize(data_sample['user_id'])
data_sample['product_id_code'], product_mapping = pd.factorize(data_sample['product_id'])

# Train-Test Split
train_data = data_sample.sample(frac=0.8, random_state=42)
test_data = data_sample.drop(train_data.index)

# Rank-Based Recommendation Function
def rank_based_recommendation(data, top_n=5):
    # Rank products based on average rating
    product_ratings = data.groupby('product_id')['rating'].mean().sort_values(ascending=False)
    return product_ratings.head(top_n).index.tolist()

# Similarity-Based Recommendation
def similarity_based_recommendation(user_id, data, top_n=5):
    data = check_and_clean_data(data)

    # Pivot data into a sparse matrix using the integer codes
    user_item_matrix = data.pivot_table(index='user_id_code', columns='product_id_code', values='rating').fillna(0)
    user_item_matrix_sparse = csr_matrix(user_item_matrix)  # Use sparse matrix format

    # Calculate cosine similarity between users
    user_similarity = cosine_similarity(user_item_matrix_sparse)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)

    # Find the integer code for the given user_id
    user_code = user_mapping.get_loc(user_id)  # Get the integer index of the user

    # Find similar users and recommend products (get only the top N similar users)
    similar_users = user_similarity_df[user_code].sort_values(ascending=False).index[1:top_n + 1]
    
    recommended_products = []
    product_limit = 5  # Limit the number of unique recommended products

    for similar_user in similar_users:
        # Get products for the similar user
        user_products = user_item_matrix.loc[similar_user].sort_values(ascending=False).index.tolist()

        for product_code in user_products:
            # Check if the product is already in the recommendations
            if product_mapping[product_code] not in recommended_products:
                recommended_products.append(product_mapping[product_code])
            
            # Stop adding if we've reached the product limit
            if len(recommended_products) >= product_limit:
                break
        
        if len(recommended_products) >= product_limit:
            break

    return recommended_products

# Model-Based Recommendation using SVD
def model_based_recommendation(user_id, data, top_n=5):
    # Use the integer code mapping for user_id and product_id
    user_item_matrix = data.pivot_table(index='user_id_code', columns='product_id_code', values='rating').fillna(0)
    user_item_matrix_sparse = csr_matrix(user_item_matrix)
    
    # Apply Truncated SVD
    n_components = min(user_item_matrix_sparse.shape[0], user_item_matrix_sparse.shape[1], 5)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd_matrix = svd.fit_transform(user_item_matrix_sparse)
    
    # Get the integer code for the given user_id
    user_code = user_mapping.get_loc(user_id)
    
    # Get predicted ratings for the user
    user_ratings = svd_matrix[user_code]
    
    # Create a list of product_id and predicted ratings
    product_ratings = [(product_id, rating) for product_id, rating in zip(user_item_matrix.columns, user_ratings)]
    sorted_ratings = sorted(product_ratings, key=lambda x: x[1], reverse=True)[:top_n]

    # Map back from integer product codes to actual product IDs
    return [product_mapping[product_id] for product_id, _ in sorted_ratings]

# Select User ID for Recommendations
user_id = st.selectbox("Select User ID for Recommendation", data_sample['user_id'].unique())

# Rank-Based Recommendations
rank_recommendations = rank_based_recommendation(train_data, top_n=5)
st.write(f"Rank-Based Recommendations for User {user_id}: {rank_recommendations}")

# Similarity-Based Recommendations
similarity_recommendations = similarity_based_recommendation(user_id, train_data, top_n=5)
st.write(f"Similarity-Based Recommendations for User {user_id}: {similarity_recommendations}")

# Model-Based Recommendations (SVD)
model_recommendations = model_based_recommendation(user_id, train_data, top_n=5)
st.write(f"Model-Based (SVD) Recommendations for User {user_id}: {model_recommendations}")
