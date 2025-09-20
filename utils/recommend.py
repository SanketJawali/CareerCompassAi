import pandas as pd
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
import os

load_dotenv()
# Load BERT model
BERT_MODEL_NAME = os.getenv("BERT_MODEL_NAME")
model = SentenceTransformer(BERT_MODEL_NAME)

TOP_N=5


# Load the full colleges CSV
colleges = pd.read_csv('data/colleges.csv')

# Ensure all necessary columns exist
for col in ['description', 'fields']:
    if col not in colleges.columns:
        colleges[col] = '' if col == 'description' else 'General'

# Combine relevant text fields for embeddings
college_texts = (colleges['college_name'] + ' ' + colleges['location'] + ' ' +
                colleges['category'] + ' ' + colleges['contact_number'] + ' ' + colleges['email'] + ' ' + colleges['website']).tolist()

# Precompute embeddings
college_embeddings = model.encode(college_texts, convert_to_tensor=True)

def get_recommendations(colleges_df, location, interests, field, top_n=TOP_N):
    """
    Returns top N recommended colleges based on location, interests, and field.
    """
    query = f"{location} {interests} {field}"
    query_embedding = model.encode(query, convert_to_tensor=True)

    # Compute cosine similarity
    cosine_scores = util.cos_sim(query_embedding, college_embeddings)[0]

    # Get top N indices
    top_indices = cosine_scores.argsort(descending=True)[:top_n]

    # Fetch recommended colleges
    recommended_colleges = colleges_df.iloc[top_indices]

    # Return all info from CSV
    return recommended_colleges[['college_name', 'location', 'category', 'contact_number', 'email', 'website']].to_dict(orient='records')
