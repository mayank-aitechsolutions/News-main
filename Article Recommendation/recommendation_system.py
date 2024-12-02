import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Convert to DataFrame
df = pd.read_csv("BBC News Train.csv")

# Preprocessing and vectorization
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Text'])

# Compute similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_articles(article_id, df, cosine_sim, top_n=3):
    # Find the index of the input article
    idx = df[df['ArticleId'] == article_id].index[0]
    
    # Get similarity scores for the input article
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Sort articles by similarity scores (excluding the input article itself)
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    
    # Get the indices of the recommended articles
    article_indices = [i[0] for i in sim_scores]
    
    # Return the recommended articles
    return df.iloc[article_indices][['ArticleId', 'Text', 'Category']]


# Example: Recommend articles similar to ArticleID 1
recommended_articles = recommend_articles(article_id=154, df=df, cosine_sim=cosine_sim)
print("Recommended Articles:")
print(recommended_articles)