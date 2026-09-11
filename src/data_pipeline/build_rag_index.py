import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
import os
import argparse
from tqdm import tqdm

def build_index(tweets_csv, db_path, collection_name="apple_support_replies", max_docs=5000):
    print(f"Loading data from {tweets_csv}...")
    df = pd.read_csv(tweets_csv, low_memory=False)
    
    # We want to index successful brand replies.
    # To do this, we need pairs of (customer_tweet, brand_reply)
    brand_tweets = df[df['author_id'] == 'AppleSupport'].dropna(subset=['in_response_to_tweet_id'])
    brand_tweets['in_response_to_tweet_id'] = pd.to_numeric(brand_tweets['in_response_to_tweet_id'], errors='coerce')
    
    customer_tweets = df[df['author_id'] != 'AppleSupport'].copy()
    customer_tweets['tweet_id_float'] = pd.to_numeric(customer_tweets['tweet_id'], errors='coerce')
    
    print("Merging customer inquiries with brand replies...")
    merged = pd.merge(
        customer_tweets[['tweet_id_float', 'text']], 
        brand_tweets[['in_response_to_tweet_id', 'text']], 
        left_on='tweet_id_float', 
        right_on='in_response_to_tweet_id', 
        suffixes=('_customer', '_brand')
    )
    
    print(f"Found {len(merged)} conversation pairs.")
    
    # Sample to save time/memory for the assignment
    if len(merged) > max_docs:
        merged = merged.sample(n=max_docs, random_state=42)
        print(f"Sampled down to {max_docs} pairs for RAG index.")
        
    print(f"Initializing ChromaDB at {db_path}...")
    os.makedirs(db_path, exist_ok=True)
    client = chromadb.PersistentClient(path=db_path)
    
    # Use lightweight local embeddings
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = client.get_or_create_collection(
        name=collection_name, 
        embedding_function=sentence_transformer_ef
    )
    
    documents = merged['text_customer'].tolist()
    metadatas = [{"brand_reply": reply} for reply in merged['text_brand'].tolist()]
    ids = [str(i) for i in merged['tweet_id_float'].tolist()]
    
    print("Adding documents to vector store (this may take a few minutes)...")
    # Batch add to avoid memory issues
    batch_size = 1000
    for i in tqdm(range(0, len(documents), batch_size)):
        collection.add(
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
            ids=ids[i:i+batch_size]
        )
        
    print("RAG index built successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../../data/processed/apple_support_tweets.csv")
    parser.add_argument("--db_path", type=str, default="../../data/chroma_db")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    db_path = os.path.join(script_dir, args.db_path) if not os.path.isabs(args.db_path) else args.db_path
    
    build_index(input_path, db_path)
