import pandas as pd
import argparse
from pathlib import Path
import os

def extract_brand_conversations(input_csv, output_csv, brand_author_id="AppleSupport"):
    print(f"Loading data from {input_csv}...")
    
    # We might need to chunk if memory is an issue, but pandas can usually handle 500MB
    df = pd.read_csv(input_csv)
    
    print(f"Total tweets in dataset: {len(df)}")
    
    # 1. Find all tweets by the brand
    brand_tweets = df[df['author_id'] == brand_author_id]
    print(f"Total tweets by {brand_author_id}: {len(brand_tweets)}")
    
    # 2. Find customer tweets that the brand replied to
    # in_response_to_tweet_id can be parsed as float if there are NaNs
    in_response_ids = brand_tweets['in_response_to_tweet_id'].dropna().astype(float).tolist()
    
    # Customers' inbound tweets
    df['tweet_id_float'] = df['tweet_id'].astype(float)
    customer_inbound = df[df['tweet_id_float'].isin(in_response_ids)]
    
    # 3. Combine customer inbound tweets and brand replies
    brand_conversations = pd.concat([customer_inbound, brand_tweets]).drop_duplicates(subset=['tweet_id'])
    
    print(f"Extracted {len(brand_conversations)} tweets related to {brand_author_id} conversations.")
    
    # Save to output
    brand_conversations.to_csv(output_csv, index=False)
    print(f"Saved extracted data to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract brand conversations from twcs.csv")
    parser.add_argument("--input", type=str, default="archive/twcs/twcs.csv", help="Path to raw twcs.csv")
    parser.add_argument("--output", type=str, default="data/processed/apple_support_tweets.csv", help="Output path")
    parser.add_argument("--brand", type=str, default="AppleSupport", help="Brand author_id")
    
    args = parser.parse_args()
    
    # Resolve paths
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    extract_brand_conversations(input_path, output_path, args.brand)
