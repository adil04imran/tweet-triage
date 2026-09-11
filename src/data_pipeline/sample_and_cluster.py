import pandas as pd
import argparse
import os

def sample_first_touch(input_csv, output_csv, sample_size=1000):
    print(f"Loading conversations from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # A first touch tweet is from a customer (not AppleSupport) and is not a reply to another tweet
    first_touch = df[(df['author_id'] != 'AppleSupport') & (df['in_response_to_tweet_id'].isna())]
    print(f"Found {len(first_touch)} first-touch customer tweets.")
    
    if len(first_touch) > sample_size:
        sample = first_touch.sample(n=sample_size, random_state=42)
    else:
        sample = first_touch
        print(f"Warning: Only {len(first_touch)} tweets available, taking all.")
        
    print(f"Sampled {len(sample)} tweets.")
    sample.to_csv(output_csv, index=False)
    print(f"Saved sample to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample first-touch tweets for clustering")
    parser.add_argument("--input", type=str, default="../../data/processed/apple_support_tweets.csv")
    parser.add_argument("--output", type=str, default="../../data/processed/apple_support_first_touch_sample.csv")
    parser.add_argument("--size", type=int, default=1000)
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    sample_first_touch(input_path, output_path, args.size)
