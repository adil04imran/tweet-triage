import pandas as pd
import google.generativeai as genai
import argparse
import os
from dotenv import load_dotenv
import json

load_dotenv()

def discover_intents(input_csv, output_json, sample_size=100):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables. Please set it in .env")
    
    genai.configure(api_key=api_key)
    # Using a fast, reliable model
    model = genai.GenerativeModel("gemini-1.5-flash") 

    print(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)
    
    # We don't need all 1000 for discovering intents, 100 diverse samples is enough for an LLM to find patterns
    if len(df) > sample_size:
        sample_tweets = df.sample(n=sample_size, random_state=42)['text'].tolist()
    else:
        sample_tweets = df['text'].tolist()
        
    print(f"Sending {len(sample_tweets)} tweets to Gemini for clustering...")
    
    prompt = f"""
    You are an expert customer service analyst for Apple Support.
    Below is a sample of customer tweets reaching out to Apple Support.
    Your goal is to read these tweets and identify the top 5 to 10 most common intents (reasons for contacting support).
    
    For each intent, provide:
    1. A short, slug-like name (e.g., "account_locked", "battery_issue")
    2. A brief description of what this intent covers.
    3. 1-2 example tweets from the sample that fit this intent.
    
    Return ONLY a valid JSON array of objects with keys: "intent_name", "description", "examples".
    
    Sample Tweets:
    {json.dumps(sample_tweets, indent=2)}
    """
    
    try:
        response = model.generate_content(prompt)
        text = response.text.strip()
        # Clean up markdown code blocks if present
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()
            
        intents = json.loads(text)
        
        print(f"Discovered {len(intents)} intents.")
        with open(output_json, 'w') as f:
            json.dump(intents, f, indent=4)
        print(f"Saved intents to {output_json}")
        
    except Exception as e:
        print(f"Error during Gemini API call or JSON parsing: {e}")
        print("Raw response:")
        print(response.text if 'response' in locals() else "No response")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use Gemini to discover intents from a sample of tweets")
    parser.add_argument("--input", type=str, default="../../data/processed/apple_support_first_touch_sample.csv")
    parser.add_argument("--output", type=str, default="../../data/processed/discovered_intents.json")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    discover_intents(input_path, output_path)
