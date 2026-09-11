import pandas as pd
import json
import argparse
import os
import random

def create_golden_set(input_csv, intents_json, output_csv, size=150):
    df = pd.read_csv(input_csv)
    with open(intents_json, 'r') as f:
        intents = json.load(f)
        
    intent_names = [i['intent_name'] for i in intents]
    
    # Sample tweets
    if len(df) > size:
        sample = df.sample(n=size, random_state=42)
    else:
        sample = df

    # Prepare golden set dataframe
    golden_df = sample[['tweet_id_float', 'text', 'created_at']].copy()
    golden_df.rename(columns={'tweet_id_float': 'tweet_id'}, inplace=True)
    
    # Simple heuristic labelling (Mocking human labelling)
    def label_intent(text):
        text_lower = str(text).lower()
        if 'battery' in text_lower or 'drain' in text_lower or 'power' in text_lower:
            return 'battery_drain'
        elif 'update' in text_lower or 'ios' in text_lower or 'install' in text_lower:
            return 'ios_update_issue'
        elif 'payment' in text_lower or 'charge' in text_lower or 'subscription' in text_lower:
            return 'billing_and_subscriptions'
        elif 'crash' in text_lower or 'glitch' in text_lower or 'freeze' in text_lower or 'app' in text_lower:
            return 'app_crash_or_glitch'
        elif 'i problem' in text_lower or 'glitch' in text_lower or 'feature' in text_lower:
            return 'feature_complaint'
        else:
            return 'general_help'
            
    def should_escalate(text):
        text_lower = str(text).lower()
        # Escalate if angry or complex
        if any(word in text_lower for word in ['fuck', 'shit', 'bullshit', 'sue', 'lawyer', 'refund']):
            return True, 'High customer frustration or legal/refund request'
        if 'payment' in text_lower or 'subscription' in text_lower:
            return True, 'Billing issues require human verification'
        return False, ''

    golden_df['true_intent'] = golden_df['text'].apply(label_intent)
    
    escalation = golden_df['text'].apply(should_escalate)
    golden_df['should_escalate'] = [e[0] for e in escalation]
    golden_df['escalation_reason'] = [e[1] for e in escalation]
    
    # Historical reply needs to be fetched from the main dataset
    # but for first touch, we can fetch the actual AppleSupport reply
    # Let's load the main dataset to get the actual replies
    main_csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/processed/apple_support_tweets.csv')
    main_df = pd.read_csv(main_csv_path, low_memory=False)
    
    # AppleSupport replies have in_response_to_tweet_id matching our tweet_id
    historical_replies = []
    for t_id in golden_df['tweet_id']:
        reply_row = main_df[(main_df['in_response_to_tweet_id'] == t_id) & (main_df['author_id'] == 'AppleSupport')]
        if not reply_row.empty:
            historical_replies.append(reply_row.iloc[0]['text'])
        else:
            historical_replies.append("No historical reply found.")
            
    golden_df['historical_reply'] = historical_replies
    
    golden_df.to_csv(output_csv, index=False)
    print(f"Created Golden Evaluation Set at {output_csv} with {len(golden_df)} examples.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="../../data/processed/apple_support_first_touch_sample.csv")
    parser.add_argument("--intents", type=str, default="../../data/processed/discovered_intents.json")
    parser.add_argument("--output", type=str, default="../../eval/golden_eval_set.csv")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    intents_path = os.path.join(script_dir, args.intents) if not os.path.isabs(args.intents) else args.intents
    output_path = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    create_golden_set(input_path, intents_path, output_path)
