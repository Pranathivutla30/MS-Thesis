import pandas as pd

def load_events(path="data/events.csv"):
    df = pd.read_csv(path)
    
    # Keep only rows with item IDs
    df = df.dropna(subset=['itemid'])

    # Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

    # Score mapping
    event_score = {'view': 1, 'addtocart': 3, 'transaction': 5}
    df['rating'] = df['event'].map(event_score)

    # Encode user and item IDs
    user_map = {uid: idx for idx, uid in enumerate(df['visitorid'].unique())}
    item_map = {iid: idx for idx, iid in enumerate(df['itemid'].unique())}

    df['user_id'] = df['visitorid'].map(user_map)
    df['item_id'] = df['itemid'].map(item_map)

    # Select final columns
    df_processed = df[['user_id', 'item_id', 'rating', 'timestamp', 'event']]

    # Return expected output shape
    return df_processed[['user_id', 'item_id', 'rating']], len(user_map), len(item_map)

# Test run
if __name__ == "__main__":
    df, n_users, n_items = load_events()
    print("Sample of processed events data:")
    print(df.head())
    print("\nTotal interactions:", len(df))
    print("Unique users:", n_users)
    print("Unique items:", n_items)
