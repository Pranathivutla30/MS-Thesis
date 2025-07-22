import pandas as pd
import random
from data_loader import load_events
from sklearn.model_selection import train_test_split

def generate_counterfactuals(df, num_users, num_items, negative_ratio=1):
    user_item_set = set(zip(df['user_id'], df['item_id']))
    all_items = list(range(num_items))
    negatives = []
    for user, item in zip(df['user_id'], df['item_id']):
        for _ in range(negative_ratio):
            neg_item = random.choice(all_items)
            while (user, neg_item) in user_item_set:
                neg_item = random.choice(all_items)
            negatives.append((user, neg_item, 0))

    positives = list(zip(df['user_id'], df['item_id'], [1]*len(df)))
    df_aug = pd.DataFrame(positives + negatives, columns=['user_id', 'item_id', 'rating'])
    return df_aug.sample(frac=1).reset_index(drop=True)

def preprocess_and_split():
    df, num_users, num_items = load_events()
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df = generate_counterfactuals(train_df, num_users, num_items)
    return train_df, test_df, num_users, num_items

if __name__ == "__main__":
    train_df, test_df, num_users, num_items = preprocess_and_split()
    print("Preprocessing complete.")
    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    print(train_df['rating'].unique())
    print(f"Number of users: {num_users}")
    print(f"Number of items: {num_items}")
    print("First 5 rows of training set:")
    print(train_df.head())
