from data_loader import load_events

df = load_events()
print(df.head())
print("Users:", df['user_id'].nunique())
print("Items:", df['item_id'].nunique())
print("Ratings:", df.shape[0])
