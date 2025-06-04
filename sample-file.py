import json
import pandas as pd

input_file = "yelp_academic_dataset_review.json"
output_file = "yelp_sample.csv"

# Extract 10,000 reviews
reviews = []
with open(input_file, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i >= 10000:
            break
        reviews.append(json.loads(line))

df = pd.DataFrame(reviews)
df[['text', 'stars']].to_csv(output_file, index=False)
