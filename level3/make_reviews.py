"""Builds a demo customer-review dataset (labels are NOT used for scoring; only for a sanity check)."""
import random, pandas as pd
random.seed(7)
pos = ["I absolutely love this {p}, it works perfectly and the quality is excellent!",
       "Fantastic {p}! Great value, fast delivery and amazing customer service.",
       "Very happy with the {p}. Highly recommend, it exceeded my expectations.",
       "Wonderful experience, the {p} is reliable and beautifully made."]
neg = ["Terrible {p}, it broke after two days and the quality is awful.",
       "Worst purchase ever. The {p} is useless and I am totally disappointed.",
       "Very poor service and a defective {p}. I hate it, complete waste of money.",
       "Disappointing and overpriced {p}, arrived late and damaged."]
neu = ["The {p} is okay, nothing special.", "It is an average {p}, does the job.",
       "The {p} arrived on Tuesday in a standard box.", "Decent {p} for the price, neither good nor bad."]
prods = ["phone", "laptop", "blender", "jacket", "headphones", "keyboard", "backpack", "camera"]
rows = [(random.choice(t).format(p=random.choice(prods)), l) for l, t, n in
        [("positive", pos, 40), ("negative", neg, 40), ("neutral", neu, 20)] for _ in range(n)]
random.shuffle(rows)
pd.DataFrame(rows, columns=["review", "true_label"]).to_csv("reviews.csv", index=False)
