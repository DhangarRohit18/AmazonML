import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

print("Generating random data...")
# 4 million strings
data = ["amazon india private limited " + str(i) for i in range(4000000)]

print("Training TfidfVectorizer...")
t0 = time.time()
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,3), min_df=2)
X = vec.fit_transform(data)
print(f"Time taken: {time.time() - t0:.2f} seconds")
print(f"Matrix shape: {X.shape}")
