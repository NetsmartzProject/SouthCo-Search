from flask import Flask, request, jsonify
from flask_caching import Cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fuzzywuzzy import fuzz
import pandas as pd
import re
import json
from joblib import Parallel, delayed
from collections import defaultdict
import time

app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

class SassProduct:
    def __init__(self, data_file: str, file_type: str ='csv'):
        self.df = self.load_data(data_file, file_type)
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['name_lower'])
        self.word_mapping = {
            "ds": "DZUS®",
            "hnge": "Hinge",
        }
        self.create_indexes()

    def load_data(self, data_file: str, file_type: str):
        if file_type == "csv":
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        df['name_lower'] = df['name'].str.lower()  
        df['sku_lower'] = df['sku'].astype(str).str.lower() 
        return df[['id', 'name', 'name_lower', 'sku', 'sku_lower', 'price','image','description']]  

    def create_indexes(self):
        self.name_index = defaultdict(list)
        self.sku_index = {}
        for idx, row in self.df.iterrows():
            name_words = re.findall(r'\w+', str(row['name_lower']))
            for word in name_words:
                self.name_index[word].append(idx)
            self.sku_index[row['sku_lower']] = idx 

    @cache.memoize(timeout=300)
    def search_products(self, query, page, per_page):
        start_time = time.time()

        query_lower = query.lower().strip()
        if query_lower in self.word_mapping:
            query_lower = self.word_mapping[query_lower]

        # Direct SKU match
        if query_lower in self.sku_index:
            product = self.df.iloc[self.sku_index[query_lower]]
            return {
                "page": 1,
                "per_page": 1,
                "total_items": 1,
                "total_pages": 1,
                "products": [{
                    "id": int(product["id"]),
                    "sku": str(product["sku"]),
                    "name": product["name"],
                    "price": float(product["price"]),
                }]
            }

        search_results = []
        sku_matches = self.df[self.df['sku_lower'].str.contains(query_lower, regex=False)]
        search_results.extend(sku_matches.to_dict('records'))

        number_match = re.search(r'\b(?:no|number)?\s*(\d+)\b', query_lower)
        if number_match:
            number = number_match.group(1)
            exact_matches = self.df[
                (self.df['name_lower'].str.contains(rf'\b{number}\b', regex=True)) |
                (self.df['sku_lower'] == number)
            ]
            exact_results = exact_matches.to_dict('records')
            search_results = exact_results + [r for r in search_results if r['id'] not in [er['id'] for er in exact_results]]

        if len(search_results) < 50:
            query_vec = self.vectorizer.transform([query_lower])
            cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

            fuzzy_scores = [
                (idx, max(fuzz.partial_ratio(query_lower, name), fuzz.partial_ratio(query_lower, sku)))
                for idx, name, sku in zip(self.df.index, self.df['name_lower'], self.df['sku_lower'])
            ]

            combined_scores = [
                (idx, (cosine_similarities[idx] + score / 100) / 2)
                for idx, score in fuzzy_scores
            ]

            sorted_results = sorted(combined_scores, key=lambda x: x[1], reverse=True)
            top_results = sorted_results[:150]

            search_results.extend(self.df.iloc[idx].to_dict() for idx, score in top_results if score > 0.3 and self.df.iloc[idx]['id'] not in {r['id'] for r in search_results})

        unique_results = {product['id']: product for product in search_results}.values()
        unique_results = sorted(unique_results, key=lambda product: (
            0 if query_lower in str(product['name_lower']) or query_lower in str(product['sku_lower']) else 1,
            -len(str(product['name_lower'])) if query_lower in str(product['name_lower']) else -fuzz.partial_ratio(query_lower, str(product['name_lower']))
        ))

        remaining_products = self.df[~self.df['id'].isin({product['id'] for product in unique_results})].sample(frac=1).to_dict('records')
        unique_results = list(unique_results) + remaining_products

        total_items = len(unique_results)
        total_pages = (total_items + per_page - 1) // per_page

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_results = unique_results[start_idx:end_idx]
        print(paginated_results,"hi")

        results = [{
            "id": int(product["id"]),
            "sku": str(product.get("sku", "N/A")),
            "name": product.get("name", "Unnamed Product"),
            "price": float(product.get("price", 0)),
            "description": product.get("description", "No description"),
            "image": product.get("image", "No image"),

        } for product in paginated_results]

        end_time = time.time()
        print(f"Time taken: {end_time - start_time:.4f} seconds")

        return {
            "page": page,
            "per_page": per_page,
            "total_items": total_items,
            "total_pages": total_pages,
            "products": results
        }


# import pandas as pd
# import numpy as np
# from functools import lru_cache
# from fuzzywuzzy import fuzz
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import time
# import re

# class SassProduct:
#     def __init__(self, csv_file):
#         self.df = pd.read_csv(csv_file)
#         self.df['name_lower'] = self.df['name'].str.lower()
#         self.df['sku_lower'] = self.df['sku'].astype(str).str.lower()
        
#         # Create indexes
#         self.sku_index = {sku: idx for idx, sku in enumerate(self.df['sku_lower'])}
        
#         # Create name index with proper handling of multiple matches
#         self.name_index = self.df['name_lower'].str.split().explode().reset_index()
#         self.name_index.columns = ['idx', 'word']
#         self.name_index = self.name_index.groupby('word')['idx'].apply(list).reset_index()
#         self.name_index.set_index('word', inplace=True)

#         # Prepare TF-IDF
#         self.vectorizer = TfidfVectorizer()
#         self.tfidf_matrix = self.vectorizer.fit_transform(self.df['name_lower'])

#     @lru_cache(maxsize=1000)
#     def search_products(self, query, page=1, per_page=20):
#         start_time = time.time()

#         query_lower = query.lower().strip()

#         # Direct SKU match
#         if query_lower in self.sku_index:
#             product = self.df.iloc[self.sku_index[query_lower]]
#             return self._format_results([product], 1, 1)

#         results = set()

#         # Exact word matches in name
#         query_words = query_lower.split()
#         for word in query_words:
#             if word in self.name_index.index:
#                 results.update(self.name_index.loc[word, 'idx'])

#         # SKU partial matches
#         sku_matches = self.df.index[self.df['sku_lower'].str.contains(query_lower, regex=False)]
#         results.update(sku_matches)

#         # Number matching
#         number_match = re.search(r'\b(?:no|number)?\s*(\d+)\b', query_lower)
#         if number_match:
#             number = number_match.group(1)
#             exact_matches = self.df.index[
#                 (self.df['name_lower'].str.contains(rf'\b{number}\b', regex=True)) |
#                 (self.df['sku_lower'] == number)
#             ]
#             results.update(exact_matches)

#         # If we have less than 50 results, perform fuzzy matching
#         if len(results) < 50:
#             query_vec = self.vectorizer.transform([query_lower])
#             cosine_similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
#             # Use numpy for faster operations
#             name_scores = np.array([fuzz.partial_ratio(query_lower, name) for name in self.df['name_lower']])
#             sku_scores = np.array([fuzz.partial_ratio(query_lower, sku) for sku in self.df['sku_lower']])
#             fuzzy_scores = np.maximum(name_scores, sku_scores)

#             combined_scores = (cosine_similarities + fuzzy_scores / 100) / 2
#             top_indices = combined_scores.argsort()[-150:][::-1]
            
#             results.update(top_indices[combined_scores[top_indices] > 0.3])

#         # Sort results
#         sorted_results = sorted(results, key=lambda idx: (
#             0 if query_lower in self.df.loc[idx, 'name_lower'] or query_lower in self.df.loc[idx, 'sku_lower'] else 1,
#             -len(self.df.loc[idx, 'name_lower']) if query_lower in self.df.loc[idx, 'name_lower'] else -fuzz.partial_ratio(query_lower, self.df.loc[idx, 'name_lower'])
#         ))

#         # Paginate results
#         total_items = len(sorted_results)
#         total_pages = (total_items + per_page - 1) // per_page
#         start_idx = (page - 1) * per_page
#         end_idx = start_idx + per_page
#         paginated_results = self.df.iloc[sorted_results[start_idx:end_idx]]

#         end_time = time.time()
#         print(f"Time taken: {end_time - start_time:.4f} seconds")

#         return self._format_results(paginated_results, page, total_pages, per_page, total_items)

#     def _format_results(self, products, page, total_pages, per_page=20, total_items=None):
#         results = [{
#             "id": int(product["id"]),
#             "sku": str(product["sku"]),
#             "name": product["name"],
#             "price": float(product["price"]),
#         } for _, product in products.iterrows()]

#         return {
#             "page": page,
#             "per_page": per_page,
#             "total_items": total_items or len(results),
#             "total_pages": total_pages,
#             "products": results
#         }

