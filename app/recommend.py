import pandas as pd
import numpy as np
import json
from flask import Flask, request, jsonify, abort
from sklearn.feature_extraction.text import TfidfVectorizer
import faiss
from typing import List, Dict, Any

app = Flask(__name__)

class RecommendProduct:
    def __init__(self, data_file: str, file_type: str = 'csv'):
        self.df = self.load_data(data_file, file_type)
        self.vectorizer = TfidfVectorizer()
        self.df['text'] = (self.df['name'].fillna('') + ' ' +
                           self.df['sku'].fillna('') + ' ' +
                           self.df['description'].fillna(''))
        self.tfidf_matrix = self.vectorizer.fit_transform(self.df['text'])
        self.index = faiss.IndexFlatL2(self.tfidf_matrix.shape[1])
        self.index.add(self.tfidf_matrix.toarray().astype(np.float32))

    def load_data(self, data_file: str, file_type: str):
        if file_type == "csv":
            df = pd.read_csv(data_file)
        else:
            raise ValueError(f"Unsupported file type: {file_type}")
        return df[['id', 'sku', 'name', 'description', 'price','image']]

    def get_similar_products(self, product_id: int, num_similar: int = 12) -> List[Dict[str, Any]]:
        if product_id not in self.df['id'].values:
            raise ValueError("Product not found")

        product_index = self.df.index[self.df['id'] == product_id].tolist()
        if not product_index:
            raise ValueError("Product not found")
        product_index = product_index[0]

        distances, indices = self.index.search(self.tfidf_matrix[product_index].toarray().astype(np.float32), num_similar + 1)

        similar_indices = indices[0][1:]

        similar_products = self.df.iloc[similar_indices].copy().to_dict(orient='records')
        
        
        formatted_products = []
        for product in similar_products:
            sku = product.get("sku", "N/A")
            name = product.get("name", "Unnamed Product")
            price = product.get("price", "N/A")
            description = product.get("description", "No description available")
            image=product.get("image","No image")
            formatted_products.append({
                "id": product.get("id"),
                "sku": sku,
                "name": name,
                "price": price,
                "image":image,
                "description": description,
            })

        return formatted_products
