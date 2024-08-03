import pandas as pd 
import numpy as np
import json 
from flask import Flask , request,jsonify
from app.search import SassProduct
from app.recommend import RecommendProduct
from flask import Flask, request, jsonify, abort
import math


app = Flask(__name__)

data_file_csv = r"C:\SOUTHCO\montes.csv"
product = SassProduct(data_file_csv)
product_service = RecommendProduct(data_file_csv)

df = pd.read_csv(data_file_csv, usecols=['id', 'sku', 'name', 'description','image'])


@app.route('/Allproducts', methods=['GET'])
def get_products():
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    start = (page - 1) * per_page
    end = start + per_page

    total_products = df.shape[0]
    total_pages = math.ceil(total_products / per_page)

    products = df.iloc[start:end].to_dict(orient='records')

    response = {
        'total': total_products,
        'page': page,
        'per_page': per_page,
        'total_pages':total_pages,
        'products': products
    }

    return jsonify(response)

@app.route('/search', methods=['GET'])
def search():
    query = request.args.get('query', '')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    if not query:
        return jsonify({"error": "Query parameter is required"}), 400

    results = product.search_products(query, page, per_page)
    return jsonify(results)


@app.route("/similar_products", methods=["POST"])
def get_similar_products():
    if not request.is_json:
        abort(400, description="Request must be JSON")
    
    data = request.get_json()
    product_id = data.get('product_id')
    if not product_id:
        abort(400, description="Product ID is required.")
    
    try:
        results = product_service.get_similar_products(product_id)
    except ValueError as e:
        abort(404, description=str(e))
    
    if not results:
        abort(404, description="No similar products found.")
    
    return jsonify(results), 200


if __name__ == '__main__':
    app.run(debug=True)
