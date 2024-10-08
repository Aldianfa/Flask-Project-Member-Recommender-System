from flask import Flask, render_template
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
import tfidf as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# FUNGSI UNTUK COSINE SIMILARITY
def calculate_cosine_similarity(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])
    tfidf_matrix = tfidf_matrix.toarray()

    # Hitung cosine similarity
    cosine_similarity = np.dot(tfidf_matrix, tfidf_matrix.T)
    return cosine_similarity


def calculate_cosine_similarityy(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Hitung cosine similarity
    result_cosine_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return result_cosine_similarity

# make a cosine similarity function that takes in a data frame and returns a cosine similarity matrix with A.B/(||A||.||B||) formula step by step from value of tf, idf, w from tfidf_detail function in tfidf.py
def calculate_cosine_similarity3(data):
    tfidf_matrix = tf.tfidf_detail2(data)
    tfidf_matrix = tfidf_matrix.toarray()
    
    # Hitung cosine similarity
    dot_product = np.dot(tfidf_matrix, tfidf_matrix.T)
    norm = np.linalg.norm(tfidf_matrix, axis=1)
    cosine_sim = dot_product / np.outer(norm, norm)
    
    # Mengatasi NaN akibat pembagian dengan nol
    cosine_sim = np.nan_to_num(cosine_sim)
    
    return cosine_sim

def manual_cosine_similarity(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])
    
    # Convert the sparse matrix to a dense array
    tfidf_array = tfidf_matrix.toarray()
    
    # Calculate dot product
    dot_product = np.dot(tfidf_array, tfidf_array.T)
    
    # Calculate the norm
    norm = np.linalg.norm(tfidf_array, axis=1)
    
    # Calculate cosine similarity
    cosine_sim = dot_product / np.outer(norm, norm)
    
    # Handle division by zero
    cosine_sim = np.nan_to_num(cosine_sim)
    
    return cosine_sim


