from flask import Flask, render_template
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


# def calculate_tfidf_new(data):
#     vectorizer = TfidfVectorizer()
#     tfidf_matrix = vectorizer.fit_transform(data['metadata'])

#     # Dapatkan nilai TF dari matriks TF-IDF
#     tf_values = tfidf_matrix.toarray().sum(axis=0)  # Sum across documents to get TF values for each term
#     total_words = tfidf_matrix.sum()

#     # Menghitung nilai TF dengan membagi frekuensi kata dengan total kata dalam dokumen
#     tf_values /= total_words

#     # Dapatkan nilai IDF dari objek vectorizer
#     idf_values = np.array(vectorizer.idf_)

#     # Hitung nilai W
#     w_values = tf_values * (idf_values + 1)

#     # Menambahkan kolom Nama
#     nama = data['Nama'].iloc[0]  # Menggunakan nama dari baris pertama, sesuaikan dengan kebutuhan

#     # Menampilkan hasil perhitungan TF, IDF, dan W urut mulai dari indeks 0
#     result_df = pd.DataFrame({
#         'Nama': [nama] * len(vectorizer.get_feature_names_out()),
#         'Perhitungan TF': tf_values,
#         'Perhitungan IDF': idf_values,
#         'Perhitungan W': w_values
#     })

#     return result_df

def tfidf_detail_semuakata(data, index):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data)

    # Dapatkan term dari matrix tf-idf
    terms = tfidf.get_feature_names_out()

    # Index yang ingin ditampilkan
    index_data = index
    tfidf_vector = tfidf_matrix[index_data]

    # Inisialisasi list untuk menyimpan hasil
    result = []

    # Loop untuk menghitung TF-IDF per kata
    for term_index, term in enumerate(terms):
        tf = tfidf_matrix[index_data, term_index]
        idf = tfidf.idf_[term_index]
        w = tf * (idf + 1)

        # Simpan hasil perhitungan dengan format tiga angka di belakang koma
        tf = "{:.3f}".format(tf)
        idf = "{:.3f}".format(idf)
        w = "{:.3f}".format(w)

        result.append({'Term': term, 'Perhitungan TF': tf, 'Perhitungan IDF': idf, 'Perhitungan W': w})
    
    # Simpan hasil ke dalam DataFrame
    result_df = pd.DataFrame(result)
    return result_df

# Fungsi untuk menghitung TF-IDF
def calculate_tfidf(data):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data['metadata'])

    # Dapatkan nilai TF dari matriks TF-IDF
    tf_values = tfidf_matrix.sum(axis=1).A1

    # Dapatkan nilai IDF dari objek vectorizer
    idf_values = np.array(vectorizer.idf_)

    # Pastikan tf_values dan idf_values memiliki panjang yang sama
    min_length = min(len(tf_values), len(idf_values))
    tf_values = tf_values[:min_length]
    idf_values = idf_values[:min_length]

    # Hitung nilai W
    w_values = tf_values * (idf_values + 1)

    # Menambahkan kolom Index
    nama = data['Nama']
    index = data.index

    # Menampilkan hasil perhitungan TF,IDF, dan W urut mulai dari indeks 0
    result_df = pd.DataFrame({
        'Dokumen' : index,
        'Nama': nama
    })

    return result_df

def tfidf_detail(data, index=None):
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(data)

    # Dapatkan term dari matrix tf-idf
    terms = tfidf.get_feature_names_out()

    # Index yang ingin ditampilkan
    index_data = index
    tfidf_vector = tfidf_matrix[index_data]

    # Inisialisasi list untuk menyimpan hasil
    result = []

    # Perulangan untuk menghitung TF-IDF
    for term_index in tfidf_vector.indices:
        terms_name = terms[term_index]
        tf = data[index_data].count(terms_name) / len(data[index_data].split())  # Hitung nilai TF dengan membagi jumlah kata yang muncul dengan total kata dalam dokumen
        df = sum(1 for doc in data if terms_name in doc.split())  # Hitung jumlah dokumen yang mengandung term tersebut
        idf = np.log(len(data) / df) + 1  # Hitung nilai IDF
        w = tf * (idf)

        # Simpan hasil perhitungan dengan format tiga angka di belakang koma
        tf = "{:.3f}".format(tf)
        idf = "{:.3f}".format(idf)
        w = "{:.3f}".format(w)

        result.append({'Term': terms_name, 'Perhitungan TF': tf, 'Perhitungan IDF': idf, 'Perhitungan W': w})
    
    # Simpan hasil ke dalam DataFrame
    result_df = pd.DataFrame(result)

    return result_df


    

