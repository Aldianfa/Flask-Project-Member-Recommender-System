from flask import Flask, render_template
from flask_mysqldb import MySQL
import pandas as pd
import numpy as np
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer

# function preprocessing data cleaning, case folding, stopword, stemming from database and return to dataframe
def get_data():
    cur = MySQL.connection.cursor()
    cur.execute("SELECT * FROM mhs_raw")
    projects = cur.fetchall()
    cur.close()
    print("Data fetched successfully")
    return projects

# FUNGSI UNTUK MELAKUKAN CLEANING DATA
def cleaning(data):
    cleaned_data = []
    for project in data:
        cleaned_project = []
        for item in project:
            if isinstance(item, str):
                cleaned_item = re.sub(r'[^a-zA-Z0-9\s]', '', item)  # Menghapus karakter non-alphanumeric
                cleaned_item = ' '.join([w for w in cleaned_item.split() if len(w) > 3])
                cleaned_project.append(cleaned_item)
            else:
                cleaned_project.append(item)
        cleaned_data.append(cleaned_project)
    print("Data cleaned successfully")
    return cleaned_data

# FUNGSI UNTUK MELAKUKAN CASE FOLDING
def case_folding(data):
    folded_data = []
    for project in data:
        folded_project = [item.lower() if isinstance(item, str) else item for item in project]
        folded_data.append(folded_project)
    print("Data case folded successfully")
    return folded_data


# FUNGSI UNTUK MELAKUKAN STOPWORD REMOVAL
def stopword(data):
    stopword_factory = StopWordRemoverFactory()
    stopword_remover = stopword_factory.create_stop_word_remover()
    stopword_removed_data = []
    for project in data:
        stopword_removed_project = [stopword_remover.remove(item) if isinstance(item, str) else item for item in project]
        stopword_removed_data.append(stopword_removed_project)
    print("Stopwords removed successfully")
    return stopword_removed_data

# FUNGSI UNTUK MELAKUKAN STEMMING
def stemming(data):
    stemmer_factory = StemmerFactory()
    stemmer = stemmer_factory.create_stemmer()
    stemmed_data = []
    for project in data:
        stemmed_project = [stemmer.stem(item) if isinstance(item, str) else item for item in project]
        stemmed_data.append(stemmed_project)
    print("Data stemmed successfully")
    return stemmed_data

# FUNGSI UNTUK MEMBUAT KOLOM METADATA
def add_metadata(data):
    metadata = []
    for project in data:
        metadata.append(project[6] + ' ' + project[7] + ' ' + project[9] + ' ' + project[10] + ' ' + project[11])  
    print("Metadata added successfully")
    return metadata


def case_folding_input(data):
    if isinstance(data[0], list):  # Jika data adalah daftar dari daftar
        folded_data = []
        for project in data:
            folded_project = [item.lower() if isinstance(item, str) else item for item in project]
            folded_data.append(folded_project)
        print("Data case folded successfully")
        return folded_data
    else:  # Jika data hanya berisi satu proyek
        folded_project = [item.lower() if isinstance(item, str) else item for item in data]
        print("Data case folded successfully")
        return [folded_project]


def save_to_database(data):
    conn = MySQL.connection  # Menggunakan properti .connection untuk mendapatkan koneksi
    cur = conn.cursor()  # Menggunakan properti .cursor() dari objek koneksi
    cur.execute("""
        CREATE TABLE IF NOT EXISTS mhs_cleaned (
            id INT AUTO_INCREMENT PRIMARY KEY, 
            Email TEXT, 
            Nama TEXT, 
            NIM TEXT, 
            Semester TEXT, 
            NoWa TEXT, 
            Personality TEXT, 
            Bahasa TEXT, 
            Komunitas TEXT, 
            Role TEXT, 
            Proyek TEXT, 
            Deskripsi TEXT, 
            Github TEXT, 
            Metadata TEXT
        )
    """)
    for project in data:
        cur.execute("""
            INSERT INTO mhs_cleaned (
                Email, 
                Nama, 
                NIM, 
                Semester, 
                NoWa, 
                Personality, 
                Bahasa, 
                Komunitas, 
                Role, 
                Proyek, 
                Deskripsi, 
                Github, 
                Metadata
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            project[1], 
            project[2], 
            project[3], 
            project[4], 
            project[5], 
            project[6], 
            project[7], 
            project[8], 
            project[9], 
            project[10], 
            project[11], 
            project[12], 
            project[13]
        ))
    MySQL.connection.commit()
    cur.close()
    print("Data saved successfully")


# 
    


