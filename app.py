from io import StringIO
from flask import Flask, render_template, request, redirect
from flask_mysqldb import MySQL
import preprocessing as pre
import tfidf as tf
import cosine as cos
import pandas as pd
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# from config_data import get_data


app = Flask(__name__)

# KONFIGURASI DATABASE
app.config["MYSQL_HOST"] = "localhost"
app.config["MYSQL_USER"] = "root"
app.config["MYSQL_PASSWORD"] = ""
app.config["MYSQL_DB"] = "db_skripsi"
mysql = MySQL(app)


def get_data():
        cur = mysql.connection.cursor()
        cur.execute("SELECT * FROM mhs_raw2")
        project = cur.fetchall()
        cur.close()
        print("Data fetched successfully")
        return project

def get_data_cleaned():
        cur = mysql.connection.cursor()
        # cur.execute("SELECT * FROM mhs_data2")
        cur.execute("SELECT * FROM mhs_data2")
        projects = cur.fetchall()
        cur.close()
        df = pd.DataFrame(projects, columns=['No', 'Email', 'Nama', 'NIM', 'Semester', 'NoWa', 'Personality', 'Bahasa', 'Komunitas', 'Role', 'Proyek', 'Deskripsi', 'Github', 'metadata'])  # Menentukan kolom 'metadata' untuk DataFrame
        print("Data fetched successfully")
        return df

def preprocessing_input(text):
    # Cleaning: Menghapus karakter non-alfanumerik dan angka
    text = ''.join([char for char in text if char.isalnum() or char.isspace()])

    # Case Folding: Mengubah teks menjadi lowercase
    text = text.lower()

    # Stopword Removal
    factory = StopWordRemoverFactory()
    stopword_remover = factory.create_stop_word_remover()
    text = stopword_remover.remove(text)

    # Stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    text = stemmer.stem(text)

    return text




@app.route('/')
def main():
    return render_template('index.html')

@app.route('/rekomendasi')
def rekomendasi():
    return render_template('rekomendasi.html')

@app.route('/rekomkomunitas')
def rekomkomunitas():
    return render_template('rekomkomunitas.html')

@app.route('/data')
def data():
    mhs = get_data()
    return render_template('data.html', mhs=mhs)

@app.route('/preprocessing')
def preprocessing():
    data = get_data()
    data = pre.cleaning(data)
    data = pre.case_folding(data)
    data = pre.stopword(data)
    data = pre.stemming(data)
    metadata = pre.add_metadata(data)
    # save to database mhs_clean
    # data = pre.save_to_database(data)
    return render_template('preprocessing.html', data=data, metadata=metadata)

@app.route('/tfidf')
def tfidf():
    data = get_data_cleaned()
    result_tfidf = tf.calculate_tfidf(data)

    # Membulatkan nilai kolom 'Perhitungan TF', 'Perhitungan IDF', dan 'Perhitungan W' menjadi 3 angka di belakang koma
    # result_tfidf['Perhitungan TF'] = result_tfidf['Perhitungan TF'].round(3)
    # result_tfidf['Perhitungan IDF'] = result_tfidf['Perhitungan IDF'].round(3)
    # result_tfidf['Perhitungan W'] = result_tfidf['Perhitungan W'].round(3)

    return render_template('tfidf.html', tables=[result_tfidf.to_html(classes='data')], df=result_tfidf)


@app.route('/tfidf-detail/<int:index>')
def tfidf_detail(index):
    data = get_data_cleaned()
    result_tfidf = tf.tfidf_detail(data['metadata'], index)

    metadata = data['metadata'][index]
    jumlah_kata = len(metadata.split())

    return render_template('tfidf-detail.html', tables=[result_tfidf.to_html(classes='data')], df=result_tfidf, metadata=metadata, jumlah_kata=jumlah_kata)




@app.route('/cosine')
def cosine():
    data = get_data_cleaned()
    # result_cosine = cos.calculate_cosine_similarityy(data)
    result_cosine = cos.manual_cosine_similarity(data)

    # Membuat DataFrame dari hasil cosine similarity
    df = pd.DataFrame(result_cosine)

    # Membulatkan nilai cosine similarity menjadi 3 angka di belakang koma
    df=df.round(3)

    # Batasi DataFrame hanya hingga indeks 20
    df = df.iloc[:10, :10]

    return render_template('cosine.html', tables=[df.to_html(classes='data')], df=df)


@app.route('/user-recommendation', methods=['POST'])
def user_recommendation():
    if request.method == 'POST':
        # Mendapatkan input dari formulir
        personality = request.form['personality']
        listbhspemrograman = request.form.getlist('bahasapemrograman')
        role = request.form['role']
        project_name = request.form['projek']
        project_description = request.form['deskripsi']

        # Gabungkan semua bahasa pemrograman menjadi satu teks
        bhspemrograman = ' '.join(listbhspemrograman)

        # Gabungkan semua kriteria menjadi satu teks
        input_text = ' '.join([personality, bhspemrograman, role, project_name, project_description])

        # Lakukan preprocessing pada teks input
        preprocessed_text = preprocessing_input(input_text)

        # Dapatkan data mhs_data dari database
        mhs_data = get_data_cleaned()

        # Hitung TF, IDF, dan W = TF * (IDF + 1) menggunakan function dari file
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text, *mhs_data['metadata']])

        # Hitung cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

        # Ambil indeks 6 nilai cosine similarity tertinggi
        similar_projects_indices = cosine_similarities.argsort()[0][-8:][::-1]
        # Ambil indeks dengan nilai cosine similarity > 0.6
        # similar_projects_indices = np.where(cosine_similarities > 0.4)[1]
        # Ambil indeks 4 cosine similarity tertinggi dan 4 cosine terendah
        # similar_projects_indices = np.argpartition(cosine_similarities, -4)[-4:]

        # Ambil proyek yang paling mirip
        recommended_projects = mhs_data.iloc[similar_projects_indices]

        # Menambahkan kolom nilai cosine similarity ke dalam dataframe
        recommended_projects['cosine_similarity'] = cosine_similarities[0, similar_projects_indices]

        # nilai cosine 6 angka dibelakang koma
        recommended_projects['cosine_similarity'] = recommended_projects['cosine_similarity'].round(7)

        # Menampilkan hasil rekomendasi
        return render_template('recommendation-user.html', projects=recommended_projects.to_dict(orient='records'), 
                               preprocessed_text=preprocessed_text, personality=personality, bhspemrograman=bhspemrograman, 
                               role=role, project_name=project_name, project_description=project_description)


@app.route('/komunitas-recommendation', methods=['POST'])
def komunitas_recommendation():
    if request.method == 'POST':
        # Mendapatkan input dari formulir
        personality = request.form['personality']
        listbhspemrograman = request.form.getlist('bahasapemrograman')
        role = request.form['role']
        komunitas = request.form['komunitas']
        project_name = request.form['projek']
        project_description = request.form['deskripsi']

        # Gabungkan semua bahasa pemrograman menjadi satu teks
        bhspemrograman = ' '.join(listbhspemrograman)

        # Gabungkan semua kriteria menjadi satu teks
        input_text = ' '.join([personality, bhspemrograman, role, komunitas, project_name, project_description])

        # Lakukan preprocessing pada teks input
        preprocessed_text = preprocessing_input(input_text)

        # Dapatkan data mhs_data dari database
        mhs_data = get_data_cleaned()

        # Mencari data mahasiswa yang tergabung dalam salah satu komunitas yang diinput pengguna
        mhs_data = mhs_data[mhs_data['Komunitas'].str.contains(komunitas)]

        # Hitung TF, IDF, dan W = TF * (IDF + 1) menggunakan function dari file 
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform([preprocessed_text, *mhs_data['metadata']])

        # Hitung cosine similarity
        cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

        # Ambil indeks 6 nilai cosine similarity tertinggi
        similar_projects_indices = cosine_similarities.argsort()[0][-8:][::-1]

        # Ambil proyek yang paling mirip
        recommended_projects = mhs_data.iloc[similar_projects_indices]

        # Menambahkan kolom nilai cosine similarity ke dalam dataframe
        recommended_projects['cosine_similarity'] = cosine_similarities[0, similar_projects_indices]

        # Menampilkan hasil rekomendasi
        return render_template('recommendation-user.html', projects=recommended_projects.to_dict(orient='records'), 
                               preprocessed_text=preprocessed_text, personality=personality, bhspemrograman=bhspemrograman, 
                               role=role, komunitas=komunitas, project_name=project_name, project_description=project_description)





    


@app.route('/save_to_database', methods=['POST'])
def save_to_database():
    data = request.form  # Mengambil data dari formulir yang dikirim oleh pengguna

    # Lakukan penyimpanan data ke dalam database
    # Pastikan untuk mengganti bagian ini dengan logika sesuai kebutuhan Anda
    conn = mysql.connection.cursor()
    cur = conn
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
    cur.execute("""
        INSERT INTO mhs_cleaned (
            Email, Nama, NIM, Semester, NoWa, Personality, 
            Bahasa, Komunitas, Role, Proyek, Deskripsi, Github, Metadata
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        data['Email'], data['Nama'], data['NIM'], data['Semester'], data['NoWa'],
        data['Personality'], data['Bahasa'], data['Komunitas'], data['Role'],
        data['Proyek'], data['Deskripsi'], data['Github'], data['Metadata']
    ))
    conn.commit()
    cur.close()
    print("Data saved successfully")

    # Redirect user back to the preprocessing page with a success message
    return redirect('/preprocessing')







     



@app.route('/coba')
def hello_world():

    # Looping
    hari = ['Senin', 'Selasa', 'Rabu', 'Kamis', 'Jumat', 'Sabtu', 'Minggu']

    # Conditional
    nilai = 90

    return render_template('coba.html', nilai=nilai, hari=hari)

if __name__ == '__main__':
    app.run(debug=True)