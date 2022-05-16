import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Minat Belajar",
                    page_icon=":smiley:")
st.write("""
# Prediksi Minat Belajar Matematika

Aplikasi ini digunakan untuk memprediksi minat belajar matematika pada siswa.

Projek ini dikembang untuk memenuhi tugas skripsi oleh [Saiful Islam](https://www.linkedin.com/in/saiful-islam-9326a220a/), NIM 1842200011, di Prodi Pendidikan Matematika Universitas Nurul Jadid Paiton Probolinggo. Data set yang di kelola adalah data siswa SMP Islam Raudlatul Ulum Gunggungan Kidul Kecamatan Pakuniran Kabupaten Probolinggo Tahun Pelajaran 2021-2022.
""")

st.sidebar.header('Silahkan Input Disini')

# Collects user input features into dataframe

def user_input_features():
    domisili = st.sidebar.selectbox('Kamu tinggal dimana ?',('Gunung Malang','Krajan','Dawuhan','Dulukan','Gabus Utara Ranon','Darmoyo Patemon','Pancor',))
    belajar = st.sidebar.selectbox('Apakah kamu belajar ?',('Iya','Tidak', 'Jarang'))
    cita_cita = st.sidebar.selectbox('Apakah kamu mempunyai cita-cita ?',('Ya','Tidak'))

    kelas = st.sidebar.slider('Kamu kelas berapa ? ', 7,9,8)
    durasi_belajar = st.sidebar.slider('Berapa lama (dalam menit) kamu belajar ? ', 0,120,30)
    jumlah_keluarga = st.sidebar.slider('Jumlah keluargamu ada berapa ? (Dalam satu rumah)', 0,6,3)
    data = {'domisili': domisili,
            'belajar': belajar,
            'cita_cita': cita_cita,
            'kelas': kelas,
            'durasi_belajar': durasi_belajar,
            'jumlah_keluarga': jumlah_keluarga}
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
minat_raw = pd.read_csv('minat_cleaned.csv', delimiter=';', skiprows=0, low_memory=False)
minat = minat_raw.drop(columns=['minat_belajar'])
df = pd.concat([input_df,minat],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['domisili','belajar','cita_cita']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('Hasil Input Pengguna')


st.write(df)

# Reads in saved classification model
load_clf = pickle.load(open('minat_clf.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Hasil Prediksi')
minat_belajar = np.array(['Iya','Tidak'])
st.write(minat_belajar[prediction])

st.subheader('Peluang Kebenaran Prediksi')
st.write(prediction_proba)
