'''
Author: Bappy Ahmed
Email: entbappy73@gmail.com
Date: 2021-Dec-18
'''

import joblib
import streamlit as st
import numpy as np
import pandas as pd

st.header('Book Recommender System Using Machine Learning')

# Load the model using joblib
model = joblib.load('artifacts/model.joblib')
book_names=joblib.load('artifacts/book_names.joblib')
book_pivot=joblib.load('artifacts/book_pivot.joblib')
final_rating=joblib.load('artifacts/final_rating.joblib')

def fetch_poster(suggestion):
    book_name = []
    ids_index = []
    poster_url = []

    # Fetch book names from suggestion
    for book_id in suggestion[0]:  # Adjust indexing for the returned suggestion
        book_name.append(book_pivot.index[book_id])

    # Find corresponding image URLs
    for name in book_name:
        ids = np.where(final_rating['title'] == name)[0][0]
        ids_index.append(ids)

    for idx in ids_index:
        url = final_rating.iloc[idx]['image_url']
        poster_url.append(url)

    return poster_url

def recommend_book(book_name):
    books_list = []
    book_id = np.where(book_pivot.index == book_name)[0][0]
    distance, suggestion = model.kneighbors(book_pivot.iloc[book_id, :].values.reshape(1, -1), n_neighbors=6)

    poster_url = fetch_poster(suggestion)

    for i in range(len(suggestion[0])):
        books = book_pivot.index[suggestion[0]]
        for j in books:
            books_list.append(j)
    return books_list, poster_url

selected_books = st.selectbox(
    "Type or select a book from the dropdown",
    book_names
)

if st.button('Show Recommendation'):
    recommended_books, poster_url = recommend_book(selected_books)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(recommended_books[0])
        st.image(poster_url[0])
    with col2:
        st.text(recommended_books[1])
        st.image(poster_url[1])
    with col3:
        st.text(recommended_books[2])
        st.image(poster_url[2])
    with col4:
        st.text(recommended_books[3])
        st.image(poster_url[3])
    with col5:
        st.text(recommended_books[4])
        st.image(poster_url[4])
