import streamlit as st

st.title("🎬 Movie Recommender")

user_id = st.number_input("Enter User ID", min_value=1, max_value=1000)

if st.button("Recommend"):
    recs = recommend_movies(user_id)
    for movie in recs:
        st.write(movie)
