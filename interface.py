import streamlit as st
import pandas as pd
import numpy as np
import nbimporter
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix
from NMF_Model import *  # Import your NMF model

# Load the dataset
df = pd.read_csv('./data/combined_final_dataset.csv')

# Game details dictionaries
game_to_image = df.set_index('game_name')['header_image'].to_dict()
game_to_description = df.set_index('game_name')['short_description'].to_dict()
game_to_genres = df.set_index('game_name')['genres'].to_dict()

# Pivot table for recommendation
pv_sparse, pv = create_pivot_table(df)

# Function for generating recommendations
def recommend_games(selected_games, num_recommendations):
    if len(selected_games) == 0:
        return []

    new_user_ratings = {'game_name': selected_games, 'rating': [5] * len(selected_games)}
    new_user_df = pd.DataFrame(new_user_ratings)
    new_user_df['user_id'] = 'new_user'

    df_with_new_user = pd.concat([df, new_user_df[['user_id', 'game_name', 'rating']]])
    pv_with_new_user = df_with_new_user.pivot_table(index=['user_id'], columns=['game_name'], values='rating')

    scaler = MinMaxScaler()
    pv_scaled_with_new_user = pd.DataFrame(
        scaler.fit_transform(pv_with_new_user.fillna(0)),
        columns=pv_with_new_user.columns,
        index=pv_with_new_user.index
    )

    pv_sparse_with_new_user = csr_matrix(pv_scaled_with_new_user.values)

    rmse, mae, recall_k, predicted_ratings = nmf_model_evaluation(pv_sparse_with_new_user)

    active_user_index = np.where(pv_with_new_user.index == 'new_user')[0][0]
    predicted_user_ratings = predicted_ratings[active_user_index]

    unrated_games_indices = [
        i for i in range(len(predicted_user_ratings)) if pv_with_new_user.columns[i] not in selected_games
    ]

    sorted_unrated_ratings = predicted_user_ratings[unrated_games_indices]
    sorted_indices = np.argsort(sorted_unrated_ratings)[::-1][:num_recommendations]

    recommended_games = [
        (pv_with_new_user.columns[unrated_games_indices[idx]], 
         sorted_unrated_ratings[idx], 
         game_to_image.get(pv_with_new_user.columns[unrated_games_indices[idx]], ''), 
         game_to_description.get(pv_with_new_user.columns[unrated_games_indices[idx]], ''), 
         game_to_genres.get(pv_with_new_user.columns[unrated_games_indices[idx]], '')) 
        for idx in sorted_indices
    ]

    return recommended_games

# Function to recommend unpopular games based on genre
def recommend_unpopular_games(recommended_games, num_recommendations):
    all_recommended_genres = set()
    for _, _, _, _, genres in recommended_games:
        if genres:
            all_recommended_genres.update(genres.split(','))

    unpopular_games = []
    for genre in all_recommended_genres:
        genre_games = df[df['genres'].str.contains(genre)]
        genre_games_sorted = genre_games.groupby('game_name').size().sort_values(ascending=True).head(num_recommendations)
        
        for game_name in genre_games_sorted.index:
            if game_name not in [game[0] for game in recommended_games]:
                unpopular_games.append((game_name, game_to_image.get(game_name, ''), 
                                        game_to_description.get(game_name, ''), game_to_genres.get(game_name, '')))
                if len(unpopular_games) >= num_recommendations:
                    break
        if len(unpopular_games) >= num_recommendations:
            break

    return unpopular_games

# Streamlit configuration
def main():
    st.set_page_config(layout="wide")

    # Custom CSS for buttons, modal, and hover effect
    st.markdown(
        """
        <style>
        .header-buttons {
            display: flex;
            justify-content: flex-end;
            gap: 10px;
            margin-top: -50px;
        }
        .stButton button {
            cursor: pointer;
            font-size: 18px;
            background-color: transparent;
            color: white;
            border-radius: 5px;
            border: 2px solid #ff6f61;
            padding: 10px 20px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            background-color: #ff6f61;
            color: black;
            border-color: #ff6f61;
        }
        .stApp {
            margin-top: 0;
        }
        /* Apply hover effect to image wrapper and enlarge image */
        .stImage div {
            position: relative;
            overflow: hidden;
            display: inline-block;
        }
        .stImage img {
            transition: transform 0.3s ease-in-out;
            width: 100%;
            height: auto;
            object-fit: cover;
        }
        .stImage div:hover img {
            transform: scale(1.1);
        }
        .modal-image {
            max-width: 80%;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Header navigation with buttons
    st.markdown("<h1 style='text-align: center;'>Video Games Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<div class='header-buttons'>", unsafe_allow_html=True)

    col_buttons = st.columns([1, 1, 1, 1, 1, 1, 1, 0.5, 0.9])
    with col_buttons[7]:
        if st.button("Home"):
            st.session_state.page = "Home"
    with col_buttons[8]:
        if st.button("Recommend Games"):
            st.session_state.page = "Recommend Games"

    st.markdown("</div>", unsafe_allow_html=True)

    # Set default page
    if "page" not in st.session_state:
        st.session_state.page = "Home"

    # Modal State
    if "modal" not in st.session_state:
        st.session_state.modal = {"visible": False, "content": None}

    def close_modal():
        st.session_state.modal["visible"] = False
        st.experimental_rerun()

    def open_modal(name, image, description, genres):
        st.session_state.modal["visible"] = True
        st.session_state.modal["content"] = {
            "name": name,
            "image": image,
            "description": description,
            "genres": genres,
        }

    if st.session_state.modal["visible"]:
        content = st.session_state.modal["content"]
        st.image(content["image"], use_column_width=False, width=600, caption="Header Image")
        st.markdown(f"### {content['name']}")
        st.markdown(f"**Description:** {content['description']}")
        st.markdown(f"**Genres:** {content['genres']}")
        if st.button("Close"):
            close_modal()

    if st.session_state.page == "Home":
        st.header("Top 10 Most Popular Games")
        top_games = df.groupby('game_name').size().sort_values(ascending=False).head(10)
        cols = st.columns(5)
        for idx, game_name in enumerate(top_games.index):
            with cols[idx % 5]:
                st.image(game_to_image.get(game_name, ''), use_column_width=True)
                st.markdown(f"**{game_name}**")
                if st.button("View Details", key=f"{game_name}", on_click=open_modal, args=(game_name, game_to_image.get(game_name, ''), game_to_description.get(game_name, ''), game_to_genres.get(game_name, ''))):
                    pass

    elif st.session_state.page == "Recommend Games":
        st.header("Select Games You Like")
        st.markdown(
            "<h3 style='font-size: 22px; color: #ff6f61; margin-bottom: -100px;'>🎮 You can choose more than 1 game you like:</h3>",
            unsafe_allow_html=True
        )
        selected_games = st.multiselect("", pv.columns.tolist())  # Keep the dropdown below the styled text

        st.markdown(
            "<h3 style='font-size: 22px; color: #ff6f61; margin-bottom: -100px;'>🎯 Number of games to recommend:</h3>",
            unsafe_allow_html=True
        )
        num_recommendations = st.radio("", [5, 10, 15, 20], horizontal=True, key="num_recommendations")

        if st.button("Generate Recommendations"):
            if selected_games:
                recommended_games = recommend_games(selected_games, st.session_state.num_recommendations)
                st.session_state.recommended_games = recommended_games

        if "recommended_games" in st.session_state:
            recommended_games = recommend_games(selected_games, num_recommendations)  # Use directly

            if recommended_games:
                st.markdown("<h2 style='text-align: center; color: #ff6f61;'>🎮 Recommended Games</h2>", unsafe_allow_html=True)
                st.markdown("<hr style='border: 2px solid #ff6f61;'>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                cols = st.columns(5)
                for idx, (name, _, image, description, genres) in enumerate(recommended_games):
                    with cols[idx % 5]:
                        st.image(image, use_column_width=True)
                        st.markdown(f"**{name}**")
                        if st.button("View Details", key=f"{name}", on_click=open_modal, args=(name, image, description, genres)):
                            pass

                st.markdown("<br><br>", unsafe_allow_html=True)
                st.markdown("<h2 style='text-align: center; color: #4287f5;'>🌟 Unpopular Games You Might Like</h2>", unsafe_allow_html=True)
                st.markdown("<hr style='border: 2px solid #4287f5;'>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                unpopular_games = recommend_unpopular_games(recommended_games, num_recommendations)
                if unpopular_games:
                    cols = st.columns(5)
                    for idx, (name, image, description, genres) in enumerate(unpopular_games):
                        with cols[idx % 5]:
                            st.image(image, use_column_width=True)
                            st.markdown(f"**{name}**")
                            if st.button("View Details", key=f"{name}", on_click=open_modal, args=(name, image, description, genres)):
                                pass

        if 'recommended_games' in st.session_state and st.button("Clear Recommendations"):
            # Clear the stored recommendations
            del st.session_state.recommended_games
            st.experimental_rerun()

if __name__ == "__main__":
    main()