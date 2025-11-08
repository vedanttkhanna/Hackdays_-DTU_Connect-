import streamlit as st
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os
import json
import numpy as np
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# Streamlit setup
st.set_page_config(page_title="DTU Connect", layout="centered")
st.title("DTU Connect")

st.write("Find peers or societies that match your interests and skills.")

# Sidebar switch
mode = st.sidebar.radio("Choose what to find:", ["Peers", "Societies"])

# Input section
user_bio = st.text_area("Describe yourself – your interests, skills, or goals")

# Function to generate embeddings using Gemini
def get_embedding(text):
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=text
    )
    return np.array(result["embedding"])

if st.button("Find Matches"):
    if not user_bio.strip():
        st.warning("Please enter a short description.")
    else:
        # Get embedding for user
        user_emb = get_embedding(user_bio)

        if mode == "Peers":
            with open("peers.json") as f:
                peers = json.load(f)

            results = []
            for peer in peers:
                peer_text = peer["bio"] + " " + peer["skills"]
                peer_emb = get_embedding(peer_text)

                # Compute cosine similarity
                sim = np.dot(user_emb, peer_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(peer_emb))
                results.append((peer["name"], peer["bio"], peer["skills"], sim))

            # Sort top 5
            top_matches = sorted(results, key=lambda x: x[3], reverse=True)[:5]

            st.subheader("Recommended Peers")
            for name, bio, skills, score in top_matches:
                st.markdown(f"**{name}**")
                st.write(f"Bio: {bio}")
                st.write(f"Skills: {skills}")
                st.caption(f"Similarity: {round(score * 100, 1)}%")
                st.divider()

        else:  # Society mode
            with open("societies.json") as f:
                societies = json.load(f)

            results = []
            for soc in societies:
                soc_emb = get_embedding(soc["tags"])

                sim = np.dot(user_emb, soc_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(soc_emb))
                results.append((soc["name"], soc["tags"], sim))

            top_matches = sorted(results, key=lambda x: x[2], reverse=True)[:5]

            st.subheader("Recommended Societies")
            for name, tags, score in top_matches:
                st.markdown(f"**{name}** – {tags}")
                st.progress(float(score))
                st.caption(f"Match: {round(score * 100, 1)}%")