import streamlit as st
import json
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Streamlit setup
st.set_page_config(page_title="DTU Connect", layout="centered")

st.markdown("""
<style>
/* === Futuristic Black & Green Theme === */
.stApp {
    background-color: #000000;
    color: #00FF66;
    font-family: 'Courier New', monospace;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #000000 !important;
    color: #00FF66 !important;
    border-right: 1px solid #00FF66;
}
[data-testid="stSidebar"] * {
    color: #00FF66 !important;
}

/* Input fields (text + textarea) */
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background-color: #0a0a0a !important;
    color: #00FF66 !important;
    border: 1px solid #00FF66 !important;
    border-radius: 8px;
    font-family: 'Courier New', monospace;
}
div[data-baseweb="input"]:hover input,
div[data-baseweb="textarea"]:hover textarea {
    box-shadow: 0 0 10px #00FF66;
    border: 1px solid #00FF66;
}

/* Buttons */
div.stButton > button {
    background-color: #000;
    color: #00FF66;
    border: 1px solid #00FF66;
    border-radius: 8px;
    font-weight: bold;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #00FF66;
    color: #000000;
    transform: scale(1.05);
}

/* Headings with neon glow */
h1, h2, h3, h4 {
    color: #00FF66 !important;
    text-shadow: 0 0 10px #00FF66;
}

/* Divider and progress bar */
hr {
    border: 1px solid #00FF66;
}
div[data-testid="stProgressBar"] > div > div {
    background-color: #00FF66;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 10px;
}
::-webkit-scrollbar-thumb {
    background: #00FF66;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div[data-baseweb="textarea"] textarea {
    box-shadow: 0 0 8px #00FF66;
}
</style>
""", unsafe_allow_html=True)


st.title("DTU Connect")

st.write("Find peers or societies that match your interests, skills, and class!")

# Sidebar switch
mode = st.sidebar.radio("Choose what to find:", ["Peers", "Societies"])

# Input section
user_bio = st.text_area("Describe yourself – your interests, skills, or goals")
user_class = st.text_input("Enter your class (optional, e.g., EP SEC-2)")

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

                # Class bonus
                bonus = 0.2 if user_class.strip().lower() == peer["class"].strip().lower() else 0
                sim = min(sim + bonus, 1.0)

                results.append((peer["name"], peer["bio"], peer["skills"], peer["class"], sim))

            top_matches = sorted(results, key=lambda x: x[4], reverse=True)[:5]

            st.subheader(" Recommended Peers")
            selected_peers = []
            for name, bio, skills, peer_class, score in top_matches:
                st.markdown(f"**{name}** — *{peer_class}*")
                st.write(f"**Bio:** {bio}")
                st.write(f"**Skills:** {skills}")
                st.caption(f"Match Score: {round(score * 100, 1)}%")

                if st.checkbox(f"Add {name} to your Hackathon team"):
                    selected_peers.append(name)
                st.divider()

            # If user selected peers → form team
            if selected_peers:
                st.success(f"You selected {len(selected_peers)} teammates.")
                team_name = st.text_input("Enter your Hackathon Team Name")

                if st.button("Save Team"):
                    new_team = {
                        "team_name": team_name or "Unnamed Team",
                        "created_by": user_bio,
                        "members": selected_peers
                    }

                    # Save team data
                    if os.path.exists("hackathon_teams.json"):
                        with open("hackathon_teams.json", "r") as f:
                            teams = json.load(f)
                    else:
                        teams = []

                    teams.append(new_team)

                    with open("hackathon_teams.json", "w") as f:
                        json.dump(teams, f, indent=2)

                    st.success(f" Hackathon team '{team_name or 'Unnamed Team'}' saved successfully!")

        
        else:
            with open("societies.json") as f:
                societies = json.load(f)

            results = []
            for soc in societies:
                soc_emb = get_embedding(soc["tags"])
                sim = np.dot(user_emb, soc_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(soc_emb))
                results.append((soc["name"], soc["tags"], sim))

            top_matches = sorted(results, key=lambda x: x[2], reverse=True)[:5]

            st.subheader(" Recommended Societies")
            for name, tags, score in top_matches:
                st.markdown(f"**{name}** — {tags}")
                st.progress(float(score))
                st.caption(f"Match Strength: {round(score * 100, 1)}%")