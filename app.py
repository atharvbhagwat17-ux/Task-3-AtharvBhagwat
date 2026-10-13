
import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==========================================
# Page Settings
# ==========================================

st.set_page_config(
    page_title="AI Career Recommender",
    page_icon="🚀",
    layout="wide"
)


# ==========================================
# Custom Minimalist Theme
# ==========================================

st.markdown("""
<style>
    /* Overall page */
    .stApp {
        background-color: #F5F6F8;
        color: #263445;
    }

    /* Main content width */
    .block-container {
        max-width: 1100px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }

    /* Headings */
    h1, h2, h3 {
        color: #263445 !important;
        font-weight: 650;
    }

    /* Body text */
    p, label {
        color: #465568;
    }

    /* Hero section */
    .hero {
        background-color: #FFFFFF;
        border: 1px solid #E0E5EB;
        border-radius: 16px;
        padding: 2.2rem;
        margin-bottom: 1.5rem;
    }

    .hero-label {
        color: #718096;
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 2px;
        margin-bottom: 0.8rem;
    }

    .hero h1 {
        color: #263445 !important;
        font-size: 2.5rem;
        line-height: 1.2;
        margin: 0 0 0.8rem 0;
    }

    .hero p {
        color: #596779;
        font-size: 1rem;
        line-height: 1.7;
        margin-bottom: 0;
    }

    /* Buttons */
    .stButton > button {
        background-color: #40566C !important;
        color: #FFFFFF !important;
        border: 1px solid #40566C !important;
        border-radius: 8px;
        padding: 0.6rem 1.1rem;
        font-weight: 600;
        transition: background-color 0.2s ease;
    }

    .stButton > button p,
    .stButton > button span {
        color: #FFFFFF !important;
    }

    .stButton > button:hover {
        background-color: #304457 !important;
        border-color: #304457 !important;
    }

    /* Skill selector */
    div[data-baseweb="select"] > div {
        background-color: #FFFFFF !important;
        border-color: #D6DCE3 !important;
        border-radius: 8px;
    }

    /* Selected skill tags */
    span[data-baseweb="tag"] {
        background-color: #E2E8F0 !important;
        color: #334155 !important;
        border: 1px solid #CBD5E1;
    }

    /* Dropdown menu */
    div[data-baseweb="popover"],
    div[data-baseweb="menu"] {
        background-color: #FFFFFF !important;
    }

    div[role="option"] {
        background-color: #FFFFFF !important;
        color: #263445 !important;
    }

    div[role="option"]:hover {
        background-color: #E8EDF2 !important;
    }

    /* Dashboard statistics */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E5EB;
        border-radius: 12px;
        padding: 1rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #64748B;
    }

    div[data-testid="stMetricValue"] {
        color: #40566C;
        font-weight: 650;
    }

    /* Recommendation cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E5EB;
        border-radius: 12px;
    }

    /* Progress bars */
    div[data-testid="stProgress"] > div {
        background-color: #E1E6EB;
        border-radius: 8px;
    }

    div[data-testid="stProgress"] > div > div > div {
        background-color: #71879B;
        border-radius: 8px;
    }

    /* Divider */
    hr {
        border-color: #DEE3E9;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# Load Dataset
# ==========================================

try:
    data = pd.read_csv("raw_skills.csv")

except FileNotFoundError:
    st.error(
        "Could not find raw_skills.csv. "
        "Keep it in the same folder as app.py."
    )
    st.stop()

required_columns = {"job_role", "skills"}

if not required_columns.issubset(data.columns):
    st.error(
        "The CSV file must contain 'job_role' and 'skills' columns."
    )
    st.stop()

data = data.dropna(
    subset=["job_role", "skills"]
).reset_index(drop=True)

if data.empty:
    st.error("No career roles were found in the dataset.")
    st.stop()


# ==========================================
# Extract Available Skills
# ==========================================

all_skills = set()

for skill_list in data["skills"]:
    for skill in str(skill_list).split(","):
        if skill.strip():
            all_skills.add(skill.strip())

all_skills = sorted(all_skills)


# ==========================================
# Hero Section
# ==========================================

st.markdown("""
<div class="hero">
    <div class="hero-label">
        CAREER DISCOVERY · AI RECOMMENDATIONS
    </div>

    <h1>Find your next career direction.</h1>

    <p>
        Explore technology careers that align with your skills.
        Select your strengths and let the recommendation engine
        identify your closest matches.
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# Dashboard Statistics
# ==========================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Career Paths",
        value=len(data)
    )

with col2:
    st.metric(
        label="Available Skills",
        value=len(all_skills)
    )

with col3:
    st.metric(
        label="Minimum Skills Required",
        value=3
    )


st.divider()


# ==========================================
# Skill Selection
# ==========================================

st.subheader("Select Your Skills")

st.write(
    "Choose the technical skills you currently know. "
    "You must select at least three."
)

selected_skills = st.multiselect(
    "Technical skills",
    options=all_skills,
    placeholder="Search and select your skills...",
)

st.write(f"Skills selected: {len(selected_skills)}")


# ==========================================
# Recommendation System
# ==========================================

if st.button(
    "Find My Career Matches",
    type="primary"
):

    if len(selected_skills) < 3:
        st.warning(
            "Please select at least 3 skills before continuing."
        )

    else:
        # Convert career skills into numerical vectors
        vectorizer = TfidfVectorizer()

        tfidf_matrix = vectorizer.fit_transform(
            data["skills"]
        )

        # Convert selected user skills into a vector
        user_skills = ", ".join(selected_skills)

        user_vector = vectorizer.transform(
            [user_skills]
        )

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(
            user_vector,
            tfidf_matrix
        )[0]

        # Rank careers from highest to lowest similarity
        ranked_roles = sorted(
            range(len(data)),
            key=lambda i: similarity_scores[i],
            reverse=True
        )

        # Display top three recommendations
        st.divider()

        st.subheader("Your Top 3 Career Matches")

        st.write(
            "These career paths have the closest skill profiles "
            "to your selection."
        )

        for rank, i in enumerate(
            ranked_roles[:3],
            start=1
        ):
            role = data["job_role"].iloc[i]
            score = similarity_scores[i]

            # Find the selected skills that overlap with this role
            role_skills = [
                skill.strip().lower()
                for skill in data["skills"].iloc[i].split(",")
            ]

            matched_skills = [
                skill
                for skill in selected_skills
                if skill.lower() in role_skills
            ]

            # Recommendation card
            with st.container(border=True):

                left, right = st.columns([3, 1])

                with left:
                    st.markdown(
                        f"### {rank:02d}. {role}"
                    )

                    st.caption("Matching skills")

                    if matched_skills:
                        st.write(
                            ", ".join(matched_skills)
                        )
                    else:
                        st.write("No exact skill matches")

                with right:
                    st.metric(
                        label="Similarity Score",
                        value=f"{score:.3f}"
                    )

                st.progress(float(score))

        st.caption(
            "Similarity scores indicate how closely your selected "
            "skills match the career descriptions in the dataset. "
            "They are not probabilities of career success."
        )
