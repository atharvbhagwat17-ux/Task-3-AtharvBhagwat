
# Project 3 - AI Recommendation Logic

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the dataset
data = pd.read_csv("raw_skills.csv")

# Store all skills available in the dataset
known_skills = set()

for skill_list in data["skills"]:
    for skill in skill_list.split(","):
        known_skills.add(skill.strip().lower())

# Get at least 3 valid skills from the user
while True:
    user_input = input(
        "Enter at least 3 skills, separated by commas: "
    )

    skills = [
        skill.strip().lower()
        for skill in user_input.split(",")
        if skill.strip()
    ]

    # Remove duplicate skills
    skills = list(dict.fromkeys(skills))

    if len(skills) < 3:
        print("Please enter at least 3 different skills.")
        continue

    # Check which skills are supported
    valid_skills = [
        skill for skill in skills
        if skill in known_skills
    ]

    unknown_skills = [
        skill for skill in skills
        if skill not in known_skills
    ]

    if unknown_skills:
        print("Skills not found in dataset:", unknown_skills)

    if len(valid_skills) < 3:
        print("Please enter at least 3 skills from the dataset.")
        continue

    break

# Convert career skills into numerical vectors
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["skills"])

# Convert user skills into a numerical vector
user_skills = ", ".join(valid_skills)
user_vector = vectorizer.transform([user_skills])

# Calculate similarity scores
similarity_scores = cosine_similarity(
    user_vector,
    tfidf_matrix
)[0]

# Rank careers by similarity
ranked_roles = sorted(
    range(len(data)),
    key=lambda i: similarity_scores[i],
    reverse=True
)

# Display the top 3 recommendations
print("\nTop 3 Recommended Career Paths:")

for i in ranked_roles[:3]:
    print(
        data["job_role"].iloc[i],
        "- Similarity Score:",
        round(similarity_scores[i], 3)
    )
