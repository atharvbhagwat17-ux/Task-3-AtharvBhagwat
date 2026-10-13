
# AI Career Recommendation System 🚀

An AI-based recommendation system that suggests suitable technology career paths based on a user's technical skills.

This project uses **TF-IDF vectorization** and **cosine similarity** to compare a user's selected skills with the skills associated with different career roles. It provides the top three matching career recommendations through an interactive Streamlit web interface.

## 🎯 Project Objective

The objective of this project is to understand the fundamentals of AI recommendation logic by:

- Collecting user skill preferences
- Representing skills using TF-IDF
- Calculating similarity between skill profiles
- Ranking career paths based on similarity scores
- Displaying the top three recommendations

## ✨ Features

- Interactive Streamlit web interface
- Multi-skill selection
- Skill-based career recommendations
- TF-IDF text vectorization
- Cosine similarity calculation
- Automatic ranking of career matches
- Similarity scores for each recommendation
- Display of overlapping skills
- Minimalist, professional user interface

## 📊 Dataset

The project uses a small, manually created CSV dataset containing technology career paths and their associated skills.

The dataset includes roles such as:

- Data Scientist
- Machine Learning Engineer
- Backend Developer
- Cloud Architect
- DevOps Engineer
- Data Analyst
- Cybersecurity Analyst
- Full Stack Developer
- AI Engineer
- System Administrator

The dataset is intended for learning and demonstration. Its skill descriptions are illustrative and are not verified industry hiring requirements.

## ⚙️ How It Works

1. **Dataset Loading:** Pandas loads career roles and skills from `raw_skills.csv`.
2. **Skill Selection:** The user selects at least three skills through the web interface.
3. **TF-IDF Vectorization:** The career skill descriptions are converted into numerical vectors.
4. **Cosine Similarity:** The system compares the user's skill profile with each career profile.
5. **Ranking:** Career roles are sorted by their similarity scores.
6. **Recommendation:** The top three career matches are displayed with their scores and overlapping skills.

## 🧠 Recommendation Workflow

```text
Career Skills Dataset
          ↓
     TF-IDF
  Vectorization
          ↓
 User Selects Skills
          ↓
  Cosine Similarity
     Calculation
          ↓
    Rank Careers
          ↓
  Top 3 Recommendations
```

## 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Streamlit

## 📁 Project Structure

```text
AI-Career-Recommender/
│
├── app.py
├── recommender.py
├── raw_skills.csv
├── requirements.txt
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_LINK
```

### 2. Open the project folder

```bash
cd AI-Career-Recommender
```

### 3. Install the dependencies

```bash
py -m pip install -r requirements.txt
```

### 4. Launch the Streamlit application

```bash
py -m streamlit run app.py
```

The application will open in your browser.

### 5. Get recommendations

1. Select at least three technical skills.
2. Click **Find My Career Matches**.
3. View the top three recommended career paths.
4. Compare their similarity scores and matching skills.

## 📈 Understanding Similarity Scores

A higher cosine similarity score indicates a closer match between the selected skills and a career's skill profile.

These scores measure similarity; they are not probabilities of career success or guarantees of job suitability.

## 📚 Key Concepts Learned

- AI recommendation logic
- Content-based recommendation
- Text vectorization using TF-IDF
- Cosine similarity
- Ranking and filtering
- Dataset handling using Pandas
- Building interactive web applications with Streamlit

## 🔮 Future Improvements

- Expand the dataset with more career paths and skills.
- Add detailed descriptions of each career.
- Allow users to enter custom skills.
- Improve recommendations using a larger, validated dataset.
- Deploy the application online.

## 👨‍💻 Author

**Atharv Bhagwat**

Project developed as part of an Artificial Intelligence internship.

## 📌 Project Status

Core recommendation logic and the Streamlit interface implemented.
