import pandas as pd
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Download the dataset from: https://www.kaggle.com/datasets/gauravduttakiit/resume-dataset
# Or create sample data if you don't have the dataset

def cleanResume(resumeText):
    """Clean resume text for processing"""
    resumeText = re.sub(r'\b\w{1,2}\b', '', resumeText)  # remove short words
    resumeText = re.sub(r'[^a-zA-Z]', ' ', resumeText)  # remove numbers and special characters
    return resumeText.lower()

def create_sample_data():
    """Create sample data if the original dataset is not available"""
    sample_data = {
        'Resume': [
            "Python developer with 3 years experience in Django and Flask. Worked on web applications and REST APIs. Skills include Python, JavaScript, SQL, Git.",
            "React developer with expertise in JavaScript, HTML, CSS. Built multiple web applications using React, Node.js, and MongoDB. Experience with Redux.",
            "Data scientist with experience in machine learning, Python, R, SQL. Worked on predictive models using scikit-learn, pandas, numpy.",
            "Java developer with Spring Boot experience. Built enterprise applications using Java, Spring, MySQL, AWS. Knowledge of microservices architecture.",
            "Frontend developer specializing in React and Vue.js. Experience with TypeScript, webpack, and modern JavaScript frameworks.",
            "Full stack developer with Python and React experience. Built web applications using Django, PostgreSQL, and React. DevOps experience with Docker.",
            "Mobile developer with React Native and Flutter experience. Published apps on iOS and Android. Experience with Firebase and REST APIs.",
            "Backend developer with Node.js and Express experience. Built scalable APIs using MongoDB and Redis. Experience with microservices.",
            "DevOps engineer with AWS and Docker experience. Automated CI/CD pipelines using Jenkins and GitHub Actions. Experience with Kubernetes.",
            "UI/UX designer with Figma and Adobe Creative Suite experience. Designed web and mobile applications. Experience with user research.",
        ],
        'Category': [
            'Python Developer',
            'Web Developer', 
            'Data Science',
            'Java Developer',
            'Web Developer',
            'Python Developer',
            'Mobile Developer',
            'Web Developer',
            'DevOps Engineer',
            'Designer'
        ]
    }
    return pd.DataFrame(sample_data)

def train_model():
    """Train the model and save pickle files"""
    try:
        # Try to load the original dataset
        print("Attempting to load UpdatedResumeDataSet.csv...")
        resumeDataSet = pd.read_csv('UpdatedResumeDataSet.csv', encoding='utf-8')
        print("Dataset loaded successfully!")
    except FileNotFoundError:
        print("Original dataset not found. Using sample data...")
        resumeDataSet = create_sample_data()
    
    # Clean the resume text
    print("Cleaning resume text...")
    resumeDataSet['Cleaned_Resume'] = resumeDataSet['Resume'].apply(lambda x: cleanResume(str(x)))
    
    # Create TF-IDF vectorizer
    print("Creating TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(max_features=1000)
    X = tfidf.fit_transform(resumeDataSet['Cleaned_Resume'])
    y = resumeDataSet['Category']
    
    # Split the data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    print("Training model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Test the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Save the model and vectorizer
    print("Saving model...")
    with open('resume_classifier.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)
    
    print("Saving vectorizer...")
    with open('tfidf_vectorizer.pkl', 'wb') as vectorizer_file:
        pickle.dump(tfidf, vectorizer_file)
    
    print("Model and vectorizer saved successfully!")
    print("Files created:")
    print("- resume_classifier.pkl")
    print("- tfidf_vectorizer.pkl")
    
    return model, tfidf

if __name__ == "__main__":
    model, tfidf = train_model()
