# 🎯 Aspect-based Sentiment Analysis - Flask Application

This project implements an Aspect-based Sentiment Analysis system using Flask, allowing users to analyze sentiment in event feedback and reviews.

## 🚀 Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Virtual environment (recommended)

## 📦 Required Packages

The project requires the following Python packages:
- Flask: Web framework
- Flask-Cors: Cross-Origin Resource Sharing support
- gunicorn: WSGI HTTP Server
- supabase: Database integration
- nltk: Natural Language Processing toolkit
- scikit-learn: Machine Learning library
- pandas: Data manipulation and analysis
- numpy: Numerical computing

## 🛠️ Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/BunjanMark/Aspect-based-Sentiment-Analysis---FLASK.git
cd Aspect-based-Sentiment-Analysis---FLASK
```

2. Create and activate a virtual environment (recommended):
```bash
# Windows
python -m venv venv
venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Download required NLTK data:
```bash
python -c "import nltk; nltk.download('vader_lexicon')"
```

5. Configure environment variables:
```bash
# Copy the example environment file
cp .env.example .env
```
Then update the `.env` file with your configuration:
```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
```

6. Prepare the dataset:
- Ensure `FlipkartProductReviewsWithSentimentDataset.csv` is present in the project root directory

## 📁 Project Structure

- `app.py`: Main Flask application file
- `requirements.txt`: List of Python dependencies
- `*.pkl`: Pre-trained models and vectorizers
- Various CSV files: Training and testing datasets

## 🚀 Running the Application

1. Ensure your virtual environment is activated
2. Run the Flask application:
```bash
python app.py
```
3. The application will:
   - 📊 Normalize the sentiment dataset
   - 🤖 Train and evaluate the Naive Bayes model
   - 🌐 Start the Flask server on `http://localhost:5000`

⚠️ Note: By default, the server runs in development mode. For production deployment, modify the host and port settings in `app.py` accordingly.

## 🔌 API Endpoints

The application provides several endpoints:
- `/submit_feedback` (POST): Submit new feedback with multiple aspects
- `/get_feedback` (GET): Retrieve all feedback
- `/get_feedback_by_event/<event_id>` (GET): Get feedback for a specific event
- `/count_feedback` (GET): Get sentiment statistics
- `/metrics` (GET): Download model metrics

## 🤖 Models and Data

The project includes several pre-trained models:
- `sentiment_model.pkl`: Main sentiment analysis model
- `naive_bayes_model.pkl`: Naive Bayes classifier
- `tfidf_vectorizer.pkl`: TF-IDF vectorizer for text processing
- `vectorizer.pkl`: Additional text vectorizer

## 📊 Dataset Information

The project uses various datasets for training and testing:
- Event feedback datasets
- Sentiment analysis datasets
- Product review datasets

## 📝 Note

Make sure all the required model files (`.pkl`) and datasets (`.csv`) are present in the project directory before running the application.

## 📝 License

This project is open source and available under the MIT License.
