from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from supabase import create_client, Client
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime
import pandas as pd
import json
import numpy as np

# Initialize Flask app
app = Flask(__name__)

CORS(app)

SUPABASE_URL = "https://ktmddejbdwjeremvbzbl.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt0bWRkZWpiZHdqZXJlbXZiemJsIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTcyODgzNTU3MSwiZXhwIjoyMDQ0NDExNTcxfQ.jXOW4DixYvrYp-2ctv2hUhILI-E_wAtDTuepyDNtuOE"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
response = supabase.table("feedback").select("*").execute()
print(response)
# Initialize Sentiment Analyzers
sia = SentimentIntensityAnalyzer()

# Naive Bayes setup for aspect-based sentiment
vectorizer = TfidfVectorizer(max_features=5000)
classifier = MultinomialNB()

# Normalize Sentiment Labels
def normalize_sentiments(file_path):
    data = pd.read_csv(file_path)
    data["Sentiment"] = data["Sentiment"].str.strip().str.lower()
    mapping = {"positive": "positive", "negative": "negative", "neutral": "neutral"}
    data["Sentiment"] = data["Sentiment"].map(mapping)
    data.to_csv(file_path, index=False)
    print("Sentiments normalized and dataset saved.")

# Load Dataset
def load_dataset(file_path):
    data = pd.read_csv(file_path)
    summaries = data["Summary"].fillna("")
    sentiments = data["Sentiment"].fillna("")
    return summaries, sentiments

# Inspect Dataset Sentiment Distribution
def inspect_sentiment_distribution(file_path):
    data = pd.read_csv(file_path)
    sentiment_counts = data["Sentiment"].value_counts()
    print("Sentiment Distribution:")
    print(sentiment_counts)

# Train the Naive Bayes Model
def train_naive_bayes():
    summaries, sentiments = load_dataset("FlipkartProductReviewsWithSentimentDataset.csv")
    train_features = vectorizer.fit_transform(summaries)
    classifier.fit(train_features, sentiments)
    print("Model trained successfully")

# Evaluate the Naive Bayes Model
def evaluate_naive_bayes():
    summaries, sentiments = load_dataset("FlipkartProductReviewsWithSentimentDataset.csv")
    valid_data = pd.DataFrame({"Summary": summaries, "Sentiment": sentiments})
    valid_data = valid_data[valid_data["Sentiment"].isin(["positive", "negative", "neutral"])]

    if valid_data.empty:
        print("No valid samples found for evaluation. Check your dataset.")
        return

    summaries = valid_data["Summary"]
    sentiments = valid_data["Sentiment"]

    test_features = vectorizer.transform(summaries)
    predictions = classifier.predict(test_features)

    labels = ["positive", "neutral", "negative"]
    cm = confusion_matrix(sentiments, predictions, labels=labels)
    acc = accuracy_score(sentiments, predictions)
    precision = precision_score(sentiments, predictions, average='weighted', zero_division=0)
    recall = recall_score(sentiments, predictions, average='weighted', zero_division=0)
    f1 = f1_score(sentiments, predictions, average='weighted', zero_division=0)

    metrics = {
        "confusion_matrix": cm.tolist(),
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }

    with open("metrics.json", "w") as f:
        json.dump(metrics, f)

    print("Metrics saved to 'metrics.json'")
    print(f"Confusion Matrix:\n{cm}")
    print(classification_report(sentiments, predictions, target_names=labels))

# Analyze Sentiment for Feedback
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    sentiment = analyzer.polarity_scores(text)

    # Determine the label based on the compound score with a threshold of 0.1
    if sentiment['compound'] >= 0.1:
        label = "positive"
    elif sentiment['compound'] <= -0.1:
        label = "negative"
    else:
        label = "neutral"

    return {
        "neg": sentiment['neg'],
        "neu": sentiment['neu'],
        "pos": sentiment['pos'],
        "compound": sentiment['compound'],
        "label": label
    }


# Calculate overall sentiment based on the average compound score
# Calculate overall sentiment based on the average compound score
def calculate_overall_sentiment(feedback_data):
    # Get all the compound scores from the individual sentiments
    compound_scores = [
        feedback_data["event_sentiment"]["compound"],
        feedback_data["venue_sentiment"]["compound"],
        feedback_data["catering_sentiment"]["compound"],
        feedback_data["decoration_sentiment"]["compound"],
        feedback_data["food_catering_sentiment"]["compound"],
        feedback_data["accommodation_sentiment"]["compound"],
        feedback_data["transportation_sentiment"]["compound"],
        feedback_data["photography_sentiment"]["compound"],
        feedback_data["videography_sentiment"]["compound"],
        feedback_data["host_sentiment"]["compound"],
        feedback_data["entertainment_sentiment"]["compound"],
        feedback_data["sound_sentiment"]["compound"],
        feedback_data["lighting_sentiment"]["compound"],
        feedback_data["venue_management_sentiment"]["compound"],
        feedback_data["marketing_sentiment"]["compound"],
        feedback_data["other_sentiment"]["compound"]
    ]
    
    # Filter out None values in case some sentiments are not available
    compound_scores = [score for score in compound_scores if score is not None and score != 0]
    
    # Calculate the average of the compound scores
    if compound_scores:  # Ensure there are scores to average
        average_compound = np.mean(compound_scores)
    else:
        average_compound = 0  # Default to neutral if no scores available

    # Determine overall sentiment based on average compound score
    if average_compound >= 0.1:
        return "positive"
    elif average_compound <= -0.1:
        return "negative"
    else:
        return "neutral"
@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    print(request.json)  # Log the entire request body
    # Extract relevant feedback categories from the request
    event_feedback = request.json.get('event_feedback', '')
    venue_feedback = request.json.get('venue_feedback', '')
    catering_feedback = request.json.get('catering_feedback', '')
    decoration_feedback = request.json.get('decoration_feedback', '')
    food_catering_feedback = request.json.get('food_catering_feedback', '')
    accommodation_feedback = request.json.get('accommodation_feedback', '')
    transportation_feedback = request.json.get('transportation_feedback', '')
    photography_feedback = request.json.get('photography_feedback', '')
    videography_feedback = request.json.get('videography_feedback', '')
    host_feedback = request.json.get('host_feedback', '')
    entertainment_feedback = request.json.get('entertainment_feedback', '')
    sound_feedback = request.json.get('sound_feedback', '')
    lighting_feedback = request.json.get('lighting_feedback', '')
    venue_management_feedback = request.json.get('venue_management_feedback', '')
    marketing_feedback = request.json.get('marketing_feedback', '')
    other_feedback = request.json.get('other_feedback', '')

    event_id = request.json.get('event_id', None)
    customer_name = request.json.get('customer_name', None)
    customer_id = request.json.get('customer_id', None)


   # Manually encode feedback data
    feedback_data = {
        "event_feedback": event_feedback,
        "event_sentiment": analyze_sentiment(event_feedback),
        "venue_feedback": venue_feedback,
        "venue_sentiment": analyze_sentiment(venue_feedback),
        "catering_feedback": catering_feedback,
        "catering_sentiment": analyze_sentiment(catering_feedback),
        "decoration_feedback": decoration_feedback,
        "decoration_sentiment": analyze_sentiment(decoration_feedback),
        "food_catering_feedback": food_catering_feedback,
        "food_catering_sentiment": analyze_sentiment(food_catering_feedback),
        "accommodation_feedback": accommodation_feedback,
        "accommodation_sentiment": analyze_sentiment(accommodation_feedback),
        "transportation_feedback": transportation_feedback,
        "transportation_sentiment": analyze_sentiment(transportation_feedback),
        "photography_feedback": photography_feedback,
        "photography_sentiment": analyze_sentiment(photography_feedback),
        "videography_feedback": videography_feedback,
        "videography_sentiment": analyze_sentiment(videography_feedback),
        "host_feedback": host_feedback,
        "host_sentiment": analyze_sentiment(host_feedback),
        "entertainment_feedback": entertainment_feedback,
        "entertainment_sentiment": analyze_sentiment(entertainment_feedback),
        "sound_feedback": sound_feedback,
        "sound_sentiment": analyze_sentiment(sound_feedback),
        "lighting_feedback": lighting_feedback,
        "lighting_sentiment": analyze_sentiment(lighting_feedback),
        "venue_management_feedback": venue_management_feedback,
        "venue_management_sentiment": analyze_sentiment(venue_management_feedback),
        "marketing_feedback": marketing_feedback,
        "marketing_sentiment": analyze_sentiment(marketing_feedback),
        "other_feedback": other_feedback,
        "other_sentiment": analyze_sentiment(other_feedback),
        "event_id": event_id,
        "customer_id": customer_id,
        "customer_name": customer_name,
        "timestamp": datetime.utcnow().isoformat()
    }
    feedback_data["overall_sentiment"] = calculate_overall_sentiment(feedback_data)
    print(feedback_data)
    # Calculate overall sentiment based on compound scores of the individual sentiments
    try:
        # Insert structured feedback data into Supabase
        response = supabase.table("feedback").insert(feedback_data).execute()
        return jsonify({"message": "Feedback submitted successfully", "data": response.data}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_feedback', methods=['GET'])
def get_feedback():
    try:
        response = supabase.table("feedback").select("*").execute()
        feedback_data = response.data
        return jsonify(feedback_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_feedback_event', methods=['GET'])
def get_feedback_by_event():
    event_id = request.args.get('event_id', None)
    try:
        if event_id:
            response = supabase.table("feedback").select("*").eq("event_id", event_id).execute()
        else:
            response = supabase.table("feedback").select("*").execute()
        feedback_data = response.data
        return jsonify(feedback_data), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    count_feedback

@app.route('/count_feedback_by_aspect', methods=['GET'])
def count_feedback_by_aspect():
    event_id = request.args.get('event_id', None)  # Event ID from the request
    aspect = request.args.get('aspect', None)  # Aspect from the request (e.g., "event_feedback", "venue_feedback", etc.)
    
    if not event_id:
        return jsonify({"error": "Event ID is required"}), 400
    
    if not aspect:
        return jsonify({"error": "Aspect is required"}), 400

    try:
        # Fetch feedback data for the given event ID
        response = supabase.table("feedback").select("*").eq("event_id", event_id).execute()
        feedback_data = response.data

        if not feedback_data:
            return jsonify({"error": "No feedback found for the given event ID"}), 404

        # Initialize counters for each sentiment type
        sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}

        # Iterate through the feedback data and count sentiment for the specified aspect
        for feedback in feedback_data:
            # Get sentiment value for the specified aspect, ensuring it's not None or an empty value
            sentiment = feedback.get(f"{aspect}_sentiment", {}).get("label", "").strip()

            if sentiment == "positive":
                sentiment_count["positive"] += 1
            elif sentiment == "negative":
                sentiment_count["negative"] += 1
            elif sentiment == "neutral":
                sentiment_count["neutral"] += 1
            # Skip empty, null, or undefined sentiments (nothing is counted for them)

        return jsonify({"sentiment_count": sentiment_count}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/count_feedback_by_aspect_all_events', methods=['GET'])
def count_feedback_by_aspect_all_events():
    aspect = request.args.get('aspect', None)  # Aspect from the request (e.g., "service", "food", etc.)
    
    if not aspect:
        return jsonify({"error": "Aspect is required"}), 400

    try:
        # Fetch all feedback data (no filtering by event_id)
        response = supabase.table("feedback").select("*").execute()
        feedback_data = response.data

        if not feedback_data:
            return jsonify({"error": "No feedback found"}), 404

        # Initialize counters for each sentiment type
        sentiment_count = {"positive": 0, "negative": 0, "neutral": 0}

        # Iterate through the feedback data and count sentiment for the specified aspect
        for feedback in feedback_data:
            # Check if the feedback entry contains the specified aspect's sentiment
            aspect_sentiment = feedback.get(f"{aspect}_sentiment", {}).get("label", None)
            
            if aspect_sentiment:
                if aspect_sentiment == "positive":
                    sentiment_count["positive"] += 1
                elif aspect_sentiment == "negative":
                    sentiment_count["negative"] += 1
                elif aspect_sentiment == "neutral":
                    sentiment_count["neutral"] += 1

        return jsonify({"sentiment_count": sentiment_count}), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/count_feedback', methods=['GET'])
def count_feedback():
    event_id = request.args.get('event_id', None)  # Event ID from the request
    count = request.args.get('count', None)  # Count flag from the request to trigger counting all feedback
    if not event_id and count != 'true':
        return jsonify({"error": "Event ID is required or count flag must be true"}), 400

    try:
        # Fetch feedback data for the specified event or all events if 'count' is true
        if event_id:
            response = supabase.table("feedback").select("*").eq("event_id", event_id).execute()
        elif count == 'true':  # If count is true, fetch feedback for all events
            response = supabase.table("feedback").select("*").execute()

        feedback_data = response.data

        if not feedback_data:
            return jsonify({"message": "No feedback found"}), 404

        # Initialize sentiment counters
        total_feedback = len(feedback_data)
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        # Count sentiment labels
        for feedback in feedback_data:
            overall_sentiment = feedback.get("overall_sentiment")
            if overall_sentiment == "positive":
                positive_count += 1
            elif overall_sentiment == "negative":
                negative_count += 1
            elif overall_sentiment == "neutral":
                neutral_count += 1

        # Return the counts
        result = {
            "total_feedback": total_feedback,
            "positive_count": positive_count,
            "negative_count": negative_count,
            "neutral_count": neutral_count,
        }

        # If event_id was provided, include it in the response
        if event_id:
            result["event_id"] = event_id

        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_events_with_sentiment_counts', methods=['GET'])
def get_events_with_sentiment_counts():
    try:
        # Fetch all feedback data
        response = supabase.table("feedback").select("*").execute()
        feedback_data = response.data

        if not feedback_data:
            return jsonify({"error": "No feedback found"}), 404

        # Initialize a dictionary to store the sentiment counts for each event
        event_sentiments = {}

        # Iterate through the feedback data and count the sentiment for each event
        for feedback in feedback_data:
            event_id = feedback.get("event_id")
            overall_sentiment = feedback.get("overall_sentiment")

            if event_id not in event_sentiments:
                event_sentiments[event_id] = {"negative": 0, "neutral": 0, "positive": 0}

            if overall_sentiment == "positive":
                event_sentiments[event_id]["positive"] += 1
            elif overall_sentiment == "negative":
                event_sentiments[event_id]["negative"] += 1
            elif overall_sentiment == "neutral":
                event_sentiments[event_id]["neutral"] += 1

        # Convert the dictionary to an array of events with sentiment counts
        events_with_sentiment_counts = []
        for event_id, sentiment_counts in event_sentiments.items():
            events_with_sentiment_counts.append({
                "event_id": event_id,
                "negative": sentiment_counts["negative"],
                "neutral": sentiment_counts["neutral"],
                "positive": sentiment_counts["positive"]
            })

        return jsonify({"events": events_with_sentiment_counts}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
@app.route('/download-metrics', methods=['GET'])
def download_metrics():
    try:
        return send_file("metrics.json", as_attachment=True)
    except Exception as e:
        return {"error": str(e)}, 500

@app.route('/get_aspects_with_sentiments', methods=['GET']) 
def get_aspects_with_sentiments():
    event_id = request.args.get('event_id', None)  # Get event_id from the request

    if not event_id:
        return jsonify({"error": "Event ID is required"}), 400

    try:
        # Fetch feedback data for the given event ID
        response = supabase.table("feedback").select("*").eq("event_id", event_id).execute()
        feedback_data = response.data

        if not feedback_data:
            return jsonify({"error": "No feedback found for the given event ID"}), 404

        # Initialize a dictionary to store sentiment counts and raw feedback data for all aspects
        aspects_sentiment_count = {}
        aspects_raw_feedback = {}
        aspects_customer_info = {}

        # Iterate through the feedback data
        for feedback in feedback_data:
            for key, value in feedback.items():
                # Check if the key represents an aspect's sentiment and if the field contains valid feedback
                if "_sentiment" in key and isinstance(value, dict):
                    aspect = key.replace("_sentiment", "")  # Extract aspect name (e.g., "service")
                    
                    # Skip aspects with empty/null feedback text
                    feedback_text_key = f"{aspect}_feedback"  # Assume feedback text key follows the pattern <aspect>_feedback
                    feedback_text = feedback.get(feedback_text_key)
                    if not feedback_text:  # If feedback text is missing or empty, skip this aspect
                        continue
                    
                    sentiment = value.get("label", "").strip()

                    if sentiment:  # Only process if sentiment is not empty
                        if aspect not in aspects_sentiment_count:
                            aspects_sentiment_count[aspect] = {"positive": 0, "negative": 0, "neutral": 0}
                            aspects_raw_feedback[aspect] = []
                            aspects_customer_info[aspect] = []

                        # Update sentiment count for the aspect
                        if sentiment == "positive":
                            aspects_sentiment_count[aspect]["positive"] += 1
                        elif sentiment == "negative":
                            aspects_sentiment_count[aspect]["negative"] += 1
                        elif sentiment == "neutral":
                            aspects_sentiment_count[aspect]["neutral"] += 1

                        # Add raw feedback text for this aspect
                        aspects_raw_feedback[aspect].append({
                            "feedback_text": feedback_text,
                            "customer_name": feedback.get("customer_name"),
                            "customer_id": feedback.get("customer_id")
                        })

        # Return both sentiment counts and raw feedback data
        return jsonify({
            "aspects_sentiment_count": aspects_sentiment_count,
            "aspects_raw_feedback": aspects_raw_feedback
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
# Main Script Execution
if __name__ == "__main__":
    normalize_sentiments("FlipkartProductReviewsWithSentimentDataset.csv")
    inspect_sentiment_distribution("FlipkartProductReviewsWithSentimentDataset.csv")
    train_naive_bayes()
    evaluate_naive_bayes()
    app.run(host="0.0.0.0", port=5000)
    # app.run(host="192.168.0.168", port=8000)  
    app.run(debug=True)  # Enabling debug mode

