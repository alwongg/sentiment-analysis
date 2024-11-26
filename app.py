from flask import Flask, request, render_template, jsonify
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch

# Initialize Flask app
app = Flask(__name__)

# Load fine-tuned model
def load_fine_tuned_model():
    model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_distilbert_sentiment")
    tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_distilbert_sentiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

model, tokenizer, device = load_fine_tuned_model()

# Generate dynamic response based on predicted sentiment
def generate_response(sentiment, review):
    if sentiment == "positive":
        return f"Thank you for your positive review! We're glad you enjoyed the movie: '{review}'"
    elif sentiment == "negative":
        return f"We're sorry to hear you didn't enjoy the movie. We appreciate your feedback: '{review}'"
    else:
        return f"Thank you for your review: '{review}'"

# Predict sentiment
def predict_sentiment(model, tokenizer, device, review):
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    # Get model outputs (logits) for the input review
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Use softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)
    sentiment_score = probabilities[0].cpu().numpy()

    # Dynamically decide sentiment based on score
    sentiment = "positive" if sentiment_score[1] > 0.5 else "negative"
    return generate_response(sentiment, review)

# Define routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    review = request.json.get("review")
    if not review:
        return jsonify({"error": "No review provided"}), 400

    response = predict_sentiment(model, tokenizer, device, review)
    return jsonify({"response": response})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
