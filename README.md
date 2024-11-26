# Sentiment Analysis with DistilBERT

## Project Overview

This project demonstrates the use of **DistilBERT** for **sentiment analysis** on movie reviews from the IMDb dataset. It involves fine-tuning a pre-trained DistilBERT model to classify movie reviews as **positive** or **negative**, and then generating dynamic responses based on the predicted sentiment.

The project explores manual fine-tuning of a Large Language Model (LLM), giving control over model training, dataset balancing, and performance optimization.

## Features

- **Sentiment Analysis**: Classify movie reviews as either positive or negative.
- **Dynamic Response Generation**: Automatically generate responses based on the sentiment of the review.
- **Manual Fine-Tuning**: Train a pre-trained DistilBERT model on a balanced IMDb dataset using PyTorch.
- **Balanced Dataset**: The training dataset is balanced with 10% of positive and negative reviews to improve model generalization.

## Requirements

- Python 3.10+
- PyTorch
- Transformers (by Hugging Face)
- Datasets (by Hugging Face)
- Other dependencies: see `requirements.txt`

## Installation

1. Install the required Python libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the IMDb dataset (this is handled automatically by the `datasets` library during runtime).

## Usage

### 1. Fine-Tune the Model

To fine-tune the model on the IMDb dataset, run the script with the `retrain` argument set to `true`. This will load and balance the dataset, then fine-tune the DistilBERT model for sentiment classification.

```bash
python3 distilbert_sentiment_analysis.py true
```

This command will:
- Load 10% of positive and negative reviews from the IMDb dataset.
- Fine-tune the DistilBERT model for binary classification.
- Save the fine-tuned model in the `./fine_tuned_distilbert_sentiment/` directory.

### 2. Use the Pre-Trained Model

If you have already fine-tuned the model, you can use it to predict the sentiment of movie reviews and generate responses:

```bash
python3 distilbert_sentiment_analysis.py false
```

This command will:
- Load the fine-tuned model and tokenizer from `./fine_tuned_distilbert_sentiment/`.
- Prompt you to enter a movie review.
- Predict the sentiment and generate a dynamic response.

### Example:

```bash
Please enter your review: The movie was fantastic!
Sentiment probabilities: [0.00123, 0.99877]
Predicted sentiment: positive

Dynamic Response: Thank you for your positive review! We're glad you enjoyed the movie: 'The movie was fantastic!'
```

How to get started with a Web Interface:

1. Install Flash:
   ```bash
   pip install flask
   ```
2. Create app.py that contains the Flask app that integrates with your existing python program code
3. Create the HTML Template that will render your web page
4. Run the Flask app with:
   ```bash
   python app.py
   ```
5. Visit http://127.0.0.1:5000 in your web browser
