import argparse
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import random

# Load the IMDb dataset and balance with 10% of each class (negative/positive)
def load_balanced_dataset_for_finetuning():
    print("Loading IMDb dataset...")
    dataset = load_dataset("imdb", split="train")  # Load the full training dataset
    print(f"Total samples: {len(dataset)}")
    
    # Separate positive and negative reviews
    positive_samples = [example for example in dataset if example['label'] == 1]
    negative_samples = [example for example in dataset if example['label'] == 0]
    
    # Take 10% of positive and 10% of negative reviews
    num_positive_samples = int(0.1 * len(positive_samples))
    num_negative_samples = int(0.1 * len(negative_samples))
    
    sampled_positive = random.sample(positive_samples, num_positive_samples)
    sampled_negative = random.sample(negative_samples, num_negative_samples)
    
    # Combine and shuffle the balanced dataset
    balanced_dataset = sampled_positive + sampled_negative
    random.shuffle(balanced_dataset)
    
    print(f"Balanced samples: {len(balanced_dataset)}")
    print(f"Label distribution: {len([example for example in balanced_dataset if example['label'] == 0])} negative, {len([example for example in balanced_dataset if example['label'] == 1])} positive")
    
    return balanced_dataset

# Tokenization function for sentiment analysis
def tokenize_function(examples, tokenizer):
    inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    inputs["labels"] = torch.tensor(examples["label"]).long()  # 0 for negative, 1 for positive
    return inputs

# Fine-tune DistilBERT for sentiment classification
def fine_tune_model():
    print("Loading DistilBERT model and tokenizer...")
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Load and tokenize dataset
    dataset = load_balanced_dataset_for_finetuning()
    tokenized_datasets = [tokenize_function(example, tokenizer) for example in dataset]

    # DataLoader to convert the dataset into PyTorch tensors
    def collate_fn(batch):
        input_ids = torch.tensor([item["input_ids"] for item in batch], dtype=torch.long)
        attention_mask = torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long)
        labels = torch.tensor([item["labels"] for item in batch], dtype=torch.long)  # 0 for negative, 1 for positive
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    train_loader = DataLoader(tokenized_datasets, batch_size=16, collate_fn=collate_fn)

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Move model to device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print("Starting training...")
    model.train()

    # Training loop for 1 epoch
    for epoch in range(1):  # 1 epoch
        for batch_idx, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Batch {batch_idx + 1} - Loss: {loss.item()}")

    print("Saving fine-tuned model...")
    model.save_pretrained("./fine_tuned_distilbert_sentiment")
    tokenizer.save_pretrained("./fine_tuned_distilbert_sentiment")
    print("Model saved successfully.")

# Load the fine-tuned model for prediction
def load_fine_tuned_model():
    print("Loading fine-tuned model...")
    model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_distilbert_sentiment")
    tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_distilbert_sentiment")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, tokenizer, device

# Generate dynamic response based on predicted sentiment
def generate_response(sentiment, review):
    if sentiment == "positive":
        return f"Thank you for your positive review! We're glad you enjoyed the movie: '{review}'"
    elif sentiment == "negative":
        return f"We're sorry to hear you didn't enjoy the movie. We appreciate your feedback: '{review}'"
    else:
        return f"Thank you for your review: '{review}'"

# Predict sentiment using the fine-tuned model
def predict_sentiment(model, tokenizer, device, review):
    inputs = tokenizer(review, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)

    # Get model outputs (logits) for the input review
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # Use softmax to get probabilities
    probabilities = torch.softmax(logits, dim=1)
    sentiment_score = probabilities[0].cpu().numpy()

    print(f"Sentiment probabilities: {sentiment_score}")

    # Dynamically decide sentiment based on score
    sentiment = "positive" if sentiment_score[1] > 0.5 else "negative"
    print(f"Predicted sentiment: {sentiment}")

    # Generate dynamic response based on sentiment
    response = generate_response(sentiment, review)
    return response

# Main function to run the app
def main():
    parser = argparse.ArgumentParser(description="Fine-tune or load a DistilBERT model for sentiment analysis.")
    parser.add_argument("retrain", type=str, help="Set to 'true' to retrain the model, 'false' to use the saved model.")
    args = parser.parse_args()

    if args.retrain.lower() == "true":
        fine_tune_model()
    else:
        model, tokenizer, device = load_fine_tuned_model()

        # Input a review and predict sentiment
        review = input("Please enter your review: ")
        response = predict_sentiment(model, tokenizer, device, review)
        print(f"\nDynamic Response: {response}")

if __name__ == "__main__":
    main()
