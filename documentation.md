# Sentiment Analysis with DistilBERT: Documentation

## Project Overview

This project focuses on using **DistilBERT**, a distilled version of BERT (Bidirectional Encoder Representations from Transformers), to perform **sentiment analysis** on movie reviews from the IMDb dataset. The goal of this project is to:

1. **Fine-tune a pre-trained DistilBERT model** on a balanced subset of the IMDb dataset for binary classification (positive/negative sentiment).
2. **Predict sentiment dynamically** and generate responses based on whether the input review is positive or negative.
3. Learn the principles of **model fine-tuning**, **prompt engineering**, and **sentiment analysis** using **Large Language Models (LLMs)**.

### DistilBERT as an LLM

**DistilBERT** is considered a **Large Language Model (LLM)**, derived from BERT. It is trained using knowledge distillation to reduce its size while retaining the performance of the original BERT model. This makes DistilBERT an efficient yet powerful tool for natural language processing tasks like sentiment analysis, text classification, and more.

## Why DistilBERT?

### Initial Challenges with GPT-2

Initially, GPT-2 was considered for this task, but GPT-2 is designed for **generative tasks** rather than **binary classification**. Specifically, GPT-2:

1. **Struggles with binary classification**: GPT-2 generates text rather than classifying inputs into discrete categories like "positive" or "negative."
2. **Complexity in training**: GPT-2 fine-tuning required additional engineering to perform binary sentiment analysis, making it less suitable for classification tasks without significant modifications.

As a result, I went from GPT-2 to **DistilBERT**, which is specifically suited for **classification** tasks. DistilBERT is a **transformer-based model** pre-trained for **sequence classification**, making it a more appropriate choice for binary sentiment analysis. I did not understand this and kept training GPT-2 at 100% with epoch 3 and it took 12 hours to train and in the end, always gave a negative sentiment.

## Manual Training vs. Hugging Face's Trainer API

Instead of using Hugging Face's **Trainer API** (which abstracts a lot of training steps), I chose **manual training** to better understand the entire fine-tuning process. This allowed for greater control over:

- **Data balancing**: By manually sampling and balancing the dataset, I ensured that positive and negative examples were equally represented.
- **Custom training loop**: Manual control over epochs, optimizers, and loss functions provided a deeper understanding of how models like DistilBERT are fine-tuned.

The Trainer API simplifies training but abstracts away key elements that are useful for learning purposes, such as how gradients are calculated and how weights are updated during backpropagation. But most importantly, it didn't work for me at all because of local module errors, which forced me to find other approaches.

## Project Structure

- **distilbert_sentiment_analysis.py**: The main script that handles everything from loading the dataset, fine-tuning the DistilBERT model, and generating dynamic responses based on sentiment.
- **requirements.txt**: Contains the dependencies required for this project (e.g., `torch`, `transformers`, `datasets`).

## Workflow Overview

### 1. Loading and Balancing the Dataset

The IMDb dataset is loaded using the `datasets` library. The dataset is balanced by selecting **10% of positive and negative reviews** to create a dataset that contains equal numbers of positive and negative samples. This ensures that the model does not overfit to one sentiment class.

Key Steps:
- **Separate positive and negative samples**.
- **Sample 10%** of both positive and negative reviews.
- **Shuffle** the combined dataset for balanced learning.

### 2. Tokenizing and Preparing Data for Fine-tuning

The tokenizer (`DistilBertTokenizer`) is used to process the input text into tokens that the model can understand. The inputs are padded and truncated to a maximum length of 128 tokens.

Key Steps:
- **Tokenize** the reviews and convert the labels (0 for negative, 1 for positive) into tensors.
- The data is prepared and loaded into a **PyTorch DataLoader** for batch processing during fine-tuning.

### 3. Fine-tuning the DistilBERT Model

A pre-trained **DistilBERT** model (`DistilBertForSequenceClassification`) is used and fine-tuned using the IMDb dataset for 1 epoch. During fine-tuning:
- The model learns to classify reviews into **positive** or **negative** sentiment.
- The **CrossEntropyLoss** function is used since this is a classification task.
- The AdamW optimizer is employed to update model parameters.

### 4. Generating Predictions and Dynamic Responses

After fine-tuning, the model is loaded and used to predict the sentiment of a new review. The model’s output is passed through a **softmax function** to generate probabilities for each sentiment class (positive or negative). Based on the prediction, a dynamic response is generated.

Example dynamic responses:
- Positive: "Thank you for your positive review! We're glad you enjoyed the movie."
- Negative: "We're sorry to hear you didn't enjoy the movie. We appreciate your feedback."

## Challenges Faced

### 1. Dataset Balance and Representation
Initially, while training on 1% of the dataset, the results were inconsistent. After analyzing the label distribution, it became clear that the training data was imbalanced (many more negative reviews than positive ones). To fix this:
- I sampled **10%** from both positive and negative reviews.
- Ensured that the model was trained on a **balanced dataset** for better generalization.

### 2. Moving from GPT-2 to DistilBERT
As discussed earlier, GPT-2 was ill-suited for binary classification. DistilBERT, a model designed for classification tasks, was a much better fit for this sentiment analysis project. The transition allowed me to leverage DistilBERT’s pre-trained weights for **sequence classification**, saving time and improving performance.

### 3. Model Fine-tuning
Manually training the model allowed me to control the dataset’s batch size, number of epochs, and optimizer behavior. This helped address issues like overfitting or underfitting the model by adjusting the learning rate and other hyperparameters.

### 4. Module Errors and Dependency Issues
During development, I faced various issues with dependencies, particularly:
- **Numpy and Torch version conflicts**: Ensuring compatibility between these libraries required installing the correct versions, which I managed through pip and virtual environments.
- **Dataset tokenization and padding**: Understanding how the tokenizer works and ensuring that the inputs were correctly padded and truncated was critical to avoid errors during training.

## Sentiment Prediction and Response Generation

After fine-tuning, the model is able to classify a movie review into either **positive** or **negative** sentiment. The predicted sentiment score is then used to dynamically generate a response to the user.

### Example Workflow:
1. **Input Review**: "The movie was awesome."
2. **Predicted Sentiment**: Positive
3. **Generated Response**: "Thank you for your positive review! We're glad you enjoyed the movie."

### Code Walkthrough

1. **Fine-tuning Function (`fine_tune_model`)**:
   - Loads and balances the IMDb dataset.
   - Tokenizes the reviews using `DistilBertTokenizer`.
   - Trains the model for **1 epoch** using **CrossEntropyLoss** and the **AdamW optimizer**.
   - Saves the fine-tuned model for later use.

2. **Sentiment Prediction (`predict_sentiment`)**:
   - Loads the fine-tuned model and tokenizer.
   - Processes new reviews using the tokenizer.
   - Classifies the review by passing it through the model and obtaining probabilities using the **softmax** function.

3. **Dynamic Response Generation (`generate_response`)**:
   - Based on the predicted sentiment (positive or negative), the system generates a custom response.
   
   For example:
   - If the sentiment is **positive**, the response might be: "Thank you for your positive review!"
   - If the sentiment is **negative**, the response might be: "We're sorry to hear you didn't enjoy the movie."

## Conclusion

This project demonstrates the use of **DistilBERT** for fine-tuning on a binary sentiment classification task. The project helped explore:
- How to manually fine-tune a pre-trained LLM like DistilBERT.
- The challenges of balancing datasets and ensuring fair representation of both positive and negative samples.
- How to dynamically generate responses based on the predicted sentiment, making the model suitable for customer service or chatbot applications.

The code also exemplifies how to address common challenges like model training, dataset balancing, and controlling model behavior without using high-level abstractions like the Hugging Face **Trainer API**. This hands-on approach provides a deep understanding of how transformer-based models can be used in real-world applications.
