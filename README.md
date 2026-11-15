# Sentiment Classification of Customer Feedback: SVM vs. BERT

Companies receive far more written feedback than anyone can read by hand. This project builds an end-to-end NLP pipeline that sorts text into **positive, negative or neutral**, and compares a classical model (TF-IDF + Linear SVM) with a fine-tuned transformer (BERT).

## Dataset

- [Emotion Analysis Based on Text](https://www.kaggle.com/datasets/simaanjali/emotion-analysis-based-on-text/data) (Kaggle): short English texts labelled with emotions
- Emotions mapped to sentiment: joy, love, surprise and happy → positive; anger, hate, sadness, fear and disgust → negative; neutral → neutral
- **839,555** rows, reduced to **358,596** after removing duplicates
- Highly imbalanced: **88.4% neutral**, 5.9% positive, 5.7% negative
- The CSV is too large for GitHub (over 25 MB) and can be downloaded from Kaggle

## Approach

1. **Exploratory analysis:** class balance, text length per class, common words and word clouds
2. **Preprocessing:** lowercasing, removing URLs, HTML, punctuation and numbers, stop-word removal, lemmatisation (NLTK)
3. **Classical models:** TF-IDF features (5,000), then Logistic Regression, Random Forest and Linear SVM with balanced class weights. SVM tuned with grid search over `C`
4. **Transformer:** `bert-base-uncased` fine-tuned for 3 epochs with weighted cross-entropy loss, on a 1,000-text sample because of limited compute
5. **Evaluation:** accuracy, weighted F1, log loss and confusion matrices

## Results

| Model | Training data | Accuracy | Weighted F1 |
|---|---|---|---|
| Logistic Regression | 286,876 texts | 0.996 | 0.996 |
| Random Forest | 286,876 texts | 0.998 | 0.998 |
| **Linear SVM** | 286,876 texts | **0.998** | **0.998** |
| BERT (fine-tuned) | 800 texts | 0.88 | 0.89 |

Linear SVM was chosen as the best classical model because it also had the lowest log loss (0.019), meaning its confidence scores were the most reliable.

## What the results really say

- **SVM beat BERT here, but the comparison isn't equal.** SVM learned from about 287,000 texts and BERT from 800. With the same amount of data, BERT would very likely close the gap.
- **BERT's minority-class scores are unreliable.** Its validation set had only 12 negative and 11 positive texts, so a single prediction changes the F1 score noticeably.
- **Near-perfect accuracy is a warning sign, not just a success.** Scores of 99.8% suggest the labels follow surface keywords that TF-IDF picks up easily. Some labels also look questionable, such as "im so full of life i feel appalled" marked as neutral. Real customer feedback would very likely be harder to classify.

## Next steps

- Fine-tune BERT on a much larger sample (e.g. on a GPU) for a fair comparison
- Test both models on real customer reviews to check how well they transfer
- Use LIME or SHAP to explain individual predictions

## Tech stack

Python · pandas · NumPy · NLTK · scikit-learn · PyTorch · Hugging Face Transformers · Matplotlib · seaborn · WordCloud · Jupyter

## How to run

1. Install the dependencies:
   ```bash
   pip install pandas numpy nltk scikit-learn torch transformers matplotlib seaborn wordcloud jupyter
   ```
2. Download `emotion_sentimen_dataset.csv` from Kaggle and put it in the project folder.
3. Open `Sentiment Evaluation Of Customer Feedback.ipynb` and run all cells. A GPU is recommended for the BERT section.
