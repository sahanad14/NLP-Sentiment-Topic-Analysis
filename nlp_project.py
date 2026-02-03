import pandas as pd
import re
from deep_translator import GoogleTranslator

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("reviews.csv")

# =========================
# 2. Translate to English
# =========================
def translate_text(text):
    try:
        return GoogleTranslator(source='auto', target='en').translate(text)
    except:
        return text

print("Translating text...")
df["translated"] = df["text"].apply(translate_text)

# =========================
# 3. Clean Text
# =========================
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text

df["clean_text"] = df["translated"].apply(clean_text)

# =========================
# 4. Encode Target
# =========================
df["sentiment"] = df["sentiment"].map({
    "Positive": 1,
    "Negative": 0,
    "Neutral": 2
})

# =========================
# 5. Visualize Sentiment Distribution
# =========================
plt.figure(figsize=(6,4))
sns.countplot(x="sentiment", data=df)
plt.title("Sentiment Distribution (0=Negative,1=Positive,2=Neutral)")
plt.show()

# =========================
# 6. Train Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"], test_size=0.2, random_state=42
)

# =========================
# 7. TF-IDF
# =========================
vectorizer = TfidfVectorizer(
    stop_words=stopwords.words("english"),
    max_features=500
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# =========================
# 8. Sentiment Model
# =========================
model = LogisticRegression()
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("\nSENTIMENT ANALYSIS RESULT")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# 9. Confusion Matrix
# =========================
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# =========================
# 10. Topic Modeling
# =========================
print("\nTOPIC MODELING RESULT")

lda = LatentDirichletAllocation(
    n_components=3,
    random_state=42
)

lda.fit(X_train_vec)

words = vectorizer.get_feature_names_out()

for i, topic in enumerate(lda.components_):
    print(f"\nTopic {i+1}:")
    topic_words = [words[j] for j in topic.argsort()[-10:]]
    print(", ".join(topic_words))
    
    # WordCloud for each topic
    wordcloud = WordCloud(width=600, height=300, background_color="white").generate(" ".join(topic_words))
    plt.figure(figsize=(8,4))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Topic {i+1} Word Cloud")
    plt.show()

# =========================
# 11. Test New Sentence
# =========================
test_text = "ಈ ಪ್ರಾಡಕ್ಟ್ ತುಂಬಾ ಕೆಟ್ಟದು"  # Kannada example
translated = translate_text(test_text)
cleaned = clean_text(translated)
vector = vectorizer.transform([cleaned])

prediction = model.predict(vector)[0]

print("\nTEST PREDICTION")
print("Input:", test_text)

if prediction == 1:
    print("Sentiment: Positive")
elif prediction == 0:
    print("Sentiment: Negative")
else:
    print("Sentiment: Neutral")
