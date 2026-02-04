# NLP Sentiment & Topic Analysis (Kannada → English)

## 📌 Project Overview
This project performs **Sentiment Analysis** and **Topic Modeling** on user reviews written in **Kannada**.
The reviews are first translated into **English**, then processed using **Natural Language Processing (NLP)** techniques to:
- Identify sentiment (**Positive / Negative**)
- Extract dominant discussion topics
- Visualize important words using **WordCloud**

---

## 🚀 Features
- Kannada to English translation using Google Translator
- Text preprocessing (stopword removal, tokenization)
- Sentiment classification using Machine Learning
- Topic Modeling using Latent Dirichlet Allocation (LDA)
- WordCloud visualization
- Command-line execution

---

## 🛠️ Technologies Used
- Python
- NLTK
- Scikit-learn
- GoogleTrans
- WordCloud
- Pandas
- Matplotlib

---

## 📂 Project Structure
NLP_Sentiment_Topic_Analysis/
│
├── nlp_project.py # Main Python script
├── reviews.csv # Input dataset (Kannada reviews)
├── requirements.txt # Project dependencies
├── README.md # Project documentation


---

## 📊 Output
### ✔ Sentiment Analysis
- Predicts whether a review is **Positive** or **Negative**
- Displays accuracy, precision, recall, and F1-score

### ✔ Topic Modeling
- Extracts major themes discussed in the reviews

### ✔ Visualization
- Generates WordCloud to highlight frequent keywords

---

## ▶️ How to Run the Project

### Step 1: Clone the repository
```bash
git clone https://github.com/sahanad14/NLP-Sentiment-Topic-Analysis.git
cd NLP-Sentiment-Topic-Analysis

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Run the project
python nlp_project.py

🧪 Sample Input
ಈ ಪ್ರಾಡಕ್ಟ್ ತುಂಬಾ ಕೆಟ್ಟದು

Sample Output
Sentiment: Negative

📈 Future Improvements

Improve model accuracy with larger datasets

Add web-based interface using Streamlit

Support multiple Indian languages

Add advanced deep learning models (LSTM / BERT)

👩‍💻 Author

Sahana D
Aspiring Data Scientist | NLP Enthusiast

GitHub: https://github.com/sahanad14

⭐ If you like this project, please give it a star!


---

## ✅ What you do next (VERY IMPORTANT)
1. Open **README.md** in VS Code or Notepad  
2. **Delete everything**
3. **Paste the above content**
4. Save file
5. Run these commands:

```powershell
git add README.md
git commit -m "Update README documentation"
git push