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
  <img width="600" height="400" alt="Sentiment_Distribution" src="https://github.com/user-attachments/assets/198b6b45-ae7e-46c9-b3a1-55ae0ab1f2e3" />


### ✔ Topic Modeling
- Extracts major themes discussed in the reviews

### ✔ Visualization
- Generates WordCloud to highlight frequent keywords
  <img width="500" height="400" alt="Confusion_Matrix" src="https://github.com/user-attachments/assets/e453e537-4ffe-4efe-8ecb-c780424285cb" />
<img width="800" height="400" alt="topic1_word_cloud" src="https://github.com/user-attachments/assets/dad1a438-9aee-45b3-80b7-0dc6760e1a07" />
<img width="800" height="400" alt="topic2_word_cloud" src="https://github.com/user-attachments/assets/22dce7e5-e258-4e17-8e66-3ad123770492" />
<img width="800" height="400" alt="topic3_word_cloud" src="https://github.com/user-attachments/assets/98805d4f-f676-4756-bc10-8b10eb5ac92a" />


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


