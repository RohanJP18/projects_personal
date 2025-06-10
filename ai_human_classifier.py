import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import shap
import nltk

nltk.download('punkt')
nltk.download('movie_reviews')
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_curve, roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance

from nltk.corpus import movie_reviews
import random
from sentence_transformers import SentenceTransformer

df = pd.read_csv("AI_Human.csv")
df.columns = ['text', 'label']
df = df.sample(n=8000, random_state=42).reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"\n", " ", text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    return text

df['cleaned_text'] = df['text'].apply(clean_text)

def train_naive_bayes_sentiment():
    docs = [(list(movie_reviews.words(fileid)), category)
            for category in movie_reviews.categories()
            for fileid in movie_reviews.fileids(category)]
    random.shuffle(docs)
    all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
    word_features = list(all_words)[:2000]
    
    def document_features(doc):
        words = set(doc)
        return {word: (word in words) for word in word_features}
    
    featuresets = [(document_features(d), c) for (d, c) in docs]
    classifier = nltk.NaiveBayesClassifier.train(featuresets)
    return classifier, word_features

nb_classifier, word_features = train_naive_bayes_sentiment()

def nb_sentiment(text):
    try:
        words = nltk.word_tokenize(str(text).lower())
        feats = {word: (word in words) for word in word_features}
        dist = nb_classifier.prob_classify(feats)
        return dist.prob('pos') - dist.prob('neg')
    except:
        return 0

def extract_features(df):
    features = pd.DataFrame()
    features['char_count'] = df['cleaned_text'].apply(len)
    features['word_count'] = df['cleaned_text'].apply(lambda x: len(str(x).split()))
    features['avg_word_len'] = df['cleaned_text'].apply(lambda x: np.mean([len(w) for w in str(x).split()]) if x else 0)
    features['sentence_std'] = df['text'].apply(lambda x: np.std([len(s.split()) for s in re.split(r'[.!?]', str(x)) if s]) if str(x) else 0)
    features['question_marks'] = df['text'].apply(lambda x: str(x).count('?'))
    features['parentheses'] = df['text'].apply(lambda x: str(x).count('(') + str(x).count(')'))
    features['semicolons'] = df['text'].apply(lambda x: str(x).count(';'))
    
    discourse_markers = ['however', 'therefore', 'moreover', 'thus', 'furthermore']
    features['discourse_markers'] = df['cleaned_text'].apply(lambda x: sum(1 for w in str(x).split() if w in discourse_markers))
    
    features['sentiment'] = df['cleaned_text'].apply(nb_sentiment)
    
    pronouns = ['i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours']
    features['authorial_voice'] = df['cleaned_text'].apply(lambda x: sum(1 for p in str(x).split() if p in pronouns))
    
    equivocal = ['maybe', 'perhaps', 'possibly', 'might', 'could', 'seems']
    features['equivocal'] = df['cleaned_text'].apply(lambda x: sum(1 for w in str(x).split() if w in equivocal))
    
    return features

linguistic_features = extract_features(df)

vectorizer = TfidfVectorizer(max_features=100)
tfidf_matrix = vectorizer.fit_transform(df['cleaned_text'])
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=[f"tfidf_{i}" for i in range(100)])

embedder = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedder.encode(df['cleaned_text'].tolist(), show_progress_bar=True)
embed_df = pd.DataFrame(embeddings, columns=[f"embed_{i}" for i in range(embeddings.shape[1])])

X = pd.concat([linguistic_features.reset_index(drop=True), tfidf_df, embed_df], axis=1)
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Permutation Importance
perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
linguistic_importances = perm.importances_mean[:len(linguistic_features.columns)]
feature_names = linguistic_features.columns

plt.figure(figsize=(10, 6))
sns.barplot(x=linguistic_importances, y=feature_names)
plt.title("Top Linguistic Features by Permutation Importance")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

# Feature Importances
importances = model.feature_importances_
imp_df = pd.Series(importances[:len(linguistic_features.columns)], index=linguistic_features.columns)
imp_df.sort_values().plot(kind='barh', figsize=(8,6), title="Random Forest Feature Importances")
plt.tight_layout()
plt.show()

# Feature Distributions by Label
df_plot = linguistic_features.copy()
df_plot['label'] = y

for col in linguistic_features.columns:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df_plot, x='label', y=col)
    plt.title(f"{col} Distribution by Label (0=Human, 1=AI)")
    plt.tight_layout()
    plt.show()

# Mean Feature Differences
mean_diff = df_plot.groupby('label').mean().T
mean_diff['diff'] = mean_diff[1] - mean_diff[0]
mean_diff = mean_diff.sort_values('diff')

plt.figure(figsize=(8, 6))
sns.barplot(x='diff', y=mean_diff.index, data=mean_diff)
plt.axvline(0, color='gray', linestyle='--')
plt.title("Mean Feature Differences (AI - Human)")
plt.xlabel("Mean Difference")
plt.tight_layout()
plt.show()

# Correlation Heatmap
corr_matrix = linguistic_features.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Heatmap of Linguistic Features")
plt.tight_layout()
plt.show()

# Pairplot
subset = linguistic_features[['char_count', 'word_count', 'avg_word_len', 'sentiment']].copy()
subset['label'] = y.astype(int)
sns.pairplot(subset, hue='label', diag_kind='hist', plot_kws={'alpha':0.6})
plt.suptitle("Pairwise Feature Relationships (Colored by Label)", y=1.02)
plt.show()

# ROC Curve
probs = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, probs)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, probs):.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, probs)

plt.figure(figsize=(6, 4))
plt.plot(recall, precision, label=f"Avg Precision = {average_precision_score(y_test, probs):.2f}")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.show()

# Label Distribution
df['label'].value_counts().plot(kind='pie', labels=['Human', 'AI'], autopct='%1.1f%%', 
                              colors=['#66b3ff','#99ff99'], startangle=90)
plt.title('Dataset Label Distribution')
plt.ylabel("")
plt.tight_layout()
plt.show()

# Correlation with Label
corrs = df_plot.corr()['label'].drop('label').sort_values()
plt.figure(figsize=(8, 6))
sns.barplot(x=corrs.values, y=corrs.index, palette="vlag")
plt.axvline(0, color='gray', linestyle='--')
plt.title("Pearson Correlation with Label (0=Human, 1=AI)")
plt.xlabel("Correlation Coefficient")
plt.tight_layout()
plt.show()

# SHAP Summary Plot
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)
shap.summary_plot(shap_values[1], X_test, feature_names=list(X.columns), plot_size=(12,8))
plt.tight_layout()
plt.show()