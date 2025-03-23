# Twitter-Sentiment-Analysis
This project analyzes sentiments from Twitter data using Natural Language Processing (NLP). It classifies tweets as positive, negative using machine learning classifiers - Logistic Regression and Random Forest trained on Sentiment140 dataset.

**Table of Content**

1. Importing all the essential libraries
    - ZipFile : to extract data from zip file
    - numpy : library for numerical computing in Python, widely used for handling large datasets and performing mathematical operations efficiently.
    - pandas : Python library used for data manipulation and analysis
    - skleran : TfidfVectorizer, test_train_split, Logistic Regression, accuracy_score, f1_score
    - re : regular expression
    - nltk : stopswords, porterstemmer
    - pickle :to save and deploy model over web
2. Loading the dataset
    - Sentiment140 dataset
    - ```pip install kaggle
    - ```!mkdir -p ~/.kaggle
    - ```!cp kaggle.json ~/.kaggle/
    - ```!chmod 600 ~/.kaggle/kaggle.json
    - ```!kaggle datasets download -d kazanova/sentiment140
    - zipfile is extracted and csv file is loaded
3. Data Preprocessing and Transformation
    - check if there exist null values
    - check data distribution
    - convert target label to 1 or 0
    - regular expression is used to remove non-alphabets
    - stemming : a text preprocessing technique that reduces words to their base or root form (stem) by removing affixes (prefixes and suffixes), improving accuracy and efficiency in tasks like information retrieval and text analysis.
    - X : stemmed_content.values
    - Y : target.values
4. Dividing the dataset
    - dataset is divided in to 20% test size, stratify by Y(prevents imbalance in distribution of data) and random state = 2(ensures randomization on every run)
    - _TfidfVectorizer_ (Term Frequency-Inverse Document Frequency Vectorizer) from Scikit-learn is used to convert textual data (tweets) into numerical features for machine learning models. It assigns importance to words based on their frequency in a tweet and across the dataset.
5. Training and Evaluation
    - LogisticRegression(max_iter = 1000) Accuracy_score : 77.66%, F1 score : 78.08%
    - RandomForestClassifier(n_estimators=50,n_jobs=-1,random_state=42) Accuracy_score : 68.97%, F1 score : 70.15%
    - _Accuracy_score_ : Measures the proportion of correctly classified tweets among the total tweets.
    - _F1_score_ : A harmonic mean of Precision and Recall, which ensures a balance between false positives and false negatives. It is especially useful for imbalanced datasets.
    - _Precision_ : Measures how many of the predicted positive tweets are actually positive.
    - _Recall_ : Measures how many actual positive tweets were correctly identified.
7. Creating pickle file
   - creating a pickle file using .sav file and dump in file location
