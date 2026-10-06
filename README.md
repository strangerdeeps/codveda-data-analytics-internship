# Codveda Data Analytics Internship - Submission

| Level | Tasks completed | Files |
|---|---|---|
| 1 (Basic) | Data Cleaning & Preprocessing, Exploratory Data Analysis | `level1/Level1_Data_Cleaning_and_EDA.ipynb` (+ `.py`), `raw_data.csv`, `cleaned_data.csv`, `plots/` |
| 2 (Intermediate) | Regression Analysis, K-Means Clustering | `level2/Level2_Regression_and_KMeans.ipynb` (+ `.py`), `plots/` |
| 3 (Advanced) | Classification (with grid search), NLP Sentiment Analysis | `level3/Level3_Classification.ipynb`, `level3/Level3_NLP_Sentiment.ipynb` (+ `.py`), `reviews.csv`, `plots/` |

## Key results
- **L1:** 30 duplicates removed, missing values imputed (median/mode), mixed date/gender/city/category formats standardized; unit price and order total correlate at 0.85.
- **L2:** Simple regression (BMI) R2 = 0.23; multiple regression R2 = 0.45. K-Means on wine data: k = 3 (silhouette 0.285).
- **L3:** Tuned Logistic Regression best on test set: accuracy 0.974, precision 0.973, recall 0.986, F1 0.979.

## Run
`pip install pandas numpy matplotlib seaborn scikit-learn` (+ `nltk textblob wordcloud` for the NLP task), then run each script from inside its level folder.

## Notes
- The NLP task uses nltk + TextBlob + wordcloud when installed and a built-in fallback otherwise; the notebook prints which tools ran. The demo reviews are synthetic, so 100% agreement with their labels is not a real accuracy figure. Rerun with real reviews and the libraries installed for meaningful results.
- Level 1 uses a synthetic messy dataset; Level 2 and the Level 3 classification task use scikit-learn built-in datasets.
