[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/kntb0107/fraud-detection-in-online-consumer-reviews-using-machine-learning-techniques/main/4.Deployment.py#fraud-detection-in-online-consumer-reviews-using-machine-learning-techniques)

# Fraud Detection in Online Consumer Reviews Using Machine Learning Techniques
Final Year Project describing the path to creating classifiers which aids in identifying the fake reviews from the real one using an Amazon dataset.

# Abstract
In today's world, both businesses and customers believe reviews to be quite beneficial. It's no surprise that review fraud has devalued the whole experience, from nasty reviews putting harm to the business's credibility to breaking international laws. This has been seen as a developing problem, and because it is related to natural language processing, it was critical to develop various machine learning methodologies and techniques to achieve a breakthrough in this sector. Many e-commerce sites, such as Amazon, have their systems in place, including Verified Purchase, which labels review language as accurate when items are purchased directly from the website. This work proposes to use Amazon's verified purchases label to train three classifiers for supervised training on Amazon’s labelled dataset. MNB, SVM, and LR were chosen as classifiers, and model tuning was done using two distinct vectorizers, Count Vectorizer and TF-IDF Vectorizers. Overall, all of the trained models had an accuracy rate of 80%, indicating that the vectorizers functioned admirably and that there are distinctions between false and actual reviews. Out of the two, the count vectorizer improved the models' performance more, and out of the three inside counts, LR performed the best, with an accuracy rate of 85% and a recall rate of 92%. The LR classifier was used, and it was accessible to the public to see if the reviews entered were genuine or not, with a probability score.

# Proceeding with the files
The notebooks and the python file has been numbered in order, and hence for easier readibility please refer to them in order.

This project has already been deployed online, and hence you can simply click the button on top to be referred to the web application


# MISC

DATASET: https://bit.ly/2Rzvjqf [KAGGLE]

main dataset used: amazon_reviews_2019.csv



# INFO

IDE: Jupyter Notebook

Language: Python

Models utlized: Logistic Regression, SVM, MNB 

Deployment Platform: Streamlit.io


# Future Enchancements
The dataset utilized in this research was found to be focused on reviews from the United Kingdom, namely from the grocery section of well-known supermarkets and pharmacies available on Amazon. Furthermore, because the reviews have mainly positive sentiments, it can be argued that the web application performed better when reviews fit the aforementioned criteria against product reviews from other areas and sites when it came down to the classification. 

This restriction is a symptom of a larger problem that this industry has been grappling with a lack of a standardized dataset for the sole purpose of detecting and analysing fraudulent reviews. This introduces external elements that may have an impact on the classifiers' performance, and because authors in past works of literature have taken liberties, it is difficult to say which sort of setup and models operate best in which scenario.

One of the future enhancements suggested is the development of a large dataset with various types of online reviews from various backgrounds so that the study and performance of these reviews can be as unbiased as possible, and researchers can focus entirely on developing automated detection techniques similar to modern email spam detection systems.

