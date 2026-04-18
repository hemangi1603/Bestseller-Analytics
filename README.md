📚 Book Price Prediction using Machine Learning

📌 Overview

This project analyzes Amazon bestselling books using Exploratory Data Analysis (EDA) and builds a Machine Learning model to predict book prices based on features like user rating, number of reviews, and genre. A Streamlit web application is also developed to provide real-time predictions.

🎯 Objectives

Analyze patterns in bestselling books
Understand factors affecting book prices
Build a predictive model for price estimation
Deploy a user-friendly web application

📂 Dataset

Source: Kaggle (Amazon Top 50 Bestselling Books)
Records: ~550
Features:
Name
Author
User Rating
Reviews
Price
Year
Genre

🔍 Exploratory Data Analysis (EDA)

Genre distribution (Fiction vs Non-Fiction)
Rating distribution
Reviews analysis
Price trends
Correlation between features

🤖 Machine Learning Model

Input Features:
User Rating
Reviews
Genre
Output:
Predicted Price
Models Used:
Linear Regression
Random Forest (Final Model)
Evaluation Metrics:
MAE (Mean Absolute Error)
RMSE (Root Mean Square Error)

📊 Results

Random Forest performed better than Linear Regression
Model successfully predicts book prices
Good performance on test data

🌐 Streamlit Application

The project includes an interactive web app where users can:

Input:
User Rating
Number of Reviews
Genre
Get:
Predicted Book Price (USD & INR)

⚙️ Technologies Used

Language: Python
Libraries:
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Streamlit

🚀 How to Run the Project

1. Install Dependencies
pip install pandas numpy matplotlib seaborn scikit-learn streamlit
2. Run Streamlit App
python -m streamlit run app.py

🔮 Future Scope

Use larger dataset
Add more features (author, publisher, year trends)
Improve model accuracy
Deploy application online

👩‍💻 Author

Hemangi Solanki

⭐ If you like this project, give it a star!
