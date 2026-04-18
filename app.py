import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv(r"C:\Users\HEMANGI SOLANKI\Desktop\KS\sem-10\Brainybeam\BestSellerAnalytics\bestsellers_with_categories_2022_03_27.csv")

X = df[['Reviews', 'User_Rating', 'Genre']]
X = pd.get_dummies(X, columns=['Genre'], drop_first=True)
y = df['Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv(r"C:\Users\HEMANGI SOLANKI\Desktop\KS\sem-10\Brainybeam\BestSellerAnalytics\bestsellers_with_categories_2022_03_27.csv")

# Prepare data
X = df[['Reviews', 'User_Rating', 'Genre']]
X = pd.get_dummies(X, columns=['Genre'], drop_first=True)
y = df['Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


st.title("📚 Book Price Prediction App")

st.sidebar.header("📊 Input Features")

rating = st.sidebar.slider("User Rating", 0.0, 5.0, 4.0)
reviews = st.sidebar.number_input("Number of Reviews", min_value=0)
genre = st.sidebar.selectbox("Genre", ["Fiction", "Non Fiction"])


st.write("###  Input Summary")
st.write(f"Rating: {rating}")
st.write(f"Reviews: {reviews}")
st.write(f"Genre: {genre}")

genre_value = 1 if genre == "Fiction" else 0
input_data = [[reviews, rating, genre_value]]

# Prediction
if st.button("Predict Price"):
    prediction = model.predict(input_data)
    
    usd = prediction[0]
    inr = usd * 83   # conversion rate (approx)

    st.success("Prediction generated successfully!")
    st.markdown(f"## Estimated Price: ${usd:.2f} (₹{inr:.2f})")