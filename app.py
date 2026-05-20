import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
st.set_page_config(
    page_title="Book Price Predictor",
    layout="wide"
)
st.markdown("""
<style>

.stApp {
    background-color: black;
}

section[data-testid="stSidebar"] {
    background-color: black;
}

</style>
""", unsafe_allow_html=True)

# LOAD DATA

@st.cache_data
def load_data():

    df = pd.read_csv(
        r"processed_books.csv"
    )

    return df

df = load_data()

# PREPROCESS

df['Genre_Original'] = df['Genre']

X = df[['Reviews', 'Ratings', 'Pages', 'Genre']]
y = df['Price']

X = pd.get_dummies(X, columns=['Genre'])

# TRAIN MODEL

@st.cache_resource
def train_model(X, y):

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )
    model.fit(X, y)

    return model

model = train_model(X, y)

# SIDEBAR

st.sidebar.title("📚 Navigation")

menu = st.sidebar.radio(
    "Go To",
    [
        "🏠 Overview",
        "📊 EDA Analysis",
        "💰 Price Prediction"
    ]
)

# OVERVIEW

if menu == "🏠 Overview":

    st.title("📚 BestsellerAnalytics")

    st.markdown("""
    ## 📖 Project Overview

    BestsellerAnalytics is a Machine Learning based project developed
    to analyze Amazon bestselling books and predict book prices.

    The project performs complete Exploratory Data Analysis (EDA)
    to discover hidden insights such as:

    - Genre popularity
    - Rating trends
    - Author influence
    - Price distribution
    - Relationship between reviews and ratings

    Using Machine Learning algorithms like:

    - Linear Regression
    - Random Forest Regression

    the system predicts estimated book prices based on user inputs.

    This project helps understand how different book features
    affect pricing in the online book market.
    """)

    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "📖 Total Books",
        len(df)
    )

    col2.metric(
        "⭐ Average Rating",
        round(df['Ratings'].mean(), 2)
    )

    col3.metric(
        "💰 Average Price",
        f"₹{round(df['Price'].mean(), 2)}"
    )

    st.markdown("---")

    # ================= DATASET PREVIEW =================

    st.subheader("📄 Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # ================= FEATURES =================

    st.subheader(" Features Used in Prediction")

    feature_col1, feature_col2 = st.columns(2)

    with feature_col1:

        st.info("📝 Reviews")
        st.info("⭐ Ratings")

    with feature_col2:

        st.info("📄 Pages")
        st.info("📚 Genre")
# EDA ANALYSIS

elif menu == "📊 EDA Analysis":

    st.title("📊 Exploratory Data Analysis")
    plt.style.use('default')

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Distribution",
        "🔗 Relationships",
        "📊 Advanced",
        "⭐ Feature Importance"
    ])

    # TAB 1 — DISTRIBUTION

    with tab1:

        col1, col2 = st.columns(2)

        # ---------------- PRICE DISTRIBUTION ----------------

        with col1:

            st.subheader("Price Distribution")

            fig, ax = plt.subplots(figsize=(7,5))

            sns.histplot(
                df['Price'],
                kde=True,
                bins=30
            )

            plt.title("Price Distribution")

            st.pyplot(fig)

        # ---------------- GENRE DISTRIBUTION ----------------

        with col2:

            st.subheader("Genre Distribution")

            fig, ax = plt.subplots(figsize=(7,5))

            sns.countplot(
                x='Genre_Original',
                data=df
            )

            plt.xticks(rotation=45)

            plt.title("Genre Distribution")

            st.pyplot(fig)

        # ---------------- RATINGS DISTRIBUTION ----------------

        st.subheader("Ratings Distribution")

        fig, ax = plt.subplots(figsize=(8,5))

        sns.histplot(
            df['Ratings'],
            bins=10,
            kde=True
        )

        plt.title("Ratings Distribution")

        st.pyplot(fig)

    # TAB 2 — RELATIONSHIPS

    with tab2:

        col1, col2 = st.columns(2)

        # ---------------- RATINGS VS REVIEWS ----------------

        with col1:

            st.subheader("Ratings vs Reviews")

            fig, ax = plt.subplots(figsize=(7,5))

            sns.scatterplot(
                x='Ratings',
                y='Reviews',
                data=df
            )

            plt.title("Ratings vs Reviews")

            st.pyplot(fig)

        # ---------------- PRICE VS REVIEWS ----------------

        with col2:

            st.subheader("Price vs Reviews")

            fig, ax = plt.subplots(figsize=(7,5))

            sns.scatterplot(
                x='Reviews',
                y='Price',
                data=df
            )

            plt.title("Price vs Reviews")

            st.pyplot(fig)

    # TAB 3 — ADVANCED

    with tab3:

        # ---------------- HEATMAP ----------------

        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots(figsize=(8,6))

        corr = df.corr(numeric_only=True)

        sns.heatmap(
            corr,
            annot=True,
            cmap='coolwarm'
        )

        plt.title("Correlation Heatmap")

        st.pyplot(fig)

        # ---------------- PRICE BY GENRE ----------------

        st.subheader("Price by Genre")

        fig, ax = plt.subplots(figsize=(8,5))

        genre_order = [
            'Thriller',
            'Drama',
            'Romance',
            'Comedy',
            'Horror',
            'Fantasy'
        ]

        sns.boxplot(
            x='Genre_Original',
            y='Price',
            data=df,
            order=genre_order,
            ax=ax
        )

        ax.set_title("Price by Genre")

        ax.set_xlabel("Genre")

        ax.set_ylabel("Price")

        plt.xticks(rotation=20)

        st.pyplot(fig)

        # ---------------- TOP AUTHORS ----------------

        st.subheader("📚 Top 10 Authors")

        top_authors = df['Author'].value_counts().head(10)

        fig, ax = plt.subplots(figsize=(8,5))

        sns.barplot(
            x=top_authors.values,
            y=top_authors.index
        )

        plt.title("Top 10 Authors")

        st.pyplot(fig)

        # ---------------- YEAR TREND ----------------

        if 'Publication_Date' in df.columns:

            df['Publication_Date'] = pd.to_datetime(
                df['Publication_Date'],
                errors='coerce'
            )

            df['Year'] = df['Publication_Date'].dt.year

            yearly_price = df.groupby('Year')['Price'].mean().reset_index()

            st.subheader("📈 Average Price Trend Over Years")

            fig, ax = plt.subplots(figsize=(8,5))

            sns.lineplot(
                x='Year',
                y='Price',
                data=yearly_price,
                marker='o'
            )

            plt.title("Average Price Trend Over Years")

            st.pyplot(fig)

        # MODEL COMPARISON

        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error
        from sklearn.metrics import mean_squared_error

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # ---------- LINEAR REGRESSION ----------

        lr_model = LinearRegression()

        lr_model.fit(X_train, y_train)

        y_pred_lr = lr_model.predict(X_test)

        mae_lr = mean_absolute_error(y_test, y_pred_lr)

        rmse_lr = np.sqrt(
            mean_squared_error(y_test, y_pred_lr)
        )

        # ---------- RANDOM FOREST ----------

        rf_model = RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        )

        rf_model.fit(X_train, y_train)

        y_pred_rf = rf_model.predict(X_test)

        mae_rf = mean_absolute_error(y_test, y_pred_rf)

        rmse_rf = np.sqrt(
            mean_squared_error(y_test, y_pred_rf)
        )

        # ---------- MODEL COMPARISON GRAPH ----------

        st.subheader("🤖 Model Comparison")

        models = ['Linear Regression', 'Random Forest']

        mae_values = [mae_lr, mae_rf]
        rmse_values = [rmse_lr, rmse_rf]

        x = np.arange(len(models))
        width = 0.25

        fig, ax = plt.subplots(figsize=(7,5))

        ax.bar(
            x - width/2,
            mae_values,
            width,
            label='MAE'
        )

        ax.bar(
            x + width/2,
            rmse_values,
            width,
            label='RMSE'
        )

        ax.set_xticks(x)
        ax.set_xticklabels(models)

        plt.title("Model Comparison")
        plt.xlabel("Model")
        plt.ylabel("Error Value")

        plt.legend()

        st.pyplot(fig)

    # TAB 4 — FEATURE IMPORTANCE

    with tab4:

        st.subheader("⭐ Feature Importance")

        importance = model.feature_importances_

        genre_total = 0

        for col, imp in zip(X.columns, importance):

            if "Genre_" in col:
                genre_total += imp

        feature_map = {
            'Reviews': importance[list(X.columns).index('Reviews')],
            'Ratings': importance[list(X.columns).index('Ratings')],
            'Pages': importance[list(X.columns).index('Pages')],
            'Genre': genre_total
        }

        feat_imp = pd.Series(feature_map)

        fig, ax = plt.subplots(figsize=(8,5))

        feat_imp.sort_values().plot(
            kind='barh',
            ax=ax
        )

        plt.title("Feature Importance")

        plt.xlabel("Importance Score")

        plt.ylabel("Features")

        st.pyplot(fig)

# PRICE PREDICTION

elif menu == "💰 Price Prediction":

    st.title("💰 Book Price Prediction System")
    st.markdown("---")

    col1, col2 = st.columns(2)

    # LEFT COLUMN

    with col1:

        title = st.text_input(
            "📖 Book Name",
            placeholder="Enter book title"
        )

        author = st.text_input(
            "✍️ Author Name",
            placeholder="Enter author name"
        )
        
        genre_input = st.selectbox(
            "📚 Select Genre",
            sorted(df['Genre_Original'].unique())
        ) 
       
    # RIGHT COLUMN

    with col2:

        pages = st.select_slider(
            "📄 Number of Pages",
            options=[
            50, 100, 150, 200, 250,
            300, 400, 500, 700,
            1000, 1500, 2000
            ],
            value=300
        )

        reviews = st.select_slider(
            "📝 Number of Reviews",
            options=[
            0, 50, 100, 200, 500,
            1000, 2000, 5000,
            10000, 20000, 50000
            ],
            value=1000
        )
        rating = st.slider(
            "⭐ Rating",
            min_value=0.0,
            max_value=5.0,
            value=4.0,
            step=0.1
        )

    st.markdown("---")

    predict_btn = st.button(
        "🚀 Predict Price",
        use_container_width=True
    )

    # PREDICTION

    if predict_btn:

        input_data = pd.DataFrame({

            'Reviews': [np.log1p(reviews)],

            'Ratings': [rating],

            'Pages': [pages],

            'Genre': [genre_input]

        })

        # Convert categorical genre

        input_data = pd.get_dummies(
            input_data,
            columns=['Genre']
        )

        # Match training columns

        input_data = input_data.reindex(
            columns=X.columns,
            fill_value=0
        )

        # Predict

        prediction = model.predict(input_data)[0]

        # Realistic range

        prediction = max(100, min(prediction, 3000))

        st.success("✅ Prediction Generated Successfully")

        st.markdown("## 📊 Prediction Result")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "💰 Estimated Price",
            f"₹{prediction:.2f}"
        )

        result_col2.metric(
            "📚 Genre",
            genre_input
        )

        result_col3.metric(
            "⭐ Rating",
            rating
        )

        st.markdown("---")

        # BOOK SUMMARY

        st.subheader("📖 Book Information")

        summary_col1, summary_col2 = st.columns(2)

        with summary_col1:

            st.write(f"**Title:** {title}")

            st.write(f"**Author:** {author}")

            st.write(f"**Genre:** {genre_input}")

        with summary_col2:

            st.write(f"**Pages:** {pages}")

            st.write(f"**Reviews:** {reviews}")

            st.write(f"**Rating:** {rating}")

        st.markdown("---")

        # SMART INSIGHTS

        st.subheader("🧠 Smart Insights")

        insight_found = False

        # ---------- Rating Insight ----------

        if rating >= 4.5:

            st.success(
                "🌟 Excellent rating detected. Highly rated books usually attract premium pricing."
            )

            insight_found = True

        elif rating <= 2:

            st.warning(
                "⚠️ Low-rated books generally have lower customer demand and pricing."
            )

            insight_found = True

        # ---------- Reviews Insight ----------

        if reviews >= 10000:

            st.info(
                "🔥 This book has very high review engagement, indicating strong popularity."
            )

            insight_found = True

        elif reviews <= 100:

            st.info(
                "📉 Lower review count may indicate limited market popularity."
            )

            insight_found = True

        # ---------- Pages Insight ----------

        if pages >= 700:

            st.info(
                "📚 High page count increases production cost and selling price."
            )

            insight_found = True

        elif pages <= 150:

            st.info(
                "📘 Short books are usually more affordable."
            )

            insight_found = True

        # ---------- Genre Insight ----------

        if genre_input == "Fantasy":

            st.info(
                "🧙 Fantasy books often have premium collector-style pricing."
            )

            insight_found = True

        elif genre_input == "Thriller":

            st.info(
                "🔍 Thriller books are highly popular among readers."
            )

            insight_found = True

        elif genre_input == "Romance":

            st.info(
                "💕 Romance books usually remain budget-friendly and widely accessible."
            )

            insight_found = True

        elif genre_input == "Horror":

            st.info(
                "👻 Horror books often attract niche but loyal audiences."
            )

            insight_found = True

        # ---------- Final Insight ----------

        if not insight_found:

            st.info(
                "📊 This book falls into a moderate pricing and popularity category."
            )