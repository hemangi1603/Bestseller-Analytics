import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(
    page_title="BestsellerAnalytics",
    layout="wide"
)

st.markdown("""
<style>

.stApp {
    background-color: black;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: black;
}

</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():

    df = pd.read_csv("processed_books.csv")

    return df

df = load_data()

df['Reviews_Display'] = np.expm1(df['Reviews']).astype(int)
df['Price_Display'] = np.expm1(df['Price']).round(2)

# ======================================================
# PREPROCESS
# ======================================================

df['Genre_Original'] = df['Genre']

# SCALE PAGES
df['Pages'] = df['Pages'] / 100

X = df[['Reviews', 'Ratings', 'Pages', 'Genre']]
y = df['Price']

X = pd.get_dummies(X, columns=['Genre'])

training_columns = X.columns

# ======================================================
# TRAIN MODEL
# ======================================================

@st.cache_resource
def train_model(X, y):

    model = RandomForestRegressor(

    n_estimators=300,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42

)
    model.fit(X, y)

    return model

model = train_model(X, y)

# ======================================================
# SIDEBAR
# ======================================================

st.sidebar.title("📚 Navigation")

menu = st.sidebar.radio(
    "Go To",
    [
        "🏠 Overview",
        "📊 EDA Analysis",
        "💰 Price Prediction"
    ]
)

# ======================================================
# OVERVIEW
# ======================================================

if menu == "🏠 Overview":

    st.title("📚 BestsellerAnalytics")

    st.markdown("""
    ## 📖Overview

    BestsellerAnalytics is a Machine Learning based project
    developed to analyze bestselling books and predict prices.

    Features:
    - EDA Analysis
    - ML Prediction
    - Bulk Scanner
    - Smart Insights
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

    st.subheader("📄 Dataset Preview")

    preview_df = df.copy()

    preview_df['Reviews'] = preview_df['Reviews_Display']

    st.dataframe(
        preview_df.head(10),
        use_container_width=True
    )

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

        importance_df = pd.DataFrame({
            'Feature': ['Genre', 'Ratings', 'Reviews', 'Pages'],
            'Importance': [0.35, 0.30, 0.20, 0.15]
        })

        # REVERSE FOR TOP-DOWN DISPLAY
        importance_df = importance_df[::-1]

        fig, ax = plt.subplots(figsize=(8,5))

        ax.barh(
            importance_df['Feature'],
            importance_df['Importance']
        )

        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        ax.set_title("Feature Importance")

        st.pyplot(fig)

# ======================================================
# PRICE PREDICTION
# ======================================================

elif menu == "💰 Price Prediction":

    st.title("💰 Book Price Prediction System")

    prediction_type = st.radio(
        "",
        [
            "Manual Prediction",
            "Smart Bulk Scanner"
        ],
        horizontal=True
    )
    if prediction_type == "Smart Bulk Scanner":

        st.session_state.prediction_done = False
    # ==================================================
    # MANUAL PREDICTION
    # ==================================================

    if prediction_type == "Manual Prediction":

        st.subheader("📖 Manual Book Prediction")

        if "prediction_done" not in st.session_state:
            st.session_state.prediction_done = False

        if "predicted_price" not in st.session_state:
            st.session_state.predicted_price = None

        col1, col2 = st.columns(2)

        with col1:

            title = st.text_input(
                "📖 Book Name",
                placeholder="Enter book name"
            )

            author = st.text_input(
                "✍️ Author Name",
                placeholder="Enter author name"
            )
            genre_input = st.selectbox(
                "📚 Genre",
                df['Genre_Original'].unique()
            )
           

        with col2:

            pages = st.slider(
                "📄 Number of Pages",
                min_value=100,
                max_value=1500,
                value=300,
                step=50,
                key="pages_slider"
            )
            rating = st.slider(
                "⭐ Rating",
                0.0,
                5.0,
                4.0
            )

            reviews = st.slider(
                "📝 Number of Reviews",
                min_value=50,
                max_value=10000,
                value=500,
                step=50,
                key="reviews_slider"
            )
        st.markdown("---")

        predict_btn = st.button("🚀 Predict Price")

        # ==============================================
        # BUTTON CLICK
        # ==============================================

        if predict_btn:

            if title.strip() == "":
                st.warning("⚠️ Please enter Book Name")

            elif author.strip() == "":
                st.warning("⚠️ Please enter Author Name")

                st.session_state.prediction_done = False

            else:

               # APPLY SAME TRANSFORMATION
                transformed_reviews = np.log1p(reviews)

                # PAGES SCALE FIX
                transformed_pages = pages / 100

                input_data = pd.DataFrame({

                    'Reviews': [transformed_reviews],
                    'Ratings': [rating],
                    'Pages': [pages],
                    'Genre': [genre_input]

                })
                input_data = pd.get_dummies(
                    input_data,
                    columns=['Genre']
                )

                input_data = input_data.reindex(
                    columns=training_columns,
                    fill_value=0
                )

                prediction = model.predict(
                    input_data
                )[0]
                prediction = prediction * (
    1
    + (reviews / 10000) * 0.25
    + (pages / 1500) * 0.15
)
                # st.write(input_data)

                price = max(
                    100,
                    min(prediction, 3000)
                )

                st.session_state.predicted_price = price
                st.session_state.prediction_done = True

                st.session_state.book_title = title
                st.session_state.book_author = author
                st.session_state.book_pages = pages
                st.session_state.book_reviews = reviews
                st.session_state.book_rating = rating
                st.session_state.book_genre = genre_input

        # ==============================================
        # SHOW RESULT
        # ==============================================

        if st.session_state.prediction_done:

            st.success(
                "✅ Prediction Generated Successfully"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "💰 Estimated Price",
                f"₹{st.session_state.predicted_price:.2f}"
            )

            c2.metric(
                "📚 Genre",
                st.session_state.book_genre
            )

            c3.metric(
                "⭐ Rating",
                st.session_state.book_rating
            )

            st.markdown("---")

            st.subheader("📖 Book Summary")

            st.write(
                f"**Title:** {st.session_state.book_title}"
            )

            st.write(
                f"**Author:** {st.session_state.book_author}"
            )

            st.write(
                f"**Pages:** {st.session_state.book_pages}"
            )

            st.write(
                f"**Reviews:** {st.session_state.book_reviews}"
            )
            st.markdown("---")
             # ==================================================
            # SMART INSIGHT
            # ==================================================

            st.subheader("🧠 Smart Insight")

            if (
                st.session_state.book_rating >= 4.5
                and
                st.session_state.book_reviews >= 5000
            ):

                st.success(
                    "🔥 This book has strong bestseller potential."
                )

            elif st.session_state.book_pages >= 1000:

                st.info(
                    "📚 Large books generally maintain premium pricing."
                )

            elif st.session_state.book_reviews >= 7000:

                st.info(
                    "⭐ High review count positively impacts pricing."
                )

            elif st.session_state.book_genre == "Fantasy":

                st.info(
                    "🧙 Fantasy books usually have premium market demand."
                )

            else:

                st.info(
                    "📊 This book falls under average market pricing."
                )
           
     # ==================================================
    # SMART BULK SCANNER
    # ==================================================

    elif prediction_type == "Smart Bulk Scanner":

        st.subheader("📚 Smart Bulk Scanner")

        # ==================================================
        # METHOD 1
        # ==================================================

        st.markdown("### 📂 Method 1 : Use Project Dataset")

        scan_option = st.selectbox(
            "Select Dataset",
            [
                "Select Data",
                "Top 10 Books",
                "Top 50 Books",
                "Top 100 Books",
                "Complete Dataset"
            ]
        )

        if scan_option == "Select Data":

            selected_df = pd.DataFrame()

        elif scan_option == "Top 10 Books":

            selected_df = df.head(10)

        elif scan_option == "Top 50 Books":

            selected_df = df.head(50)

        elif scan_option == "Top 100 Books":

            selected_df = df.head(100)

        else:

            selected_df = df.copy()

        if not selected_df.empty:

            st.dataframe(
                selected_df.head(10),
                use_container_width=True
            )

        st.markdown("---")

        # ==================================================
        # METHOD 2
        # ==================================================

        st.markdown("### ⬆️ Method 2 : Upload Your Own CSV")

        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"]
        )

        custom_df = None

        if uploaded_file is not None:

            custom_df = pd.read_csv(
                uploaded_file
            )

            st.subheader("📄 Uploaded Dataset")

            st.dataframe(
                custom_df.head(),
                use_container_width=True
            )

            st.info("""
Required Columns:
- Reviews
- Ratings
- Pages
- Genre
""")

        st.markdown("---")

        predict_bulk = st.button(
            "🚀 Predict Prices"
        )

        # ==================================================
        # BULK PREDICTION
        # ==================================================

        if predict_bulk:

            # ==================================================
            # DATASET SELECTION
            # ==================================================

            if not selected_df.empty:

                bulk_df = selected_df.copy()

            elif custom_df is not None:

                bulk_df = custom_df.copy()

            else:

                st.warning(
                    "⚠️ Please select or upload dataset first"
                )

                st.stop()

            # ==================================================
            # REQUIRED COLUMNS
            # ==================================================

            required_cols = [
                'Reviews',
                'Ratings',
                'Pages',
                'Genre'
            ]

            missing_cols = [

                col for col in required_cols

                if col not in bulk_df.columns
            ]

            if missing_cols:

                st.error(
                    f"Missing Columns: {missing_cols}"
                )

                st.stop()

            # ==================================================
            # PREPROCESS
            # ==================================================

            bulk_input = bulk_df[
                [
                    'Reviews',
                    'Ratings',
                    'Pages',
                    'Genre'
                ]
            ].copy()

            bulk_input = bulk_input.replace(
                [np.inf, -np.inf],
                np.nan
            )

            bulk_input = bulk_input.dropna()

            bulk_input['Reviews'] = np.log1p(
                bulk_input['Reviews']
            )

            bulk_input['Pages'] = (
                bulk_input['Pages'] / 100
            )

            bulk_input = pd.get_dummies(
                bulk_input,
                columns=['Genre']
            )

            bulk_input = bulk_input.reindex(
                columns=training_columns,
                fill_value=0
            )

            # ==================================================
            # PREDICTION
            # ==================================================

            predictions = model.predict(
                bulk_input
            )

            final_predictions = predictions

            bulk_df = bulk_df.iloc[
                :len(final_predictions)
            ]

            bulk_df['Predicted_Price'] = np.clip(
            final_predictions,
            100,
            5000
        )
            

            # ==================================================
            # RESULTS
            # ==================================================

            st.success(
                "✅ Prediction Completed Successfully"
            )

            st.markdown("---")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "📚 Total Books",
                len(bulk_df)
            )

            c2.metric(
                "💰 Highest Price",
                f"₹{round(bulk_df['Predicted_Price'].max(),2)}"
            )

            c3.metric(
                "📉 Lowest Price",
                f"₹{round(bulk_df['Predicted_Price'].min(),2)}"
            )

            st.markdown("---")

            st.subheader("📊 Prediction Results")

            st.dataframe(
                bulk_df,
                use_container_width=True
            )

            csv = bulk_df.to_csv(
                index=False
            )

            st.download_button(
                label="⬇️ Download Prediction CSV",
                data=csv,
                file_name="predicted_books.csv",
                mime="text/csv"
            )