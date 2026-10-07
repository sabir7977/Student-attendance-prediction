
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

warnings.filterwarnings("ignore")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Attendance Prediction",
    page_icon="🎓",
    layout="centered"
)


# ============================================================
# TRAIN AND CACHE MODEL
# ============================================================

@st.cache_resource
def get_trained_model():

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    df = pd.read_csv(
        "Attendance_Prediction_2000.csv"
    )

    # --------------------------------------------------------
    # SEPARATE FEATURES AND TARGET
    # --------------------------------------------------------

    # Original 12 features are used
    X = df.drop(
        columns=[
            "student_id",
            "attendance",
            "absence_reason"
        ]
    )

    # --------------------------------------------------------
    # TARGET VARIABLE
    # --------------------------------------------------------

    y = df["attendance"].copy()

    # Normalize text labels
    if y.dtype == "object":

        y = (
            y.astype(str)
            .str.strip()
            .str.lower()
        )

        # Convert Present / Absent to 1 / 0
        if set(y.unique()).issubset(
            {"present", "absent"}
        ):

            y = y.map({
                "absent": 0,
                "present": 1
            })

    # Convert numeric-looking values
    else:

        y = pd.to_numeric(
            y,
            errors="coerce"
        )

    # Check for missing target values
    if y.isna().any():

        raise ValueError(
            "The attendance column contains "
            "invalid or missing values."
        )

    # --------------------------------------------------------
    # IDENTIFY CATEGORICAL AND NUMERICAL COLUMNS
    # --------------------------------------------------------

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[

            (
                "num",
                "passthrough",
                numerical_cols
            ),

            (
                "cat",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore"
                ),
                categorical_cols
            )
        ]
    )

    # --------------------------------------------------------
    # RANDOM FOREST MODEL
    # --------------------------------------------------------

    pipeline = Pipeline(
        steps=[

            (
                "preprocessor",
                preprocessor
            ),

            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1
                )
            )
        ]
    )

    # --------------------------------------------------------
    # TRAIN TEST SPLIT
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    pipeline.fit(
        X_train,
        y_train
    )

    # --------------------------------------------------------
    # MODEL EVALUATION
    # --------------------------------------------------------

    y_pred = pipeline.predict(
        X_test
    )

    # Accuracy
    acc = accuracy_score(
        y_test,
        y_pred
    )

    # Classification report
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    rf_model = pipeline.named_steps[
        "classifier"
    ]

    # Get encoded feature names
    encoded_features = (
        pipeline
        .named_steps["preprocessor"]
        .get_feature_names_out()
    )

    # Get importance values
    importance = (
        rf_model.feature_importances_
    )

    simple_names = []

    # --------------------------------------------------------
    # CONVERT ENCODED NAMES INTO SIMPLE NAMES
    # --------------------------------------------------------

    for feature in encoded_features:

        feature = feature.replace(
            "num__",
            ""
        )

        feature = feature.replace(
            "cat__",
            ""
        )

        if feature.startswith("age"):
            simple_name = "Age"

        elif feature.startswith("gender"):
            simple_name = "Gender"

        elif feature.startswith("course"):
            simple_name = "Course"

        elif feature.startswith("year"):
            simple_name = "Year"

        elif feature.startswith("parent_education"):
            simple_name = "Parent Education"

        elif feature.startswith("internet_access"):
            simple_name = "Internet Access"

        elif feature.startswith("hostel_resident"):
            simple_name = "Hostel Resident"

        elif feature.startswith("class_type"):
            simple_name = "Class Type"

        elif feature.startswith("weather"):
            simple_name = "Weather"

        elif feature.startswith("study_hours"):
            simple_name = "Study Hours"

        elif feature.startswith("sleep_hours"):
            simple_name = "Sleep Hours"

        elif feature.startswith("travel_time_minutes"):
            simple_name = "Travel Time"

        else:
            simple_name = feature

        simple_names.append(
            simple_name
        )

    # --------------------------------------------------------
    # FEATURE IMPORTANCE DATAFRAME
    # --------------------------------------------------------

    feature_importance_df = pd.DataFrame({

        "Feature": simple_names,

        "Importance": importance
    })

    # Combine one-hot encoded features
    feature_importance_df = (
        feature_importance_df
        .groupby(
            "Feature",
            as_index=False
        )["Importance"]
        .sum()
    )

    # Sort by importance
    feature_importance_df = (
        feature_importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
    )

    # ========================================================
    # DATASET INFORMATION
    # ========================================================

    dataset_info = {

        "total_records": len(df),

        "train_records": len(X_train),

        "test_records": len(X_test),

        # Original 12 features
        "features_count": X.shape[1],

        "accuracy": acc,

        "report": report,

        "confusion_matrix": cm,

        "feature_importance":
            feature_importance_df
    }

    return pipeline, dataset_info


# ============================================================
# GET TRAINED MODEL
# ============================================================

model, info = get_trained_model()


# ============================================================
# USER INTERFACE
# ============================================================

st.title(
    "🎓 Student Attendance Predictor"
)

st.write(
    "Enter student details below to predict "
    "class attendance probability."
)


# ============================================================
# INPUT COLUMNS
# ============================================================

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    age = st.number_input(
        "Age",
        min_value=16,
        max_value=40,
        value=20
    )

    gender = st.selectbox(
        "Gender",
        [
            "male",
            "female"
        ]
    )

    course = st.selectbox(
        "Course",
        [
            "bca",
            "bsc",
            "bcom",
            "ba",
            "bba"
        ]
    )

    year = st.selectbox(
        "Year",
        [
            "1st year",
            "2nd year",
            "3rd year"
        ]
    )

    parent_education = st.selectbox(
        "Parent Education",
        [
            "school",
            "graduate",
            "postgraduate"
        ]
    )

    internet_access = st.selectbox(
        "Internet Access",
        [
            "yes",
            "no"
        ]
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    hostel_resident = st.selectbox(
        "Hostel Resident",
        [
            "yes",
            "no"
        ]
    )

    class_type = st.selectbox(
        "Class Type",
        [
            "offline",
            "online"
        ]
    )

    weather = st.selectbox(
        "Weather",
        [
            "sunny",
            "cloudy",
            "rainy"
        ]
    )

    study_hours = st.slider(
        "Study Hours / Day",
        0.0,
        10.0,
        4.0,
        0.1
    )

    sleep_hours = st.slider(
        "Sleep Hours / Day",
        3.0,
        10.0,
        7.0,
        0.1
    )

    travel_time_minutes = st.number_input(
        "Travel Time (minutes)",
        min_value=5,
        max_value=180,
        value=30
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

if st.button(
    "Predict Attendance",
    type="primary"
):

    # --------------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------------

    input_data = pd.DataFrame([
        {
            "age": age,

            "gender": gender,

            "course": course,

            "year": year,

            "parent_education":
                parent_education,

            "internet_access":
                internet_access,

            "hostel_resident":
                hostel_resident,

            "class_type":
                class_type,

            "weather":
                weather,

            "study_hours":
                study_hours,

            "sleep_hours":
                sleep_hours,

            "travel_time_minutes":
                travel_time_minutes
        }
    ])

    # --------------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------------

    pred = model.predict(
        input_data
    )[0]

    # --------------------------------------------------------
    # GET PROBABILITY
    # --------------------------------------------------------

    proba = model.predict_proba(
        input_data
    )[0]

    # ========================================================
    # RESULT
    # ========================================================

    st.markdown("---")

    st.subheader(
        "📌 Prediction Result"
    )

    # --------------------------------------------------------
    # HANDLE PREDICTION PROBABILITIES
    # --------------------------------------------------------

    classes = list(
        model.classes_
    )

    present_probability = 0.0
    absent_probability = 0.0

    # Normalize class names
    normalized_classes = [
        str(c).strip().lower()
        for c in classes
    ]

    # --------------------------------------------------------
    # CASE 1:
    # Present / Absent labels
    # --------------------------------------------------------

    if (
        "present" in normalized_classes
        and
        "absent" in normalized_classes
    ):

        for i, class_name in enumerate(classes):

            class_name = (
                str(class_name)
                .strip()
                .lower()
            )

            if class_name == "present":

                present_probability = (
                    proba[i]
                )

            elif class_name == "absent":

                absent_probability = (
                    proba[i]
                )

    # --------------------------------------------------------
    # CASE 2:
    # Numeric labels
    #
    # 0 = Absent
    # 1 = Present
    # --------------------------------------------------------

    elif (
        0 in classes
        and
        1 in classes
    ):

        absent_probability = (
            proba[
                classes.index(0)
            ]
        )

        present_probability = (
            proba[
                classes.index(1)
            ]
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    else:

        absent_probability = proba[0]

        present_probability = proba[1]

    # ========================================================
    # DISPLAY PREDICTION
    # ========================================================

    pred_normalized = (
        str(pred)
        .strip()
        .lower()
    )

    # Present prediction
    if (
        pred_normalized == "present"
        or
        pred == 1
    ):

        st.success(
            f"**Status: STUDENT IS LIKELY "
            f"TO BE PRESENT** "
            f"(Confidence: "
            f"{present_probability * 100:.1f}%)"
        )

    # Absent prediction
    else:

        st.error(
            f"**Status: STUDENT IS LIKELY "
            f"TO BE ABSENT** "
            f"(Confidence: "
            f"{absent_probability * 100:.1f}%)"
        )

    # ========================================================
    # PROBABILITY BAR
    # ========================================================

    st.markdown(
        "### Attendance Probability"
    )

    st.progress(
        float(present_probability)
    )

    st.caption(
        f"Absent Probability: "
        f"{absent_probability * 100:.1f}% | "
        f"Present Probability: "
        f"{present_probability * 100:.1f}%"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.markdown("---")

with st.expander(
    "📊 View Model & Training Details",
    expanded=False
):

    # ========================================================
    # MODEL OVERVIEW
    # ========================================================

    st.subheader(
        "Model Overview"
    )

    m_col1, m_col2, m_col3 = st.columns(3)

    m_col1.metric(
        "Model Architecture",
        "Random Forest"
    )

    m_col2.metric(
        "Test Accuracy",
        f"{info['accuracy'] * 100:.2f}%"
    )

    m_col3.metric(
        "Total Dataset Size",
        f"{info['total_records']} rows"
    )

    # ========================================================
    # HYPERPARAMETERS
    # ========================================================

    st.markdown(
        "#### Hyperparameters & Configuration"
    )

    st.write(
        "- **Estimators:** 300 Trees\n"
        "- **Max Depth:** None\n"
        "- **Minimum Samples Split:** 2\n"
        "- **Minimum Samples Leaf:** 1\n"
        "- **Max Features:** sqrt\n"
        "- **Train / Test Split:** 80% / 20%\n"
        "- **Features Count:** "
        f"{info['features_count']} predictors\n"
        "- **Encoding:** One-Hot Encoding"
    )

    # ========================================================
    # CLASSIFICATION METRICS
    # ========================================================

    st.markdown(
        "#### Detailed Classification Metrics"
    )

    df_report = pd.DataFrame(
        info["report"]
    ).transpose().round(2)

    st.dataframe(
        df_report,
        use_container_width=True
    )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        "#### 🔲 Confusion Matrix"
    )

    st.write(
        "The confusion matrix shows how correctly "
        "the model predicted Present and Absent students."
    )

    cm = info[
        "confusion_matrix"
    ]

    fig_cm, ax_cm = plt.subplots(
        figsize=(4, 3)
    )

    ax_cm.imshow(cm)

    ax_cm.set_xlabel(
        "Predicted"
    )

    ax_cm.set_ylabel(
        "Actual"
    )

    ax_cm.set_title(
        "Confusion Matrix"
    )

    # Display values inside matrix
    for i in range(
        cm.shape[0]
    ):

        for j in range(
            cm.shape[1]
        ):

            ax_cm.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center"
            )

    # Use actual model class names
    ax_cm.set_xticks(
        range(
            len(model.classes_)
        )
    )

    ax_cm.set_yticks(
        range(
            len(model.classes_)
        )
    )

    ax_cm.set_xticklabels(
        model.classes_
    )

    ax_cm.set_yticklabels(
        model.classes_
    )

    plt.tight_layout()

    st.pyplot(
        fig_cm
    )

    # ========================================================
    # FEATURE IMPORTANCE
    # ========================================================

    st.markdown(
        "#### 🌳 Random Forest Feature Importance"
    )

    st.write(
        "This graph shows how important each "
        "student-related feature is for predicting attendance."
    )

    feature_df = info[
        "feature_importance"
    ].copy()

    # --------------------------------------------------------
    # CREATE FEATURE IMPORTANCE GRAPH
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    ax.barh(
        feature_df["Feature"][::-1],
        feature_df["Importance"][::-1]
    )

    ax.set_xlabel(
        "Feature Importance"
    )

    ax.set_ylabel(
        "Feature"
    )

    ax.set_title(
        "Random Forest Feature Importance"
    )

    plt.tight_layout()

    st.pyplot(
        fig
    )

    # ========================================================
    # FEATURE IMPORTANCE TABLE
    # ========================================================

    st.markdown(
        "#### Feature Importance Values"
    )

    display_df = feature_df.copy()

    display_df["Importance"] = (
        display_df["Importance"]
        .round(4)
    )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

