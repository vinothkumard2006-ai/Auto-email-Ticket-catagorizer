import streamlit as st
import joblib


# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Auto Ticket Categorizer",
    page_icon="📧",
    layout="centered"
)


# -----------------------------------
# Load Saved Model and Vectorizer
# -----------------------------------

@st.cache_resource
def load_model():

    model = joblib.load("ticket_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")

    return model, vectorizer


model, vectorizer = load_model()


# -----------------------------------
# Priority Detection Function
# -----------------------------------

def detect_priority(ticket):

    urgent_keywords = [
        "urgent",
        "emergency",
        "immediately",
        "as soon as possible",
        "critical",
        "not working",
        "failed",
        "blocked",
        "unable",
        "cannot"
    ]

    ticket_lower = ticket.lower()

    for keyword in urgent_keywords:

        if keyword in ticket_lower:
            return "Urgent"

    return "Normal"


# -----------------------------------
# Application Title
# -----------------------------------

st.title("📧 Auto Email / Ticket Categorizer")

st.write(
    "Enter a customer support ticket and the system "
    "will automatically categorize it."
)


# -----------------------------------
# User Input
# -----------------------------------

ticket = st.text_area(
    "Enter Support Ticket",
    placeholder="Example: My payment failed while purchasing a product",
    height=150
)


# -----------------------------------
# Predict Button
# -----------------------------------

if st.button("🔍 Predict Category"):

    if ticket.strip() == "":

        st.warning("Please enter a support ticket.")

    else:

        # Convert text into TF-IDF
        ticket_tfidf = vectorizer.transform([ticket])

        # Predict category
        prediction = model.predict(ticket_tfidf)[0]

        # Get probabilities
        probabilities = model.predict_proba(ticket_tfidf)[0]

        # Calculate confidence
        confidence = max(probabilities) * 100

        # Detect priority
        priority = detect_priority(ticket)

        # Human review decision
        if confidence < 60:

            status = "⚠️ Needs Human Review"

        else:

            status = "✅ Automatically Categorized"


        # -----------------------------------
        # Display Results
        # -----------------------------------

        st.subheader("Prediction Results")

        st.info(
            f"📂 Predicted Category: {prediction}"
        )

        st.write(
            f"🎯 Confidence: {confidence:.2f}%"
        )

        st.write(
            f"🚨 Priority: {priority}"
        )

        st.write(
            f"📌 Status: {status}"
        )


        # -----------------------------------
        # Confidence Progress Bar
        # -----------------------------------

        st.progress(
            min(int(confidence), 100)
        )