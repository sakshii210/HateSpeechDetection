#from copyreg import pickle
import streamlit as st
import pickle
#import numpy as np
#from sklearn.tree import DecisionTreeClassifier

# Set page configuration
st.set_page_config(page_title="Hate Speech Detection", layout="wide")
#st.title("Welcome to Aditi Srivastava's Webpage")
# Define the main title and subtitle
#st.subheader("Author: Aditi Srivastava")
st.title("Hate Speech Detection")
st.markdown(
    """
    </style>
    <div style='text-align: center;'><em>Enter any speech for detection</em></div>
    """
    , unsafe_allow_html=True
)

# Create input text box for speech
speech = st.text_input( '')

#check if speech is provided
if st.button("check for hate and offensive language"):
    if not speech:
        st.error("Please enter a speech for detection.")
    else:
        # Load the pre-trained model and feature vectorizer
        model = pickle.load(open('hate_speech_model.pkl',"rb"))
        cv = pickle.load(open("cv.pkl","rb"))

        # Preprocess the input speech
        data = cv.transform([speech]).toarray()

        # Make predictions
        pred = model.predict(data)

        if pred == "Offensive Language":
            st.error("Offensive Language Detected")
            st.image("offensive_image.png", caption="Be mindful when it comes to your words. A string of some that do not mean much to you, may stick with someone else for a lifetime.", width=300)
            st.info("This speech contains offensive language. Please be cautious when using such language.")
            # st.balloons()
            st.markdown("---")
            st.subheader("Additional Information")
            st.write("Prediction Probability: ", model.predict_proba(data)[0, 1])
            st.write("Decision Path: ", model.decision_path(data))
            st.write("Feature Importance: ", model.feature_importances_)
        elif pred == "No Hate and Offensive language":
            st.success("No Hate and Offensive Language was detected")
            st.image("non_hate_image.png", caption="Kind words are a creative force, a power that concurs in the building up of all that is good, and energy that showers blessings upon the world.", width=300)
            st.success("This speech does not contain hate or offensive language. Continue spreading positivity!")
            st.markdown("---")
            st.subheader("Additional Information")
            st.write("Prediction Probability: ", model.predict_proba(data)[0, 1])
            st.write("Decision Path: ", model.decision_path(data))
            st.write("Feature Importance: ", model.feature_importances_)
        elif pred == "Hate Speech":
            st.error("Hate Speech Detected")
            st.image("hate_speech_image.png", caption="Be careful with your words. Once they are said, they can be only forgiven, not forgotten.", width=300)
            st.error("This speech contains hate speech. Please promote respect and tolerance in communication.")
            st.markdown("---")
            st.subheader("Additional Information")
            st.write("Prediction Probability: ", model.predict_proba(data)[0, 1])
            st.write("Decision Path: ", model.decision_path(data))
            st.write("Feature Importance: ", model.feature_importances_)
        else:
            st.warning("Not Found")

# Add a separator for better visual division
st.markdown("---")

# Add a section for example speeches
st.subheader("Example Speeches")
example_speeches = [
    "I hate you!",
    "This is a friendly message.",
    "You're the worst!",
    "He is a stupid boy!",
    "I completely disagree with you.",
]

selected_example = st.selectbox("Select an example speech", example_speeches)

if st.button("Check Example Speech"):
    st.text_area("Selected Example Speech", selected_example, height=100)
    # Load the pre-trained model and feature vectorizer
    model = pickle.load(open("hate_speech_model.pkl", "rb"))
    cv = pickle.load(open("cv.pkl", "rb"))

    # Preprocess the example speech
    data = cv.transform([selected_example]).toarray()

    # Make predictions
    pred = model.predict(data)

    # Display the prediction result
    if pred == "Offensive Language":
        st.error("Offensive Language Detected")
    elif pred == "No Hate and Offensive language":
        st.success("No Hate and Offensive Language was detected")
    elif pred == "Hate Speech":
        st.error("Hate Speech Detected")
    else:
        st.warning("Not Found")

# Chatbot-style interaction
st.subheader("Chatbot")
st.markdown(
    """
    This chatbot can assist you with hate speech detection. Type your question or choose an option below:
    """
)

st.markdown("---")

selected_option = st.selectbox("Select an option", ["Select One", "What is hate speech?", "How does the model detect hate speech?", "What can I do to combat hate speech?"])

if selected_option == "Select One":
    st.markdown("Please select an option from the dropdown.")
elif selected_option == "What is hate speech?":
    st.markdown("Hate speech refers to any form of communication, whether written, spoken, or symbolic, that offends, threatens, or insults individuals or groups based on attributes such as race, religion, ethnic origin, sexual orientation, disability, or gender.")
    st.markdown("It is important to be aware of the impact of hate speech and to promote respectful and inclusive communication.")

elif selected_option == "How does the model detect hate speech?":
    st.markdown("The model uses machine learning techniques to analyze the text of a speech and classify it into categories such as offensive language, hate speech, or no hate and offensive language. It is trained on a dataset of labeled examples to learn patterns and features that distinguish between different types of speech.")

elif selected_option == "What can I do to combat hate speech?":
    st.markdown("Here are some ways you can combat hate speech:")
    st.markdown("- Educate yourself about different cultures, religions, and communities to foster understanding and empathy.")
    st.markdown("- Promote respectful and inclusive language in your interactions.")
    st.markdown("- Speak up against hate speech when you encounter it.")
    st.markdown("- Support organizations and initiatives that work towards combating hate speech and promoting tolerance.")


m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: green ;
    padding: 13px 50px;
    display: block;
    margin: 0 auto;
}
</style>""", unsafe_allow_html=True)
n = st.markdown("""
<style>
div.stimage {
    size: 200px;
}
</style>""", unsafe_allow_html=True)

