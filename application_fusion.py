import streamlit as st
import pandas as pd
import joblib
import imaplib
import email
from email.header import decode_header

# Load trained model
def load_model(model_path):
    """Load the trained spam detection model."""
    with open(model_path, "rb") as file:
        return joblib.load(file)

# Function to read word list from file
def read_words_from_file(words_file):
    """Read the list of words from a file."""
    with open(words_file, 'r', encoding='utf-8') as file:
        return [line.strip() for line in file.readlines()]

# Count occurrences of words in email content
def count_words_in_email(email_content, words_file):
    """Count word occurrences in email content."""
    words = read_words_from_file(words_file)
    email_words = email_content.lower().split()
    return {word: email_words.count(word) for word in words}

# Prepare features for the model
def prepare_features(email_content, words_file):
    """Prepare feature DataFrame for prediction."""
    word_counts = count_words_in_email(email_content, words_file)
    return pd.DataFrame([word_counts])

# Language detection function
def is_vietnamese_email(email_content):
    """Determine if the email is likely in Vietnamese."""
    vietnamese_chars = "ăâêôơưđ"
    return any(char in email_content for char in vietnamese_chars)

# Function to fetch emails using IMAP
def get_emails(username, password, n=5):
    """Fetch the latest emails using IMAP."""
    emails = []
    try:
        mail = imaplib.IMAP4_SSL("imap.gmail.com")
        mail.login(username, password)
        mail.select("inbox")
        _, messages = mail.search(None, "ALL")
        message_ids = messages[0].split()[-n:]

        for msg_id in message_ids:
            _, msg_data = mail.fetch(msg_id, "(RFC822)")
            for response_part in msg_data:
                if isinstance(response_part, tuple):
                    msg = email.message_from_bytes(response_part[1])
                    subject = decode_header(msg["Subject"])[0][0]
                    subject = subject.decode() if isinstance(subject, bytes) else subject
                    date = msg["Date"]

                    # Decode email content
                    content = ""
                    if msg.is_multipart():
                        for part in msg.walk():
                            if part.get_content_type() == "text/plain":
                                content = part.get_payload(decode=True).decode('utf-8', 'ignore')
                                break
                    else:
                        content = msg.get_payload(decode=True).decode('utf-8', 'ignore')

                    emails.append({"subject": subject, "content": content, "time": date})
        mail.logout()
    except Exception as e:
        st.error(f"Error accessing Gmail: {e}")
    return emails

# Load both English and Vietnamese models
english_model = load_model(r"MODEL/naive_bayes_model1.joblib")
vietnamese_model = load_model(r"MODEL/naive_bayes_model2.joblib")

# Streamlit App Title
st.title("Spam Email Detection for English and Vietnamese")

# Sidebar for functionality selection
st.sidebar.title("Choose Functionality")
option = st.sidebar.radio("Select an option:", ["Gmail Fetch & Check", "Manual Input/File Upload"])

# Gmail Fetch & Check functionality
if option == "Gmail Fetch & Check":
    st.sidebar.title("Gmail Login")
    username = st.sidebar.text_input("Gmail Username", placeholder="your_email@gmail.com", value='nguyendinhphukhmt@gmail.com') 
    password = st.sidebar.text_input("Gmail Password", type="password", value='fmol wqbi bigc mfub')  
    fetched_emails = []
    
    if st.sidebar.button("Fetch Emails"):
        if username and password:
            st.info("Fetching emails...")
            fetched_emails = get_emails(username, password, n=50)
            if fetched_emails:
                st.success("Fetched emails successfully!")
            else:
                st.warning("No emails found or failed to fetch emails.")
        else:
            st.warning("Please enter Gmail username and password.")

    if fetched_emails:
        email_data_list = []
        for email_data in fetched_emails:
            try:
                # Determine language and choose model
                if is_vietnamese_email(email_data["content"]):
                    words_file = 'vietnamese_words.txt'
                    model = vietnamese_model
                else:
                    words_file = 'words.txt'
                    model = english_model
                
                feature_data = prepare_features(email_data["content"], words_file)
                prediction = model.predict(feature_data)
                label = "SPAM" if prediction[0] == 1 else "NOT SPAM"
                
                email_data_list.append({
                    "Subject": email_data["subject"],
                    "Content": email_data["content"],
                    "Time": email_data["time"],
                    "Label": label
                })
            except Exception as e:
                email_data_list.append({
                    "Subject": email_data["subject"],
                    "Content": email_data["content"],
                    "Time": email_data["time"],
                    "Label": f"Error: {e}"
                })

        email_df = pd.DataFrame(email_data_list)
        email_df["Time"] = pd.to_datetime(email_df["Time"], errors='coerce')
        email_df = email_df.sort_values(by="Time", ascending=False)
        st.write("### Email Table")
        st.dataframe(email_df)

# Manual Input/File Upload functionality
elif option == "Manual Input/File Upload":
    st.write("Enter an email below or upload a file to check if it's spam or not.")
    email_input = st.text_area("Email Content", placeholder="Type your email here...")
    uploaded_file = st.file_uploader("Upload a file with email content (TXT)", type=["txt"])

    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8", "ignore")
        st.text_area("Uploaded Email Content", content, height=200)
        email_input = content

    if st.button("Check Spam"):
        if email_input.strip() == "":
            st.warning("Please provide an email content either by typing or uploading a file!")
        else:
            try:
                # Determine language and choose model
                if is_vietnamese_email(email_input):
                    words_file = 'vietnamese_words.txt'
                    model = vietnamese_model
                else:
                    words_file = 'words.txt'
                    model = english_model
                
                feature_data = prepare_features(email_input, words_file)
                prediction = model.predict(feature_data)
                if prediction[0] == 1:
                    st.error("This email is classified as SPAM.")
                else:
                    st.success("This email is NOT SPAM.")
            except Exception as e:
                st.error(f"An error occurred during spam detection: {e}")
