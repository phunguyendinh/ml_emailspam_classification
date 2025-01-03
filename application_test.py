import streamlit as st
import re
import pandas as pd
import joblib
import imaplib
import email
from email.header import decode_header

# Load trained model
with open(r"MODEL/naive_bayes_model1.joblib", "rb") as file:
    model = joblib.load(file)

# App title
st.title("Spam Email Detection")

# Sidebar for functionality selection
st.sidebar.title("Choose Functionality")
option = st.sidebar.radio("Select an option:", ["Gmail Fetch & Check", "Manual Input/File Upload"])

# Read word list from file
def read_words_from_file(words_file='words.txt'):
    with open(words_file, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    return words

# Count occurrences of words in email content
def count_words_in_email(email_content, words_file='words.txt'):
    words = read_words_from_file(words_file)
    email_words = email_content.lower().split()
    word_counts = {word: email_words.count(word) for word in words}
    return word_counts

# Prepare features for the model
def prepare_features(email_content, words_file='words.txt'):
    word_counts = count_words_in_email(email_content, words_file)
    feature_df = pd.DataFrame([word_counts])
    return feature_df

# Gmail Fetch & Check functionality
if option == "Gmail Fetch & Check":
    # Sidebar for Gmail login
    st.sidebar.title("Gmail Login")
    username = st.sidebar.text_input("Gmail Username", placeholder="your_email@gmail.com")
    password = st.sidebar.text_input("Gmail Password", type="password")

    # Fetch emails using IMAP
    def get_emails(username, password, n=5):
        emails = []
        try:
            mail = imaplib.IMAP4_SSL("imap.gmail.com")
            mail.login(username, password)
            mail.select("inbox")

            # Get the latest n emails
            _, messages = mail.search(None, "ALL")
            message_ids = messages[0].split()[-n:]

            for msg_id in message_ids:
                _, msg_data = mail.fetch(msg_id, "(RFC822)")
                for response_part in msg_data:
                    if isinstance(response_part, tuple):
                        msg = email.message_from_bytes(response_part[1])
                        subject = decode_header(msg["Subject"])[0][0]
                        if isinstance(subject, bytes):
                            subject = subject.decode()
                        date = msg["Date"]
                        content = ""
                        if msg.is_multipart():
                            for part in msg.walk():
                                if part.get_content_type() == "text/plain":
                                    content = part.get_payload(decode=True).decode()
                                    break
                        else:
                            content = msg.get_payload(decode=True).decode()
                        emails.append({"subject": subject, "content": content, "time": date})
            mail.logout()
        except Exception as e:
            st.error(f"Error accessing Gmail: {e}")
        return emails

    # Fetch Emails
    fetched_emails = []
    if st.sidebar.button("Fetch Emails"):
        if username and password:
            st.info("Fetching emails...")
            fetched_emails = get_emails(username, password, n=10)  # Fetch 10 emails
            if fetched_emails:
                st.success("Fetched emails successfully!")
            else:
                st.warning("No emails found or failed to fetch emails.")
        else:
            st.warning("Please enter Gmail username and password.")

    # Display emails in a table with spam detection
    if fetched_emails:
        email_data_list = []
        for email_data in fetched_emails:
            try:
                feature_data = prepare_features(email_data["content"])
                prediction = model.predict(feature_data)
                label = "SPAM" if prediction[0] == 1 else "NOT SPAM"
                email_data_list.append({"Subject": email_data["subject"], "Content": email_data["content"], "Time": email_data["time"], "Label": label})
            except Exception as e:
                email_data_list.append({"Subject": email_data["subject"], "Content": email_data["content"], "Time": email_data["time"], "Label": f"Error: {e}"})

        email_df = pd.DataFrame(email_data_list)
        email_df["Time"] = pd.to_datetime(email_df["Time"], errors='coerce')
        email_df = email_df.sort_values(by="Time", ascending=False)

        st.write("### Email Table")
        st.dataframe(email_df)

# Manual Input/File Upload functionality
elif option == "Manual Input/File Upload":
    st.write("Enter an email below or upload a file to check if it's spam or not.")

    # Input from user
    email_input = st.text_area("Email Content", placeholder="Type your email here...")

    # Upload file
    uploaded_file = st.file_uploader("Upload a file with email content (TXT)", type=["txt"])

    # Process email from file
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        st.text_area("Uploaded Email Content", content, height=200)
        email_input = content

    # Check email for spam
    if st.button("Check Spam"):
        if email_input.strip() == "":
            st.warning("Please provide an email content either by typing or uploading a file!")
        else:
            try:
                feature_data = prepare_features(email_input)
                prediction = model.predict(feature_data)
                if prediction[0] == 1:
                    st.error("This email is classified as SPAM.")
                else:
                    st.success("This email is NOT SPAM.")
            except Exception as e:
                st.error(f"An error occurred during spam detection: {e}")
