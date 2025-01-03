import streamlit as st
import re
import pandas as pd
import joblib

# Load model đã train
with open(r"MODEL/naive_bayes_model1.joblib", "rb") as file:
    model = joblib.load(file)

# Tiêu đề ứng dụng
st.title("Spam Email Detection")
st.write("Enter an email below or upload a file to check if it's spam or not.")

# Hàm đọc danh sách từ từ file
def read_words_from_file(words_file='words.txt'):
    with open(words_file, 'r') as file:
        words = [line.strip() for line in file.readlines()]
    return words

# Hàm đếm số lần xuất hiện của từ trong email
def count_words_in_email(email, words_file='words.txt'):
    words = read_words_from_file(words_file)
    email_words = email.lower().split()
    word_counts = {word: email_words.count(word) for word in words}
    return word_counts

# Chuyển kết quả đếm từ thành DataFrame phù hợp với mô hình
def prepare_features(email, words_file='words.txt'):
    word_counts = count_words_in_email(email, words_file)
    feature_df = pd.DataFrame([word_counts])
    return feature_df

# Input từ người dùng
email_input = st.text_area("Email Content", placeholder="Type your email here...")

# Upload file
uploaded_file = st.file_uploader("Upload a file with email content (TXT)", type=["txt"])

# Xử lý email từ file
if uploaded_file is not None:
    content = uploaded_file.read().decode("utf-8")
    st.text_area("Uploaded Email Content", content, height=200)
    email_input = content

# Kiểm tra email
if st.button("Check Spam"):
    if email_input.strip() == "":
        st.warning("Please provide an email content either by typing or uploading a file!")
    else:
        # Xử lý email input
        feature_data = prepare_features(email_input)
        
        # Dự đoán kết quả
        prediction = model.predict(feature_data)
        
        # Hiển thị kết quả
        if prediction[0] == 1:
            st.error("This email is classified as SPAM.")
        else:
            st.success("This email is NOT SPAM.")
