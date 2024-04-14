import re
import os
import requests
from bs4 import BeautifulSoup
import PyPDF2
import streamlit as st
import openai  
from random import choice
import pandas as pd
from PyPDF2 import PdfReader
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from googletrans import Translator
import nltk
nltk.download('punkt')




def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text += reader.pages[page_num].extract_text()
    return text



def extract_questions(text):
    # Define pattern to identify questions
    question_pattern = r"Q\s*No\s*\d+\s*(?:a\s*)?(?:A\s*)?(?:\([a-zA-Z\d]+\))?\s*.*?(?=(?:Q\s*No\s*\d+|$))"
    questions = re.findall(question_pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return questions

def search_askpython(keyword):
    url = f"https://www.askpython.com/?s={keyword}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        questions = soup.find_all('h2', class_='entry-title')

        results = []
        for question in questions:
            title = question.text.strip()
            link = question.find('a')['href']
            results.append({'title': title, 'link': link})

        return results
    except Exception as e:
        st.error(f"Error occurred while searching: {e}")
        st.text("Raw HTML response:")
        st.text(response.text)
        return []

def search_questions_in_pdf(pdf_files, selected_pdf, keyword):
    if not pdf_files:
        st.warning("No PDF files found in the directory. Exiting...")
        return

    try:
        pdf_path = os.path.join("C:\\Users\\SANIKA\\Downloads\\project\\TY\\TY\\Comp", selected_pdf)
        text = extract_text_from_pdf(pdf_path)

        if keyword.strip():
            st.write(f"Searching for '{keyword}' in '{selected_pdf}':")
            questions = extract_questions(text)
            relevant_questions = [question.strip() for question in questions if keyword.lower() in question.lower()]
            if relevant_questions:
                st.write(f"Questions related to '{keyword}':\n")
                for question in relevant_questions:
                    st.write(question.strip() + "\n")
            else:
                st.warning(f"No questions related to '{keyword}' found.")
        else:
            st.warning("No keyword entered. Exiting...")
    except ValueError:
        st.error("Invalid input. Please enter a number. Exiting...")




# Function to search questions on AskPython
def search_questions_on_askpython(keyword):
    url = f"https://www.askpython.com/?s={keyword}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        questions = soup.find_all('h2', class_='entry-title')

        results = []
        for question in questions:
            title = question.text.strip()
            link = question.find('a')['href']
            results.append({'title': title, 'link': link})

        return results
    except Exception as e:
        st.error(f"Error occurred while searching: {e}")
        st.text("Raw HTML response:")
        st.text(response.text)
        return []

# Function to read file
def read_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        return file.read()

# Function to tokenize text
def tokenize_text(text):
    return sent_tokenize(text)

# Function to detect language
def detect_language(text):
    translator = Translator()
    result = translator.detect(text)
    if result is not None:
        lang = result.lang
        return lang
    else:
        return None  


# Function to translate text to English
def translate_to_english(query, source_lang):
    translator = Translator()
    translated_query = translator.translate(query, src=source_lang, dest='en').text
    return translated_query

# Function to translate text to the specified language
def translate_to_language(sentences, target_lang):
    translator = Translator()
    translated_sentences = [translator.translate(sentence, src='en', dest=target_lang).text for sentence in sentences]
    return translated_sentences

# Function to respond to query
def respond_to_query(query, sentences, query_lang):
    query_tokens = nltk.word_tokenize(query.lower())

    vectorizer = TfidfVectorizer()
    vectorized_sentences = vectorizer.fit_transform(sentences)
    vectorized_query = vectorizer.transform([' '.join(query_tokens)])

    similarities = cosine_similarity(vectorized_query, vectorized_sentences).flatten()

    ranked_sentences = sorted(((similarity, sentence) for similarity, sentence in zip(similarities, sentences)), reverse=True)

    relevant_sentences = [sentence for similarity, sentence in ranked_sentences if similarity > 0.2]  # Adjust similarity threshold as needed

    translated_relevant_sentences = translate_to_language(relevant_sentences, query_lang)

    return translated_relevant_sentences



# ///////////////////////////////////////////////////////////







def count_keyword_occurrences_in_pdf(pdf_path, keyword):
    keyword_count = 0
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text = page.extract_text()
            questions = re.split(r'\n\s*\n', text)
            for question in questions:
                if keyword.lower() in question.lower():
                    keyword_count += 1
                    break  
    return keyword_count

def visualize_keyword_counts(selected_pdf, keyword_list):
    keyword_counts = {}
    st.subheader(f"Keyword Counts for {selected_pdf}")
    for keyword in keyword_list[selected_pdf]:
        keyword_count = count_keyword_occurrences_in_pdf(os.path.join("C:\\Users\\SANIKA\\Downloads\\project\\TY\\TY\\Comp", selected_pdf), keyword)
        keyword_counts[keyword] = keyword_count
    
    df = pd.DataFrame({"Keyword": list(keyword_counts.keys()), "Count": list(keyword_counts.values())})

    st.bar_chart(df.set_index("Keyword"), use_container_width=True)



def generate_questions(text):
    sentences = re.split(r'[.!?]', text)
    questions = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        subject, object_ = extract_subject_and_object(sentence)

        if subject:
            questions.append(f"What is {subject}?")
        if object_:
            questions.append(f"What is the {object_}?")
        if subject and object_:
            questions.append(f"What is the relationship between {subject} and {object_}?")

    return questions

def extract_subject_and_object(sentence):
    words = sentence.split()

    subject = None
    object_ = None

    for i, word in enumerate(words):
        if word.isalpha() and (word[0].isupper() or i == 0):
            subject = word
            break

    if subject:
        object_found = False
        for j in range(i + 1, len(words)):
            if words[j].isalpha() and (words[j][0].isupper() or words[j - 1] in ["is", "was", "are", "were", "has", "have", "had"]):
                object_ = words[j]
                object_found = True
                break
        if not object_found:
            for j in range(len(words) - 1, i, -1):
                if words[j].isalpha() and words[j][0].isupper():
                    object_ = words[j]
                    break

    return subject, object_


def main():
    st.title("PrepPioneer")
    choice = st.radio("Choose an option:", ["Search PYQs", "KnowledgeHunt","Inquiry Hub","Ask Chatbot","Generate More questions"])

    if choice == "Search PYQs":
        st.title("SearchQuest")
        pdf_files = [file for file in os.listdir("C:\\Users\\SANIKA\\Downloads\\project\\TY\\TY\\Comp") if file.endswith(".pdf")]
        pdf_keywords = {
            "Cloud_Computing.pdf": ["PaaS", "IaaS", "Cloud Computing", "Hypervisor", "Docker","KVM","Virtualization","Docker" ,"AWS","Big Data"],
            "DL.pdf": ["Deep Learning", "Neural Network", "CNN", "RNN"],
            "MPMC.pdf": ["Microprocessor", "Memory", "Cache", "Pipeline"],
            "SE.pdf": ["Agile", "Scrum", "Lean", "Waterfall","Agile" ,"Lean","Webapp","TDD","User story", "Elevator pitch","Product box"],
            "CN.pdf": ["Agile", "Scrum", "Lean", "Waterfall","Agile" ,"Lean","Webapp","TDD","User story", "Elevator pitch","Product box"],
            "DAA.pdf": ["Agile", "Scrum", "Lean", "Waterfall","Agile" ,"Lean","Webapp","TDD","User story", "Elevator pitch","Product box"],
            "SDA.pdf": ["Agile", "Scrum", "Lean", "Waterfall","Agile" ,"Lean","Webapp","TDD","User story", "Elevator pitch","Product box"],

        }
        selected_pdf = st.selectbox("Select a PDF file:", list(pdf_keywords.keys()))
        keyword = st.text_input("Enter a keyword to search for:")
        if st.button("Search"):
            search_questions_in_pdf(pdf_files, selected_pdf, keyword)
            
            
            
            visualize_keyword_counts(selected_pdf, pdf_keywords)

    
    elif choice == "KnowledgeHunt":
        keyword = st.text_input("Enter a keyword to search on AskPython:")
        if st.button("Search"):
            results = search_questions_on_askpython(keyword)
            if results:
                st.write("Here are some questions found on AskPython:")
                for idx, result in enumerate(results, start=1):
                    st.write(f"{idx}. {result['title']} - {result['link']}")
            else:
                st.warning("No questions found on AskPython.")

    elif choice == "Inquiry Hub":

        st.header("Community Forum")

        hashtable = {}

        st.header("Input Questions")
        question_count = 1
        while True:
            question = st.text_input(f"Enter question {question_count} (or leave blank to stop):")
            if question:
                hashtable[question] = None  
                question_count += 1
            else:
                break

        st.header("Answer Questions")
        if hashtable:
            selected_question = st.selectbox("Select a question to answer:", list(hashtable.keys()))
            answer = st.text_area("Enter the answer:")
            if st.button("Submit Answer"):
                if answer:
                    hashtable[selected_question] = answer
                    st.success("Answer submitted successfully!")
                else:
                    st.warning("Please enter an answer.")
        else:
            st.write("No questions saved yet.")

        # Display the contents of the hashtable
        st.header("Hashtable Contents")
        if hashtable:
            for question, answer in hashtable.items():
                st.write(f"Question: {question}")
                if answer:
                    st.write(f"Answer: {answer}")
                st.write("---")
        else:
            st.write("The hashtable is empty.")



    elif choice == "Ask Chatbot":  
        st.title("Ask chatty")

        filename = "C:\\Users\\SANIKA\\Downloads\\project\\TY\\TY\\Chat\\Chat.txt"

        text = read_file(filename)

        sentences = tokenize_text(text)

        st.write("Welcome to Chatbot!")
        st.write("In doubt? I am here to solve your doubts")
        st.write("Type 'exit' to end the conversation.")

        while True:
            query = st.text_input("Your query:", key="query_input")
        
            if query.lower() == 'exit':
                st.write("Goodbye!")
                break

            query_lang = detect_language(query)

            translated_query = translate_to_english(query, query_lang)

            relevant_sentences = respond_to_query(translated_query, sentences, query_lang)

            if relevant_sentences:
                st.write("\n".join(relevant_sentences))
            else:
                st.write("Sorry, I couldn't find relevant information for your query.")

    elif choice == "Generate More questions":
        st.title("Question Generation")
        text_input = st.text_area("Enter the text:", height=200)
        if st.button("Generate Questions"):
            questions = generate_questions(text_input)
            if questions:
                st.write("Generated Questions:")
                for i, question in enumerate(questions, start=1):
                    st.write(f"{i}. {question}")
            else:
                st.write("No questions generated. Please provide some text.")

    

if __name__ == "__main__":
    main()
