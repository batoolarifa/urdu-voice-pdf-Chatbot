# Urdu Voice PDF Chatbot

This project is an **Urdu Voice Chatbot** that allows users to upload a PDF (which can be in **English**) and chat with the PDF in **Urdu** via voice. The chatbot processes text by creating embeddings with **Hugging Face Embeddings** for understanding the PDF content **Google Gemini** is used as LLM, and  converts speech to text via **Streamlit's mic recorder**, and provides responses in Urdu through **Google Text-to-Speech (gTTS)**.


## Features

- **PDF Upload**: Users can upload a PDF document.
- **Urdu Voice Interaction**: Users can ask questions or make inquiries about the PDF content in **Urdu**, and receive answers in **Urdu voice**.
- **LangChain Integration**: The project uses **LangChain** to manage the interactions with the language model and query the PDF content effectively.
- **Hugging Face Embeddings**: Embeddings are generated using **Hugging Face models** to provide deep semantic understanding of the uploaded PDF.
- **Speech to Text**: User input in Urdu is converted to text using the **Streamlit mic recorder** for processing.
- **Text to Speech**: The bot responds with text that is converted to **Urdu voice** using **gTTS (Google Text-to-Speech)**.
- **Google Gemini Integration**: The **LLM (Gemini)** processes the PDF content and provides meaningful responses based on the queries.

## Tech Stack

- **Streamlit**: For building the UI and handling voice input through the mic recorder.
- **Google Gemini API**: The language model used for generating coherent relevant accurate responses.
- **gTTS (Google Text-to-Speech)**: For converting chatbot responses to Urdu voice output.
- **LangChain**: For efficient interaction with the language model and managing PDF queries.
- **Hugging Face Embeddings**: For generating embeddings of the PDF content to improve the chatbot's understanding.
- **PyPDFLoader**: For extracting text from the uploaded PDF files.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/urdu-voice-pdf-chatbot.git
   ```

2. Navigate to the project directory:
   ```bash
   cd urdu-voice-pdf-chatbot
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Add your Google Gemini API key in the `.env` file:
   ```env
   GEMINI_API_KEY=your-api-key-here
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Upload a PDF document (English).
3. Use the microphone to ask questions in **Urdu**.
4. The chatbot will respond in **Urdu voice**, explaining the content or answering questions based on the PDF.

**Author**: Arifa Batool 

**Live Application**: https://urdu-voice-pdf-chatbot.streamlit.app/