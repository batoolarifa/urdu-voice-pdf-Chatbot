import streamlit as st
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from streamlit_mic_recorder import speech_to_text
from gtts import gTTS
import tempfile


# Load environment variables
load_dotenv()

headers = {
    "authorization":st.secrets['GOOGLE_API_KEY'],
    "content-type": "application/json",
 }
genai_api_key = genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def process_pdf_and_store_in_faiss(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=300)
    docs_chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(docs_chunks, embeddings)

    faiss_index_path = "faiss_index"
    vector_db.save_local(faiss_index_path)
    return vector_db, docs_chunks







def get_conversational_chain_urdu():
    
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """  You are interacting with a document. Use the following pieces of context to answer the user's question. 
                        Your response should be primarily in Urdu. 
                        Begin the response with a sentence in Urdu, followed by the answer in Urdu,
                        and end with another sentence in Urdu to conclude the answer.
                        Make sure the Urdu is clear and properly structured."""),
        ("assistant", "{context}"),
        ("user", "{question}"),
    ])

    
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.4)

    
    qa_chain = load_qa_chain(
        chain_type="stuff",
        llm=model,
        prompt=prompt_template
    )
    
    return qa_chain

def user_input_query(query, vector_db):
    
    relevant_documents = vector_db.similarity_search(query, k=5)
    chain = get_conversational_chain_urdu()

    inputs = {
        "input_documents": relevant_documents,  
        "question": query
    }

    response = chain(inputs)
    response_text = response['output_text']

    return response_text


def convert_text_to_audio(text, lang="ur"):  
    tts = gTTS(text=text, lang=lang)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio_file:
        temp_audio_file_path = temp_audio_file.name
        tts.save(temp_audio_file_path)
    return temp_audio_file_path



def main():
    st.markdown("<h1 style='text-align: center; color: white;'>Chat with your PDF 📄</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color:red;'>Upload your PDF 👇</h2>", unsafe_allow_html=True)

    # Custom CSS for button styling
    st.markdown("""
        <style>
        .custom-button {
            background-color: white;
            color: white;
            padding: 15px 30px;
            font-size: 18px;
            border: 2px solid #000;
            border-radius: 10px;
            cursor: pointer;
            width: 100%;
            display: block;
            text-align: center;
        }
        .custom-button:hover {
            background-color: #f0f0f0;
        }
        </style>
        """, unsafe_allow_html=True)

    # Upload PDF
    uploaded_file = st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
       
        if not os.path.exists('docs'):
            os.makedirs('docs')

        
        filepath = os.path.join("docs", uploaded_file.name)
        with open(filepath, "wb") as temp_file:
            temp_file.write(uploaded_file.read())

        if 'vector_db' not in st.session_state:
            with st.spinner("Just a moment, we're almost there... preparing your file! "):
                st.session_state.vector_db, st.session_state.docs_chunks = process_pdf_and_store_in_faiss(filepath)
            st.success('Embeddings created successfully!')

        

        
        st.markdown('<h2 style="text-align: center;">Record Your Question</h2>', unsafe_allow_html=True)
        text = speech_to_text(
            language='ur', 
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            just_once=False,
            use_container_width=True
)


        if text:
            
            response_text = user_input_query(text, st.session_state.vector_db)
            response_audio = convert_text_to_audio(response_text)

            
            st.markdown(f"**Response:** {response_text}")

            
            st.audio(response_audio)

if __name__ == "__main__":
    main()
