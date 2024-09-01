from langchain_community.document_loaders import TextLoader  # type: ignore
from langchain.indexes import VectorstoreIndexCreator  # type: ignore
from langchain.text_splitter import CharacterTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings  # type: ignore
import os
from dotenv import load_dotenv  # type: ignore

# Load environment variables
load_dotenv()

# Initialize the LLM using Google Generative AI
llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))  # Fixed the env var name

try:
    loader = TextLoader("data.txt")
except Exception as e:
    print("Error while loading file:", e)

# Initialize embeddings with the correct model
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Set up the text splitter and index creator
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
index_creator = VectorstoreIndexCreator(
    embedding=embedding,
    text_splitter=text_splitter
)

# Create the index from the loader
index = index_creator.from_loaders([loader])

# Querying the LLM with user input
flag = True
while flag:
    human_message = input("How can I help you? ")
    
    # Stop the loop if the user types 'stop'
    if human_message.lower() == "stop":
        flag = False
        break

    # Query the index and print the response
    try:
        response = index.query(human_message, llm=llm)
        print(response)
    except Exception as e:
        print("Error querying the index:", e)
