# Automated-Metadata-Generation-with-Llama2-FAISS-and-RAG-Techniques

utils.py : contains all the functions required for creating vector db, Q/A, model and pipeline creation

main.py : main code to execute the pipeline with questions as user input

configs.py: variable/configuration storage

streamlit_main.py : streamlit ui code to upload a file and get answer to a single question

requirements.txt : all the required libraries with respective versions

Approach to Metadata Generation

1. Data Extraction

The first step involves extracting data from the project documentation and code files to create a comprehensive dataset. This process handles various file formats, such as text files, PDFs, and code files.

Libraries Used:

Textract: Extracts text from PDFs.
Tika: Parses different document types like Word files, HTML, and more.
Langchain: Orchestrates parsing operations across multiple formats, unifying extracted data into a large textual dataset.

2. Embedding and Vectorization
   
To process the extracted text efficiently, it needs to be converted into a numerical form through embeddings.

HuggingFace Transformers: This library is used to embed the textual dataset, converting the large text into numerical vectors that represent semantic information.
Llama2 (13B Quantized): Llama2 is used in a quantized form, which allows efficient use of resources (both memory and computation) without compromising model performance.

3. FAISS Vector Database
   
Once the data is embedded into vectors, it’s stored in a FAISS database for quick retrieval during metadata generation.

Vector Storage: FAISS (Facebook AI Similarity Search) stores embeddings in a highly efficient manner, allowing for fast, approximate nearest neighbor search.
Structure: The database is structured to hold and index metadata and project-related information for rapid access during query tasks.

4. Prompt Engineering
   
To control and guide Llama2 in generating accurate metadata, prompt engineering is crucial. The aim is to give the model clear instructions on how to interpret and respond to the input data.

Template Creation: A carefully designed prompt template helps ensure the model retrieves and generates context-aware metadata, minimizing errors (e.g., hallucinations, irrelevant data).
Example: A prompt may look like: "Given the following code file and project documentation, generate metadata including author, file format, key concepts, and implementation details."

5. Question-Answering Pipeline
The metadata generation process is closely linked to a question-answering (QA) pipeline, where the system uses Llama2 to answer queries about the project’s metadata.

Components:
Llama2 Model and Tokenizer: Loaded into memory along with the vector database for answering user queries.
RAG (Retrieval-Augmented Generation): RAG enables the model to retrieve relevant information from FAISS before generating answers, combining search with generation to improve the quality of responses.

6. Streamlit Interface
A user-friendly graphical interface is built using Streamlit, allowing users to interact with the system without writing code.

Streamlit Features:
Simple input fields for uploading documentation or code files.
Button triggers to initiate metadata generation.
Display fields to show generated metadata and allow users to download it.

How It Works: Process Flow

File Upload: Users upload project documentation and code files (PDFs, text files, etc.) through the Streamlit interface.

Data Parsing: Textract and Tika parse the uploaded files, and Langchain orchestrates the integration of the parsed content into a unified dataset.

Embedding Generation: HuggingFace Transformers converts the dataset into embeddings using Llama2.

FAISS Storage: The vector database stores the embeddings for quick retrieval during the metadata generation process.

Prompt Engineering: The system uses pre-defined templates to craft input prompts, guiding Llama2 to generate the required metadata.

Metadata Generation: The system retrieves relevant data from FAISS using RAG and produces the metadata.

User Interaction: Users can ask specific questions about the dataset (e.g., "What’s the author’s name?") via the Streamlit interface, and the system retrieves the information through the QA pipeline.
