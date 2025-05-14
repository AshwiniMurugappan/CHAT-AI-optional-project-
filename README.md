# CHAT-AI-optional-project-

</div>

## 📈 1. Introduction and Project Goal:

- To develop an intelligent chatbot capable of answering frequently asked questions (FAQs) from bank customers.
- To build a conversational AI agent that can assist customers with basic banking details, such as different account openings, policies for the customer, few important abbreviations for customers.
- To create a chatbot that improves customer service efficiency and provides instant support for common inquiries.

## 🎯 2. Methodology and Approach:

### Data Collection:
- Used a PDF file which has all the important details about banking for its customers. 
- The Data consist of 4 Pages with details about different bank account details, important Banking Services, Loan Products, Interest rate Charges and Customer Related Queries.

### Tools and Technologies:

The following Python libraries were utilized in the development of the Bank Customer Chatbot:

* **streamlit:** Used for creating the interactive web interface for the chatbot, allowing users to easily interact with the conversational AI.
* **langchain:** A framework for building applications powered by large language models. It was instrumental in orchestrating the different components of the chatbot, including embeddings, language models, vector stores, prompts, retrieval chains, and memory management.
* **HuggingFaceEmbeddings:** Employed to generate vector embeddings of the text data (e.g., banking documents, FAQs). These embeddings capture the semantic meaning of the text, enabling efficient similarity search.
* **CTransformers:** Facilitated the use of quantized large language models (LLMs) for efficient inference. This library allows running LLMs on consumer hardware with reduced memory and computational requirements. You would likely have used this to load and run your Llama model or another compatible LLM.
* **FAISS (Facebook AI Similarity Search):** Used to build and maintain the vector store. FAISS provides efficient indexing and searching of the embeddings, allowing the chatbot to quickly retrieve relevant information based on the user's query.
* **PromptTemplate:** Utilized to create structured prompts that guide the language model to generate appropriate and informative responses based on the retrieved information and conversation history.
* **ConversationalRetrievalChain:** A specific chain provided by LangChain that combines a retrieval mechanism (using the vector store) with a language model to have conversational interactions grounded in the retrieved documents. It handles both retrieving relevant context and generating natural language responses.

### Core Chatbot Logic (Python with LangChain and Llama)

The backend logic of the Bank Customer Support Chatbot is implemented using Python and the LangChain framework, leveraging a Llama language model for intelligent responses. The following key components and functions are involved:

**1. Loading the Language Model (`load_llm()`):**

* This function, decorated with `@st.cache_resource`, ensures that the language model is loaded only once when the Streamlit application starts, optimizing performance.
* It utilizes the `CTransformers` library to load a quantized version of the **Mistral 7B Instruct v0.1** model and stored locally. `CTransformers` enables efficient inference on consumer hardware.
* The model is configured with parameters such as `max_new_tokens` (limiting the length of the generated response) and `temperature` (controlling the randomness of the output, set to a low value of 0.01 for more deterministic and factual answers).
* Error handling is included to gracefully manage potential issues during model loading.

**2. Setting up the Conversational QA Chain (`setup_conversational_qa()`):**

* This function, also cached using `@st.cache_resource(show_spinner=False)`, sets up the core conversational retrieval chain using LangChain. This chain orchestrates the process of retrieving relevant information and generating context-aware responses.
* **Prompt Template:** A `PromptTemplate` is defined to structure the input to the language model. It instructs the model to act as a helpful bank chatbot, use the provided context to answer the user's question, and truthfully say "I don't know" if the context is insufficient.
* **Memory:** A `ConversationBufferMemory` is initialized to store the history of the conversation, allowing the chatbot to maintain context over multiple turns.
* **Retriever:** The `as_retriever()` method is used on the pre-loaded `vector_store` (created using `FAISS` and `HuggingFaceEmbeddings`) to create a retriever. This component fetches the most relevant documents (in this case, `k=2`) from the vector store based on the user's query.
* **ConversationalRetrievalChain:** The core chain is created using `ConversationalRetrievalChain.from_llm()`. It integrates:
    * The loaded language model (`_llm`).
    * The retriever (`retriever`) to fetch relevant context.
    * The conversation memory (`memory`) to track chat history.
    * The custom prompt (`PROMPT`) to guide the language model's response generation.

**3. Main Streamlit Application (`main()`):**

* Sets the title of the Streamlit application.
* **Loading Vector Store:** Loads the pre-existing FAISS vector store from the local directory, which contains the embedded banking knowledge. `HuggingFaceEmbeddings` is used to ensure compatibility with the vector store.
* **Loading LLM and Setting up QA Chain:** Calls the `load_llm()` and `setup_conversational_qa()` functions to initialize the language model and the conversational retrieval chain.
* **Chat History Management:** Initializes an empty list `messages` in `st.session_state` to store the conversation history. It then displays existing messages on each rerun of the app.
* **User Input Handling:**
    * Uses `st.chat_input()` to get user input.
    * Implements logic to recognize and respond to phrases indicating the end of the chat.
    * Appends the user's message to the chat history and displays it.
* **Generating and Displaying Assistant Response:**
    * Uses `st.chat_message("assistant")` to create a container for the chatbot's response.
    * Displays a "Thinking..." spinner while the response is being generated.
    * Calls the `qa_chain` with the user's question to get the answer.
    * Displays the generated answer using `st.markdown()`.
    * Appends the assistant's response to the chat history.

**In essence, this code demonstrates a Retrieval-Augmented Generation (RAG) approach. The chatbot first retrieves relevant information from the vector store based on the user's query and then uses the Llama language model, guided by a specific prompt and the conversation history, to generate a contextually appropriate and informative answer.**

**Key Libraries in Action:**

* **streamlit:** Provides the user interface for interacting with the chatbot.
* **langchain:** Acts as the orchestration layer, connecting the language model, vector store, prompt, and memory.
* **CTransformers:** Enables the use of the powerful Llama model efficiently.
* **FAISS:** Provides fast similarity search over the embedded banking knowledge.
* **HuggingFaceEmbeddings:** Creates the semantic embeddings for effective information retrieval.
* **PromptTemplate:** Structures the instructions given to the language model.
* **ConversationBufferMemory:** Manages the conversation history for contextual awareness.

## 🛠️ 3. Focus on Deployment

* For the purpose of demonstrating the Streamlit application, it was necessary to expose it to the public internet. This was achieved by installing and utilizing localtunnel within the Google Colaboratory environment.
* To deploy the Streamlit application from within the Google Colaboratory environment and make it accessible over the internet, the following steps were executed in a single command.
* This command starts the Streamlit server, exposes it to the internet via localtunnel, and prints the external IP address.

## 🧩 User Interaction

* Once the Streamlit application was successfully deployed and exposed to the internet via localtunnel, the provided URL was used to interact with the Bank Customer Support Chatbot. Users could access the chatbot through this URL using any web browser.

* The chatbot interface, as described in the "User Interface" section, allowed users to type their banking-related questions and receive instant responses. The conversational flow, powered by the Llama language model and LangChain, enabled users to ask follow-up questions and engage in a natural dialogue.


