# [-----------------IMPORTS-----------------]

#[OpenAI - LLM]
import openai
#[Pinecone - Vector database]
import pinecone
#[Langchain - Question answering system]
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores.pinecone import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains import LLMChain, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain import OpenAI
#[pypdf - PDF to text]
import pypdf
#[Streamlit - UI]
import streamlit as st
#[Configuration file] 
import config
#[Other tools]
import copy

# [-----------------FUNCTIONS-----------------]

def ui_spacer(n=2, line=False, next_n=0):
    """
    Add spaces & tabs 
    """
    for _ in range(n):
        st.write('')
    if line:
        st.tabs([' '])
    for _ in range(next_n):
        st.write('')


def ui_info():
    """
    Display the UI information
    """
    st.markdown(f"""
    # PDF Solver
    version {VERSION}

    Question answering system built on top of GPT 3.5 (ChatGPT)

    """)
    ui_spacer(1)
    st.write("Made by [Javier Chico García](https://www.linkedin.com/in/javier-chico-garcía-ofc/)")
 
    ui_spacer(1)
    st.markdown("""
		Please be aware that this is only a Proof of Concept system
		and may contain bugs or unfinished features.

        """)
    ui_spacer(1)
    st.markdown('Source code can be found [here](https://github.com).')


def ui_task():
    """
    Display the main prompt on the advanced settings
    """
    st.text_area('Main task', config.AI_TASK_TEMPLATE, key='prompt', height=200, disabled=True)
    st.text_area('Eval task', config.AI_EVAL_TEMPLATE, key='eval_prompt', height=200, disabled=True)


def ui_model_type():
    """
    Display the model type on the advanced settings
    """
    st.text_area('Model type', config.MODEL_TYPE, key='model')


def ui_chain_type():
    """
    Display the chain on the advanced settings
    """
    st.text_area('Chain type', config.CHAIN_TYPE, key='qa_chain')


def ui_docs_retrieved():
    """
    Display the number of docs retrieved on the advanced settings
    """
    st.text_area('Number of docs retrieved', config.NUM_SIMILAR_DOCS, key='num_docs')


def ui_api_key():
    """
    Display the input field to write down the Open API key
    """
    st.write('## OpenAI API key')

    if st.button('Get API key from text file'): 
        ss['api_key'] =  config.OPENAI_API_KEY

    st.text_input('OpenAI API key', type='password', key='api_key', label_visibility="collapsed")

    openai.api_key = ss.get('api_key')
    if ss.get('api_key'):
        st.write("API key uploaded!")


def ui_pdf_file():
    """
    Display the upload pdf system
    """
    st.write('## Upload your PDF file')
    # disabled = not ss.get('api_key')

    file_uploaded = st.session_state.file_uploaded
    
    if not file_uploaded:

        st.write('File not uploaded yet')
        file = st.file_uploader('pdf file', type='pdf', key='pdf_file', label_visibility="collapsed")

        if file is not None:

            with st.spinner('Converting PDF...'):  
                # PDF to text
                raw_docs = pdf_to_text()
                
                # Split text into chunks
                docs = create_chunks(raw_docs)

                st.session_state.docs = copy.deepcopy(docs)

                st.session_state.file_uploaded = True


def ui_question():
    """
    Display the input field for questions
    """
    st.write('## Ask questions'+(f' to {ss["filename"]}' if ss.get('filename') else ''))
    disabled = False
    st.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)


def pdf_to_text():
    """
    Convert the pdf file to raw text
    """
    uploaded_file = ss.get('pdf_file')

    reader = pypdf.PdfReader(uploaded_file)

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    return raw_text


def create_chunks(raw_docs):
    """
    Split the raw text into chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    texts = text_splitter.split_text(raw_docs)
    
    return texts


def get_chat_history(inputs) -> str:
    """
    Used to format the chat_history string.
    """
    res = []
    for human, ai in inputs:
        res.append(f"Human:{human}\nAI:{ai}")
    return "\n".join(res)


def execute_qa_chain(query, docsearch):
    """
    Execute the QA chain
    """
    # Prompt for generating questions
    prompt = PromptTemplate(
        template=ss.get('prompt'),
        input_variables=["question", "context"]
    )

    # Santized query
    sanitized_query = query.replace("\n", " ") # OpenAI recommends to remove newlines

    # Base model
    llm = OpenAI(openai_api_key=ss.get('api_key'), temperature=0, model_name=ss.get('model'))

    # Chain for generating questions
    question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)

    # Chain for answering questions
    doc_chain = load_qa_chain(llm=llm, chain_type=ss.get('qa_chain'), prompt=prompt)

    # Custom chain: Retrieval + Question Generation + Answering
    chain = ConversationalRetrievalChain(
        retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":int(ss.get('num_docs'))}),
        question_generator=question_generator,
        combine_docs_chain=doc_chain,
        return_source_documents=True,
        get_chat_history=get_chat_history
    )

    # Initialize chat history
    chat_history = []

    # Get the answer
    with get_openai_callback() as cb:

        response = chain({"question":sanitized_query, "chat_history":chat_history})['answer']

    # Show costs of the request in console
    print("Costs of the question request: \n", cb)

    if config.EVALUATION:
        # Prompt for evaluation
        eval_prompt = PromptTemplate(
            template=ss.get('eval_prompt'),
            input_variables=["question", "context", "response"]
        )

        # Evaluation chain
        eval_retriever=docsearch.as_retriever(search_type="similarity", search_kwargs={"k":int(ss.get('num_docs'))})
        docs_returned=eval_retriever.get_relevant_documents(sanitized_query)

        # Evaluation chain
        eval_doc_chain = LLMChain(llm=llm, prompt=eval_prompt)

        # Get the answer
        with get_openai_callback() as e_cb:
            eval_response = eval_doc_chain.run(question=sanitized_query, context=docs_returned, response=response)
        
        print("[----------------------EVALUATION----------------------]")

        print("Eval response: ", eval_response)

        # Show costs of the request in console
        print("Costs of the eval request: \n", e_cb)

    # Add the query and response to the chat history
    chat_history.append((sanitized_query, response))

    return response


def generate_namespace():
    """
    Generate a namespace for Pinecone based on the PDF name
    """

    pdf_name = ss.get('pdf_file').name

    return f'QA_{pdf_name[0:59]}'
    

def process(question):
    """
    Process the question and return the answer
    """
    # Get the documents indexed
    docs = copy.deepcopy(st.session_state.docs)

    with st.spinner('Loading embeddings and vector database...'):  
        # Set up OpenAI embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=ss.get('api_key'))

        # Set up Pinecone
        try:
            pinecone.init(
                api_key=config.PINECONE_API_KEY,
                environment=config.PINECONE_ENVIRONMENT
            )
        except pinecone.exceptions.PineconeException:
            st.markdown(f"## Connection to Pinecone failed. Please check your API key and environment.")
            return
        
    with st.spinner('Searching in vector database...'):
        # Generate namespace
        pinecone_namespace= generate_namespace() 

        # Check for relevant documents embedded in Pinecone
        docsearch = Pinecone.from_texts(docs, embeddings, index_name=config.PINECONE_INDEX_NAME, namespace=pinecone_namespace)

    # Execute QA chain
    with st.spinner('Preparing answer...'):  
        try:
            response = execute_qa_chain(question, docsearch)
   
        except openai.error.AuthenticationError:
            st.markdown(f"## :red[API key invalid]")
            return

    ss['output'] = response
    

def button_predict():
    """
    Display the button to predict the answer
    """
    disabled = not ss.get('api_key')
    question = ss.get('question','')
    if st.button('Get answer', type='primary', disabled=disabled, use_container_width=True):
        question = ss.get('question','')

        process(question)    


def ui_output(): 
    """
    Display the output response of the system
    """
    st.markdown(f"""
    ## Output
    """)
    output = ss.get('output','')
    st.markdown(output)


if __name__ == "__main__":

    # [PARAMETERS]
    APP_NAME = "PDF Solver"
    VERSION = "1.0"

    # [STREAMLIT CONFIG]
    st.set_page_config(layout='wide', page_title=f'{APP_NAME} {VERSION}')
    ss = st.session_state

    st.session_state.docs = []
    st.session_state.file_uploaded = False

    # [LAYOUT]
    with st.sidebar:
        ui_info()
        with st.expander('advanced settings'):
            ui_task()
            ui_model_type()
            ui_chain_type()
            ui_docs_retrieved()

    c1,c2 = st.columns([1,1])
    with c1:
        ui_api_key()
        ui_pdf_file()
        ui_question()
        button_predict()
    with c2:
        ui_output()