from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader


template = """Question: {question}

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["question"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="llama-2-7b-chat.ggmlv3.q8_0.bin",
    n_ctx=6000,
    n_gpu_layers=512,
    n_batch=30,
    callback_manager=callback_manager,
    temperature = 0.9,
    max_tokens = 4095,
    n_parts=1,

)


llm_chain = LLMChain(prompt=prompt, llm=llm)

# embedding engine
hf_embedding = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# load from local
db = FAISS.load_local("faiss_Ebook/", embeddings=hf_embedding)

query = "What is the definion of EDA"
search = db.similarity_search(query, k=2)

template = '''Context: {context}

Based on Context provide me answer for following question
Question: {question}

Tell me the information about the fact. The answer should be from context only
do not use general knowledge to answer the query'''

prompt = PromptTemplate(input_variables=["context", "question"], template= template)
final_prompt = prompt.format(question=query, context=search)
llm_chain.run(final_prompt)

