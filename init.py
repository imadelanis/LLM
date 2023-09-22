from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader


#LOAD WORD DOCUMENT
#loader = UnstructuredFileLoader("AI discovers over 20K taxable French swimming pools.docx")
#documents = loader.load()
#text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
#docs = text_splitter.split_documents(documents)

#LOAD PDF
loader = UnstructuredPDFLoader("ebook.pdf")
data = loader.load()
print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

#SPLIT TEXT INTO CHUNKS
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)

print (f'Now you have {len(texts)} documents')

# embedding engine
hf_embedding = HuggingFaceInstructEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
db = FAISS.from_documents(texts, hf_embedding)
# save embeddings in local directory
db.save_local("faiss_Ebook")
