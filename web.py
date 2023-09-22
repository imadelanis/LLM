from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as ct
import asyncio

DB_FAISS_PATH = 'faiss_Ebook'
#custom_prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
#custom_prompt_template = """answer the question sent by the user if out of context reply I dont know the answer, please ask a relavent question.
custom_prompt_template = """answer the question sent by the user if out of context reply I dont know the answer, please ask a relavent question.
Context: {context}
Question: {question}

Response for Questions asked.
answer:
"""

def create_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context','question'])
    return prompt


#retreival Chain
def get_response_from_qa_chain (llm, prompt, db):
    chain_type_kwargs={'prompt': prompt}
    retreival_chain = RetrievalQA.from_chain_type(llm=llm, chain_type='stuff', retriever=db.as_retriever(search_kwargs={'k': 1}),return_source_documents=True,chain_type_kwargs=chain_type_kwargs)
    return retreival_chain

#Loading the local model into LLM
def load_llama2_llm ():
    # Load the model llama-2-7b-chat.ggmlv3.q8_0.bin that was downloaded locally 
    llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q8_0.bin", model_type="llama", max_new_tokens=512, temperature= 0.5)
    return llm


#answering bot creation
def answering_bot():
	embeddings = HuggingFaceEmbeddings (model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})
	vectorstore = FAISS.load_local(DB_FAISS_PATH, embeddings)
	llm = load_llama2_llm()
	message_prompt = create_prompt()
	response = get_response_from_qa_chain (llm, message_prompt, vectorstore)
	return response

#display the result of the question asked
def final_result(query):
	bot_result = answering_bot()
	bot_response = bot_result({'query': query})
	return bot_response

#chainlit code you can refer to the chainlit.io website for more details.
@ct.on_chat_start
async def start():
	chain = answering_bot()
	msg = ct.Message(content="The bot is getting initialized, please wait!!!!")
	await msg.send()
	msg.content = "Q&A bot is ready. Ask questions on the documents indexed?"
	await msg.update()
	ct.user_session.set("chain", chain)

@ct.on_message
async def main (message):
	chain = ct.user_session.get("chain")
	cb = ct.AsyncLangchainCallbackHandler(
		stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
	)
	cb.answer_reached = True
	res = await chain.acall(message, callbacks = [cb])
	answer = res["result"]
	await ct.Message(content=answer).send()


