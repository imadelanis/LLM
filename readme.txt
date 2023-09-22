#download LLAMA2 Model or any other Model
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main 

#Download any PDF file. Example PDF related to EDA and renmame it to ebook.pdf
https://ddd.uab.cat/pub/tesis/2017/hdl_10803_457967/mjllr1de1.pdf

#install python
#install langchain[all]
#install chainlit

#init.py
#create the vector store from the provided PDF file 

#prompt.py
#test the LLM model with prompt. Change query to change the question 

#web.py
#start a web application like ChatGPT on port 8000 
#start chainlit
chmod +x web.py
chainlit run web.py&