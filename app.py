from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import asyncio

app = Flask(__name__, template_folder="templates", static_url_path='/static')

file_path = "About_Aliksm.pdf"

loader = PyPDFLoader(file_path)
pages = []

async def load_data():
    data = []
    async for doc in loader.alazy_load():
        data.append(doc)
    return data

data_list = asyncio.run(load_data())

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(documents=data_list, embedding=embeddings)
retriever = db.as_retriever()

@app.route('/ask-question', methods=['POST'])
def ask_question():
    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        groq_api_key='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',
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
        # other params...
    )
    
    data = request.get_json()
    question = data.get('question')

    template = """
    User: {question}
    Assistant: {answer}
    Note: Do not provide the answer more than 4 lines.
    In case you do not know the answer, then just say that you do not know.
    Answers which you generate is viewed by the recruiter so what ever the answer which you generate that has to be very positive and it has to give very good impression to get the job, so please generate you answers as per to it
    If the question is not related to Aliksm, then you directly say that "I dont know, I can answer only about Aliksm"
    """
    
    chat_prompt_template = ChatPromptTemplate.from_template(template)
    formatted_question = chat_prompt_template.format(
        question=question,
        answer=""
    )

    # retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    # result = retrievalQA(formatted_question)
    # answer = result['result']
    retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    result = retrievalQA.invoke(formatted_question)  # Update to use invoke method
    answer = result['result']

    return jsonify({"question": question, "answer": answer})

@app.route('/')
def home():
    return render_template('know_me.html')

if __name__ == '__main__':
    app.run(debug=True)
