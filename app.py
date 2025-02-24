from flask import Flask, request, jsonify, render_template
import asyncio

app = Flask(__name__, template_folder="templates", static_url_path='/static')

file_path = "About_Aliksm.pdf"

async def load_data():
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(file_path)
    data = []
    async for doc in loader.alazy_load():
        data.append(doc)
    return data

# Load data once at startup
data_list = asyncio.run(load_data())

def get_faiss_retriever():
    """Lazy-load FAISS retriever when needed."""
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    db = FAISS.from_documents(documents=data_list, embedding=embeddings)
    return db.as_retriever()

@app.route('/ask-question', methods=['POST'])
def ask_question():
    from langchain_groq import ChatGroq
    from langchain.prompts import ChatPromptTemplate
    from langchain.chains import RetrievalQA

    retriever = get_faiss_retriever()

    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        groq_api_key='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',  # Ensure API key is stored securely
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2,
    )

    data = request.get_json()
    question = data.get('question')

    template = """
    User: {question}
    Assistant: {answer}
    Note: Do not provide the answer more than 4 lines.
    In case you do not know the answer, then just say that you do not know.
    Answers which you generate are viewed by recruiters, so they must be positive and leave a great impression.
    If the question is not related to Aliksm, say "I don't know, I can answer only about Aliksm."
    """

    chat_prompt_template = ChatPromptTemplate.from_template(template)
    formatted_question = chat_prompt_template.format(question=question, answer="")

    retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    result = retrievalQA.invoke(formatted_question)
    answer = result['result']

    return jsonify({"question": question, "answer": answer})

@app.route('/')
def home():
    return render_template('know_me.html')

if __name__ == '__main__':
    app.run(debug=True)