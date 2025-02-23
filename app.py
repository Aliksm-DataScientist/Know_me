from flask import Flask, request, jsonify, render_template
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_core.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA
import asyncio

# app = Flask(__name__)
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

llm = ChatGroq(
    model="mixtral-8x7b-32768",
    groq_api_key='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',
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
db = FAISS.from_documents(documents=data_list, embedding=embeddings)
retriever = db.as_retriever()
retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=retriever)

# chat_history = []

# def update_chat_history(question, answer):
#     chat_history.append({"question": question, "answer": answer})
#     if len(chat_history) > 10:
#         chat_history.pop(0)  # Remove the oldest message if more than 10

#     # Save chat history to HTML file
#     with open('chat_history.html', 'w') as f:
#         f.write(generate_chat_html(chat_history))

# def generate_chat_html(chat_history):
#     chat_html = ""
#     for msg in chat_history:
#         chat_html += f"""
#         <div class="message user-question">
#             <p>{msg['question']}</p>
#         </div>
#         <div class="message assistant-answer">
#             <p>{msg['answer']}</p>
#         </div>
#         """
#     return chat_html

@app.route('/ask-question', methods=['POST'])
def ask_question():
    
    data = request.get_json()
    question = data.get('question')

    # Define the prompt template
    # Here are the latest 10 chat messages:
    # I have all my previous questions and asnwers in this {chat_history}, in case if question related to the previous question and answer, please fetch the answer of it reframe the answer as per to the question and return the response
    template = """
    
    
    User: {question}
    Assistant: {answer}



    Note: Do not provide the answer more than 4 lines.
    In case you do not know the answer, then just say that you do not know.
    Answers which you generate is viewed by the recruiter so what ever the answer which you generate that has to be very positive and it has to give very good impression to get the job, so please generate you answers as per to it
    If the question is not related to Aliksm, then you directly say that "I dont know, I can answer only about Aliksm"
    """
    
    # Format the question using the prompt template
    chat_prompt_template = ChatPromptTemplate.from_template(template)
    formatted_question = chat_prompt_template.format(
        # chat_history=chat_history,
        question=question,
        answer=""
    )

    result = retrievalQA(formatted_question)
    answer = result['result']
    # update_chat_history(question, answer)
    return jsonify({"question": question, "answer": answer})

# @app.route('/chat_history.html')
# def chat_history_html():
#     return generate_chat_html(chat_history)

@app.route('/')
def home():
    return render_template('know_me.html')

if __name__ == '__main__':
    app.run(debug=True)
