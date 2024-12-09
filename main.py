from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

hf = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-v0.1",
    temperature=0.1,
    max_length=100,
    token='HUGGINGFACEHUB_API_TOKEN'
)

loader = PyPDFLoader("/home/ajay/Desktop/new/tes/handbook.pdf")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
docs_with_embeddings = []
for doc in docs:
    embedding = embeddings.embed_documents([doc.page_content])[0]  
    doc.metadata['embedding'] = embedding
    docs_with_embeddings.append(doc)

vectorstore = FAISS.from_documents(docs_with_embeddings, embeddings)


template = """
<s>[INST] You must follow these rules strictly: 
1. Provide ONLY ONE direct answer to the question based solely on the provided context.
2. Use ONLY the information from the following context.
3. Do NOT generate any follow-up questions or additional information.
4. Do NOT continue the conversation after providing the answer.
5. If the information is not found in the context, say "Information not found in the provided documents."
[/INST] 

Context: {context}

Question: {question}

<s>[INST] Provide a direct, concise answer. [/INST] Model answer</s>
"""

PROMPT = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)


qa_chain = RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=False,
    chain_type_kwargs={"prompt": PROMPT}
)


query = "Who is the CEO of Zania?"
retrieved_docs = vectorstore.as_retriever(search_kwargs={"k": 3}).get_relevant_documents(query)


query_embedding = np.array(embeddings.embed_query(query)).reshape(1, -1)
doc_embeddings = np.array([np.array(doc.metadata['embedding']) for doc in retrieved_docs])
similarity_scores = cosine_similarity(query_embedding, doc_embeddings).flatten()


for i, doc in enumerate(retrieved_docs):
    print(f"Document {i + 1}:")
    print(f"Text: {doc.page_content}")
    print(f"Similarity Score: {similarity_scores[i]:.4f}")
    print()


response = qa_chain.run({"query": query}).strip()
print("Initial Response:", response)


critique_prompt = f"""
<s>[INST] Critique the following response:
Question: {query}
Response: {response}

Evaluate the response based on:
1. Alignment with the context.
2. Completeness and correctness.
3. Clarity and conciseness.

Provide your critique in a single paragraph. Avoid repetition and extraneous text. [/INST]
"""

def clean_response(output: str) -> str:
   
    output = output.replace("[INST]", "").replace("[/INST]", "").replace("[MODEL]", "").strip()
   
    lines = output.split("\n")
    unique_lines = list(dict.fromkeys(lines))
    return " ".join(unique_lines).strip()


critique = hf.invoke(critique_prompt)
critique = clean_response(critique)
print("Critique:", critique)


refine_prompt = f"""
<s>[INST] Based on the critique provided below, improve the response:
Critique: {critique}

Use the feedback to revise and improve the response. Focus on improving clarity, conciseness, and alignment with the context in a natural conversational tone. 

Original Question: {query}
Original Response: {response}

Provide a revised response in a single paragraph. Do not repeat the critique, only the improved response. [/INST]
"""

refined_response = hf.invoke(refine_prompt)
refined_response = clean_response(refined_response)
print("Refined Response:", refined_response)

