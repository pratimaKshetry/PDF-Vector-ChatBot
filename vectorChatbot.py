import streamlit as st
import fitz
import faiss
import numpy as np
from openai import OpenAI
import tiktoken


AZURE_EP    = "https://abc" #Use your end point  
DEPLOYMENT  = "gpt-4-0125-preview"  
encoding=tiktoken.encoding_for_model(DEPLOYMENT )   
token_provider= get_bearer_token_provider(InteractiveBrowserCredential(),'api://abc')  #Use token given by SRE team
MODEL  = "gpt-4-0125-preview"  


client = AzureOpenAI(       
        api_key="#####",  #Use your API key 
        azure_endpoint=AZURE_EP,
        #azure_ad_token_provider=token_provider,      
        api_version= "2024-05-01-preview"  
                  
    )   

# Open pdf file and extract text from pdf document
def extract_text_from_pdf(uploaded_file):
    pdf = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf:
        text += page.get_text()
    return text

#chunking texts
def chunk_text(text, max_tokens=2000):
    enc = tiktoken.encoding_for_model("gpt-4.1")
    tokens = enc.encode(text)

    chunks = []
    for i in range(0, len(tokens), max_tokens):
        sub_tokens = tokens[i:i+max_tokens]
        chunks.append(enc.decode(sub_tokens))

    return chunks

#Following function generates a vector embedding for a given text chunk using   OpenAI's text-embedding-3-large model.
 #Returns a numpy array of float32 values.
def embed(text):
    resp = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return np.array(resp.data[0].embedding, dtype="float32")

#building FAISS index
#the following function creates a FAISS L2 index over all chunk embeddings.
#It returnsindex (FAISS index); embeddings (list of numpy vectors)

def build_faiss_index(chunks):
    dim = 3072  # embedding dimension for text-embedding-3-large
    index = faiss.IndexFlatL2(dim)

    embeddings = [embed(c) for c in chunks]
    embedding_matrix = np.vstack(embeddings)
    index.add(embedding_matrix)

    return index, embeddings

#Following function summarizes per chunk
def summarize_chunk(chunk):
    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You summarize text clearly and concisely."},
            {"role": "user", "content": f"Summarize this section:\n\n{chunk}"}
        ]
    )
    return response.choices[0].message.content

#combine summaries from chunk summaries generated from  summarize_chunk()
def combine_summaries(chunk_summaries):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You are an expert summarizer."},
            {"role": "user", "content": "Combine these summaries into one coherent final summary:\n\n" + "\n\n".join(chunk_summaries)}
        ]
    )
    return response.choices[0].message.content

#chatbot begins now
#following function answers the user's question using only information retrieved  from the FAISS index (retrieved_text) function i.e build_faiss_index(chunks).
#This enforces grounding in the PDF content.

def answer_question(question, retrieved_text):
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": "You answer questions strictly using the provided document text."},
            {"role": "assistant", "content": retrieved_text},
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content


# Main function

def main():
    st.set_page_config(page_title="PDF Summarizer Chatbot", layout="wide")

    st.title("PDF Summarizer + Chatbot (Developed using LLM + FAISS)")
    st.write("Upload a PDF, generate a summary, and ask questions about it using retrieval-augmented generation.")

    uploaded_pdf = st.file_uploader("Upload your PDF file", type=["pdf"])

    if uploaded_pdf:
        with st.spinner("Extracting text from PDF..."):
            text = extract_text_from_pdf(uploaded_pdf)

        st.success("PDF text extracted successfully!")

        # Chunking
        chunks = chunk_text(text)
        st.write(f"Total chunks created: {len(chunks)}")

        # Generate summary
        if st.button("Generate Summary"):
            chunk_summaries = []
            progress = st.progress(0)

            for i, c in enumerate(chunks):
                summary = summarize_chunk(c)
                chunk_summaries.append(summary)
                progress.progress((i + 1) / len(chunks))

            final_summary = combine_summaries(chunk_summaries)

            st.subheader("Final Summary")
            st.write(final_summary)

            st.session_state["summary"] = final_summary
            st.session_state["chunks"] = chunks

    # Q&A Section
    if "chunks" in st.session_state:
        with st.spinner("Building FAISS retrieval index..."):
            index, embeddings = build_faiss_index(st.session_state["chunks"])
            st.session_state["faiss_index"] = index
            st.session_state["embeddings"] = embeddings

        st.success("Index ready—ask questions below!")

        st.subheader("Ask a Question About the PDF")
        user_query = st.text_input("Your question")

        if user_query:
			# Get embedding for query
            q_embed = embed(user_query).reshape(1, -1)
			
			# Retrieve top 3 closest chunks
            distances, indices = st.session_state["faiss_index"].search(q_embed, k=3)

            retrieved_text = "\n\n".join(
                st.session_state["chunks"][i] for i in indices[0]
            )

            response = answer_question(user_query, retrieved_text)

            st.write("Answer:")
            st.write(response)



if __name__ == "__main__":
    main()
