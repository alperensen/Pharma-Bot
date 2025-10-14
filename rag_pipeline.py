# =================================================================================
# rag_pipeline.py: Create the Llama 3 model and the RAG chain
# =================================================================================
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
# --- THIS IS THE NEW IMPORT ---
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# ----------------------------
import config

def load_llm(hf_token):
    """
    Loads the Llama 3 model with 4-bit quantization and returns a text-generation pipeline.
    """
    print("Loading Llama 3 model...")


    tokenizer = AutoTokenizer.from_pretrained(config.LLM_MODEL_ID, token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        config.LLM_MODEL_ID,
        device_map="auto",
        token=hf_token
    )
    return pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)

def build_rag_chain(llm_pipeline, retriever):
    """
    Builds a RAG (Retrieval-Augmented Generation) chain with the given LLM and retriever.
    """
    # --- THIS IS THE FIX ---
    # Wrap the Hugging Face pipeline to make it compatible with LangChain
    llm_wrapper = HuggingFacePipeline(pipeline=llm_pipeline)
    # -----------------------

    prompt_template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Provide a concise and helpful answer based on the provided text.

    Context: {context}

    Question: {question}

    Helpful Answer:
    """
    QA_PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        # Use the wrapper instead of the raw pipeline
        llm=llm_wrapper,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    return qa_chain

