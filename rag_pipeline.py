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
    You are an expert pharmaceutical assistant. Your knowledge is based solely on the official FDA documents provided as context.
    Your task is to answer the user's question based on the following rules:

    1.  **Direct Questions**: For direct questions about a drug's usage, side effects, warnings, or contraindications, synthesize your answer directly from the relevant context sections ("Indications and Usage", "Adverse Reactions", "Warnings", "Contraindications").

    2.  **Interaction Questions**: If the user asks if two or more drugs interact, first find the "Drug Interactions" section for the primary drug mentioned.

    3.  **"How/Why" Interpretation**: If the user asks *how* or *why* an interaction occurs, you must perform a deeper analysis:
        a. State the interaction as described in the "Drug Interactions" context.
        b. Then, review the "Mechanism of Action" and "Pharmacokinetics" sections for the involved drugs.
        c. Based on those scientific sections, form a hypothesis and explain the likely biological reason for the interaction (e.g., "This interaction likely occurs because both drugs are metabolized by the same liver enzyme..." or "Drug A increases substance X, while Drug B's effect is enhanced by substance X...").

    4.  **Safety First**: If the provided context does not contain the answer, you MUST state: "I do not have enough information from the provided FDA documents to answer that question." Do not, under any circumstances, use external knowledge or make up an answer.

    Context:
    ---
    {context}
    ---

    Question: {question}

    Expert Answer:
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

