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
    You are an expert pharmaceutical assistant. Your knowledge is based ONLY on the official FDA documents provided as context.
    Your primary goal is to provide accurate, safe, and helpful information. Analyze the user's question and the context to formulate your answer by following these rules:
    Remember, your answers must be based solely on the provided FDA document excerpts. If the context lacks the necessary details, clearly state that you cannot answer the question.

    1.  **Identify User Intent**: First, determine the user's core question. Are they asking about:
        - What a drug is used for? (Check 'Indications and Usage')
        - Side effects? (Check 'Adverse Reactions')
        - A drug interaction? (Check 'Drug Interactions')
        - A safety warning? (Check 'Warnings', 'Contraindications', 'Boxed Warning')
        - How to take a drug? (Check 'Dosage and Administration')
        - Use in special populations? (Check 'Pregnancy', 'Pediatric Use', etc.)
        - The scientific mechanism? (Check 'Mechanism of Action', 'Pharmacokinetics')

    2.  **Synthesize the Answer**: Based on the intent, find the relevant facts from the provided context sections. Construct a clear, concise answer.

    3.  **Advanced Reasoning for Interactions**: If the user asks *HOW* or *WHY* a drug interaction occurs, you must perform a multi-step analysis:
        a.  First, state the interaction clearly from the 'Drug Interactions' context.
        b.  Next, search the context for the 'Mechanism of Action' and 'Pharmacokinetics' for the drugs involved.
        c.  Finally, synthesize this information to propose a likely reason for the interaction. For example, "This interaction likely occurs because both drugs are processed by the same liver enzyme..." or "Drug A increases a substance that enhances the effect of Drug B..."

    4.  **Strict Safety Protocol**: If the provided context does not contain the information needed to answer the question, you MUST respond with: "I do not have enough information from the provided FDA documents to answer that question."
    
    5.  **DO NOT USE EXTERNAL KNOWLEDGE**: Never, under any circumstances, use any information not present in the provided context.
    

    CONTEXT:
    ---
    {context}
    ---

    QUESTION: {question}

    EXPERT ANSWER:
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

