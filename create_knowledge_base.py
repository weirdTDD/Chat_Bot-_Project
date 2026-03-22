import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
from datasets import load_dataset
import os 


def create_ecommerce_knowledge_base():
    #Create a vector database from Bitext e-commerce dataset

    print("Loading Bitext e-commerce dataset...")

    #Load the dataset
    dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")


    print(f"Dataset loaded: {len(dataset['train'])} examples")

    #prepare knowledge base
    knowledge_base = []
    for example in dataset['train']:

        #clean the response 
        response = example['response']
        response = response.replace("{{Order Number}}", "your order")
        response = response.replace("{{Online Company Portal Info}}", "our website")
        response = response.replace("{{Online Order Interaction}}", "order  History")
        response = response.replace("{{Customer Support Hours}}", "business hours")
        response = response.replace("{{Customer Support Phone Number}}", "our support line")
        response = response.replace("{{Website URL}}", "our website")

        knowledge_base.append({
            'question': example['instruction'],
            'answer': response,
            'intent': example['intent'],
            'category': example['category'],
        })

    print("Loading sentence transformer model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2') 