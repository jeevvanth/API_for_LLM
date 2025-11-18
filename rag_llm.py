import dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import (
  PromptTemplate,
   SystemMessagePromptTemplate,
  HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from uvicorn import run
import requests
import asyncio
from langchain_core.runnables import RunnableLambda
import requests
import io

# from fastapi.middleware.cors import CORSMiddleware


dotenv.load_dotenv()

app=FastAPI()

chat_model = ChatGroq(
    model="llama-3.3-70b-versatile",  
    temperature=0.5
)



CHROMA_PATH = "chroma_data/"

system_template_str="""
You are an job seeker who looks for a job related to AI/ML Engineer,Generative AI Developer,Agentic AI Developer
.So the interviewer will be asking the their interview questions like introduce yourself and many personal question related to you so 
please answer it according to your context.Do remember that your from ooty,the nilgiris,TamilNadu.
Later they will ask related to AI/ML or Generative AI question and ask question about the projects that is in the context. 
If they ask about  the projects please elaborate the project to your own knownledge.
you should only generate the answer according to the questions requirements not beyond that.

Dont generate above 200 words ,dont make any gramtical errors and answer it wisely as like an working professional who attends interviews.


Here is the context:
{context}

"""

system_prompt=SystemMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["context"], template=system_template_str
    )
)

human_prompt=HumanMessagePromptTemplate(
    prompt=PromptTemplate(
        input_variables=["question"],template="{question}"
    )
)

messages=[system_prompt,human_prompt]

prompt_template=ChatPromptTemplate(
    input_variables=["context","question"],
    messages=messages,
)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector_db = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding_model
)

retriever=vector_db.as_retriever(k=10)



chain=({"context":retriever,"question":RunnablePassthrough()} |
    prompt_template 
    | chat_model)


class UserInput(BaseModel):
    question:str



@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI!"}

@app.post("/chat_response")
def text_response(user_input: UserInput):
    try:
        question = user_input.question 
        print("DEBUG incoming question:", repr(question))
        response = chain.invoke(question)

        print("DEBUG chain.invoke result:", response)


        # If response is a dict, extract 'output' or 'answer' field
        if hasattr(response, "content"):
            response_text = response.content
        else:
            response_text = str(response)

        print("response_text:",response_text)
        

        return {"response": response_text}


    except Exception as ex:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Errors: {str(ex)}")

    



if __name__=="__main__":
   run("rag_llm:app",
       host="0.0.0.0",
       port=8002,
       reload=False  
   )