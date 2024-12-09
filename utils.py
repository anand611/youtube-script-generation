#from langchain.llms import OpenAI
#The above is no longer avialable, so replaced it with the below import :)
# from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.tools import DuckDuckGoSearchRun
import os

os.environ["OPENAI_API_KEY"] = "sk-8iW6nev21doh1Lnsfkdh;alfhsFJM2pdNhYz2783g3bO98vjeuPCT"
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
os.environ["PINECONE_API_KEY"] = "pcsk_4kUXTh_Bou27oHeWgQezwntpKBfjaRvNVd6tx8HBMAhieUDBq8ALRMixxA23hrr9cGjDGx"
os.environ["MISTRAL_API_KEY"]="PzsIjy5b1ldYJEt0thzJPpBGbFKocCFy"
os.environ["PINECONE_ENV"]="us-east-1"
os.environ["PINECONE_INDEX"]="ac611"

def get_llm_model(index=1):
    models = ["mistral-large-latest","ministral-3b-latest"]
    model = ChatMistralAI(model=models[index])
    return model

# Function to generate video script
def generate_script(prompt,video_length):
    
    # Template for generating 'Title'
    title_template = PromptTemplate(
        input_variables = ['subject'], 
        template='Please come up with a title for a YouTube video on the {subject}.'
        )

    # Template for generating 'Video Script' using search engine
    script_template = PromptTemplate(
        input_variables = ['title', 'DuckDuckGo_Search','duration'], 
        template='Create a script for a YouTube video based on this title for me. TITLE: {title} of duration: {duration} minutes using this search data {DuckDuckGo_Search} '
    )

    #Setting up OpenAI LLM
    # llm = ChatOpenAI(temperature=creativity,openai_api_key=api_key,
    #         model_name='gpt-3.5-turbo') 
    llm = get_llm_model()
    
    #Creating chain for 'Title' & 'Video Script'
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)

    
    # https://python.langchain.com/docs/modules/agents/tools/integrations/ddg
    search = DuckDuckGoSearchRun()

    # Executing the chains we created for 'Title'
    title = title_chain.invoke(prompt)

    # Executing the chains we created for 'Video Script' by taking help of search engine 'DuckDuckGo'
    search_result = search.run(prompt) 
    script = script_chain.run(title=title, DuckDuckGo_Search=search_result,duration=video_length)

    # Returning the output
    return search_result,title,script


#################################################################################################################################


# import os
# from langchain_community.document_loaders import PyPDFDirectoryLoader
# from langchain_mistralai.chat_models import ChatMistralAI
# from langchain_text_splitters.character import RecursiveCharacterTextSplitter
# from langchain_huggingface import HuggingFaceEmbeddings
# from pinecone import Pinecone as PineconeClient
# from langchain_community.vectorstores import pinecone
# from langchain.chains.question_answering import load_qa_chain
# from langchain.output_parsers import StructuredOutputParser,ResponseSchema
# from langchain.prompts import ChatPromptTemplate,HumanMessagePromptTemplate

# # os.environ["OPENAI_API_KEY"] = "sk-8iW6nev21doh1Lnsfkdh;alfhsFJM2pdNhYz2783g3bO98vjeuPCT"
# # os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
# # os.environ["PINECONE_API_KEY"] = "pcsk_4kUXTh_Bou27oHeWgQezwntpKBfjaRvNVd6tx8HBMAhieUDBq8ALRMixxA23hrr9cGjDGx"


