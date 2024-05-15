

"""
ref for a clear explanation::
https://medium.com/@anoopjohny2000/building-a-conversational-chat-interface-with-streamlit-and-langchain-for-csvs-8c150b1f982d
    """

from flask import Flask, jsonify, request
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
#from langchain_text_splitters import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import CTransformers
#from langchain import PromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from urllib.parse import unquote
#from langchain import OpenAI

# import os
# os.environ["OPENAI_API_KEY"] = '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'

app=Flask(__name__)
#1. Extract Data From the Website & pdf
   

def extract_data_website(url_or_file):
    if url_or_file.endswith('.pdf'):
        # If the input is a PDF file
        loader = PyPDFLoader(url_or_file)
        pages = loader.load_and_split()
        text = ""
        for page in pages:
            text += page.page_content + " "
        return text
        
    else:
        # If the input is a URL
        loader = UnstructuredURLLoader([url_or_file])
        data = loader.load()
        text = ""
        for page in data:
            text += page.page_content + " "
        return text




chunks_prompt="""
Please summarize the below speech:
Speech:`{text}'
Summary:
"""
map_prompt_template=PromptTemplate(input_variables=['text'],
                                    template=chunks_prompt)



final_combine_prompt='''
Provide a final summary of the entire speech with these important points.
Add a Generic Motivational Title,
Start the precise summary with an introduction and provide the
summary in number points for the speech.
Speech: `{text}`
'''
final_combine_prompt_template=PromptTemplate(input_variables=['text'],
                                             template=final_combine_prompt)



def split_text_chunks_and_summary_generator(text):
    text_splitter=RecursiveCharacterTextSplitter(
                                        chunk_size=500,
                                        chunk_overlap=20)
    text_chunks=text_splitter.split_text(text)
    print(len(text_chunks))

    llm = CTransformers(
        model = "TheBloke/Llama-2-7B-Chat-GGML",
        model_type="llama",
        max_new_tokens = 512,
        temperature = 0.5,
        context_length= 4096
    )
    #llm = OpenAI()

    docs = [Document(page_content=t) for t in text_chunks]
    chain=load_summarize_chain(llm=llm, chain_type='map_reduce', map_prompt=map_prompt_template,
    combine_prompt=final_combine_prompt_template, verbose=True)
    summary = chain.run(docs)
    return summary

#text=extract_data_website('https://en.wikipedia.org/wiki/LLaMA')
text=extract_data_website('apj_2_page.pdf')
sum=split_text_chunks_and_summary_generator(text)
print(sum)

# @app.route('/', methods=['GET', 'POST'])
# def home():
#     return "Summary Generator"

# @app.route('/summary_generate', methods=['GET', 'POST'])
# def summary_generator():
#     encode_url=unquote(unquote(request.args.get('https://en.wikipedia.org/wiki/LLaMA')))
#     if not encode_url:
#         return jsonify({'error':'URL is required'}), 400
#     text=extract_data_website(encode_url)
#     #text_chunks=split_text_chunks(text)
#     #print(len(text_chunks))
#     summary=split_text_chunks_and_summary_generator(text)
#     print("Here is the Complete Summary", summary)
#     response= {
#         'submitted_url': encode_url,
#         'summary': summary
#     }
#     return jsonify(response)
# if __name__ == '__main__':
#     app.run(debug=True)
