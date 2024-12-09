import streamlit as st
import os
from utils import *

os.environ["MISTRAL_API_KEY"]="PzsIjy5b1ldYJEt0thzJPpBGbFKocCFy"

st.title('❤️ Get answer from documents') 

# # Sidebar to capture the OpenAi API key
# st.sidebar.title("😎🗝️")
# st.session_state['API_Key']= st.sidebar.text_input("What's your API key?",type="password",value="efHp2fnjFAWmhjsTKOL9EmxQ2qFutOmm")
# docs_loaded = st.sidebar.button("Load docs",on_click=get_documents)
# # st.sidebar.image('./Youtube.jpg',width=300, use_container_width=True)


# # Captures User Inputs
# prompt = st.text_input('Please provide the topic of the video',key="prompt")  # The box for the text prompt
# # video_length = st.text_input('Expected Video Length 🕒 (in minutes)',key="video_length")  # The box for the text prompt
# # creativity = st.slider('Creativity limit ✨ - (0 LOW || 1 HIGH)', 0.0, 1.0, 0.2,step=0.1)

# submit = st.button("Get answer")


def get_answer_clicked(query):
    st.session_state['docs_loaded'] = True
    if not st.session_state['docs_loaded']:
        st.warning("Data not loaded")
    # st.toast(f"Query: {query}")
    if query and st.session_state['docs_loaded']:
        st.toast("Question submitted..")
        # llm=get_llm_model()
        # chain = get_chain(llm=llm)
        # answer = get_answer(chain=chain,query=query)
        # st.success(answer)
        # st.write("MCQs from the response....")
        # markdown_text = generate_mcq_from_document(answer)
        # json_string = re.search(r'{(.*?)}',markdown_text,re.DOTALL).group(1)
        # st.write(json_string)
        
        documents = get_documents(from_file=False,file_path='')
        splitted_data = splitting_text(documents=documents)
        embeddings,vectorstore = get_embeddings_model(api_key=os.environ["MISTRAL_API_KEY"])
        push_to_vectorstore(vectorstore=vectorstore,splitted_text=splitted_data)
        
        results = vectorstore.similarity_search(query=query)
        print(results)
        st.markdown(results)
        
def main():
    st.session_state['docs_loaded'] = True
    # st.set_page_config(
    # page_title="Resolve Query",
    # page_icon="🗃",
    # )
    st.write("# Resolve Query 👋")
    # directory = file_selector()
    # st.button("Load Data",on_click=load_data,args=[directory])
    query = st.text_input("Question",placeholder="what is your query?")
    st.button("Get Answer",on_click=get_answer_clicked,args=[query])
if __name__=="__main__":
    main()    
