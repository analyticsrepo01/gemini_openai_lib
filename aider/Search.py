import os
import sys
from dotenv import load_dotenv
import streamlit as st

from typing import List

from google.api_core.client_options import ClientOptions
from google.cloud import discoveryengine_v1 as discoveryengine
import vertexai
from vertexai.preview.generative_models import (
    GenerationConfig,
    GenerativeModel,
    Tool,
    grounding,
)
import re

def replace_bold_tags(text):
  return re.sub(r"<b>(.*?)</b>", r"**\1**", text)

if len(sys.argv) < 3:
    print("Usage: python Search.py <PROJECT_ID> <DATASTORE_ID>")
    sys.exit(1)

PROJECT_ID = sys.argv[1]
LOCATION = "global"
DATASTORE_ID = sys.argv[2]


def search_sample(
    project_id: str,
    location: str,
    data_store_id: str,
    search_query: str,
    filter_query: str,
    no_results: int ,
    no_snippets: int,
    no_extract: int,
    no_extract_seg: int,
    no_top_results: int
) -> list[discoveryengine.SearchResponse]:

    client_options = (
        ClientOptions(api_endpoint=f"{location}-discoveryengine.googleapis.com")
        if LOCATION != "global"
        else None
    )

    client = discoveryengine.SearchServiceClient(client_options=client_options)

    serving_config = client.serving_config_path(
        project=project_id,
        location=location,
        data_store=data_store_id,
        serving_config="default_config",
    )
    content_search_spec = {
        'snippet_spec': {'return_snippet': True if no_snippets == 1 else False},
        'extractive_content_spec': {
            'max_extractive_answer_count': no_extract,
            'max_extractive_segment_count': no_extract_seg,
            'return_extractive_segment_score': True,
        },
        'summary_spec': {
            'summary_result_count': no_top_results,
            'include_citations': True,
        },
    }

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=search_query,
        page_size=no_results,
        content_search_spec=content_search_spec,
        query_expansion_spec=discoveryengine.SearchRequest.QueryExpansionSpec(
            condition=discoveryengine.SearchRequest.QueryExpansionSpec.Condition.AUTO,
        ),
        spell_correction_spec=discoveryengine.SearchRequest.SpellCorrectionSpec(
            mode=discoveryengine.SearchRequest.SpellCorrectionSpec.Mode.AUTO
        ),
    )
    if filter_query: # Only add filter if it's not empty
        request.filter = filter_query

    response = client.search(request)
    return response

def llm_prompt(
    project_id: str,
    location: str,
    data_store_id: str,
    llm_model: str,
    prompt: str,
    temp,
    top_k,
    top_p
):
    vertexai.init(project=project_id, location="us-central1")

    model = GenerativeModel(llm_model)
    tool = Tool.from_retrieval(grounding.Retrieval(grounding.VertexAISearch(datastore=data_store_id, project=project_id, location="global")))

    response = model.generate_content(
        prompt,
        tools=[tool],
        generation_config=GenerationConfig(
            temperature=temp,
            top_p=top_p,
            top_k=top_k
        ),
    )

    return(response.text)

def rerank(
    project_id: str,
    location: str,
    prompt: str,
    top_n,
    records
):
    reformatted_record = reformat_rerank(records)
    client = discoveryengine.RankServiceClient()
    
    ranking_config = client.ranking_config_path(
        project=project_id,
        location="global",
        ranking_config="default_ranking_config",
    )
    request = discoveryengine.RankRequest(
        ranking_config=ranking_config,
        model="semantic-ranker-512@latest",
        top_n=top_n,
        query=prompt,
        records=reformatted_record
    )

    response = client.rank(request=request)
    return response

def reformat_rerank(unformatted):
    records = []
    for result in unformatted:
        for extract in result.document.derived_struct_data['extractive_answers']:
            record = discoveryengine.RankingRecord(
                id=f"{extract['pageNumber']}_{result.id}",
                title=result.document.derived_struct_data['title'],
                content=extract['content']
            )
            records.append(record)
    return records

load_dotenv()

def main():
# --- App layout ---
    st.set_page_config(page_title="Search", layout='wide')
    st.title('Search')  

    with st.sidebar:
        st.title('Configurations')
        llm_model = st.selectbox(
            "LLM Model",
            ("Vertex AI Search(default)", "gemini-1.5-flash-001", "gemini-1.5-pro-001"),
        )
        if llm_model != "Vertex AI Search(default)":
            st.divider()
            temp = st.number_input("Temp", 0.0, 1.0, 0.0)
            top_k = st.number_input("Top K", 0, 100, 40)
            top_p = st.number_input("Top P", 0.0, 1.0, 0.95)
        st.divider()
        re_rank = st.toggle("Re-Rank")
        if re_rank:
            top_n = st.number_input("Top N", 1, 10, 10)
        no_results = st.slider("Results per Page", 1, 10, 10)
        no_snippets = st.slider("Snippets per Result", 0, 1, 1)
        no_extract = st.slider("Extractive Answers per Page", 0, 5, 3)
        no_extract_seg = st.slider("Extractive Segments per Page", 1, 10, 3)
        no_top_results = st.slider("Summary Results", 1, 5, 5)
    
    search_prompt = st.text_input("Search Internal Documents", value="what is the revenue for 2024?")
    filter_prompt = st.text_input("Filter Results (Optional)", placeholder="Enter structured filter query (e.g., category:\"Menswear\" OR brand:\"Google\" AND color:\"Blue\")") # Updated placeholder

    if st.button("Generate"):
        search_response = search_sample(PROJECT_ID, LOCATION, DATASTORE_ID, search_prompt, filter_prompt, no_results, no_snippets, no_extract, no_extract_seg, no_top_results) # Pass filter to search
        st.subheader("AI Answer")

        container = st.container(border=True)
        if llm_model != "Vertex AI Search(default)":
            container.markdown(replace_bold_tags(llm_prompt(PROJECT_ID, LOCATION, DATASTORE_ID, llm_model, search_prompt, temp, top_k, top_p)))
        else:
            container.markdown(replace_bold_tags(search_response.summary.summary_text))
        
        st.subheader("Results")
        if re_rank:
            response = rerank(PROJECT_ID, LOCATION, search_prompt, top_n, search_response.results)
            for result in response.records:
                container = st.container(border=True)
                container.subheader(result.title)
                container.write(f"Page {result.id.split('_')[0]}")
                container.write(f"Score: {result.score}")
                container.markdown(replace_bold_tags(result.content))
        else:
            for result in search_response.results:
                container = st.container(border=True)
                container.subheader(result.document.derived_struct_data['title'])
                for extract in result.document.derived_struct_data['extractive_answers']:
                    container.write(f"Page {extract['pageNumber']}")
                    container.markdown(replace_bold_tags(extract['content'])) 

# --- End of App layout ---

if __name__ == "__main__":
    main()
