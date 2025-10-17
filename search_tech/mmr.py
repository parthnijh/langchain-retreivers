from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
documents = [
    Document(
        page_content=(
            "Climate change is intensifying extreme weather events. "
            "Rising global temperatures increase the frequency of heatwaves, storms, and heavy rainfall. "
            "These extreme events damage infrastructure, disrupt agriculture, and displace communities."
        ),
        metadata={"source": "impact_extreme_weather"}
    ),
    Document(
        page_content=(
            "Global warming is causing sea levels to rise due to melting ice caps and glaciers. "
            "Coastal areas face flooding, erosion, and saltwater intrusion, endangering small islands "
            "and freshwater supplies."
        ),
        metadata={"source": "impact_sea_levels"}
    ),
    Document(
        page_content=(
            "Climate change threatens biodiversity by altering habitats and disrupting ecosystems. "
            "Species are forced to migrate or face extinction, while coral reefs suffer bleaching due to warmer oceans. "
            "This loss of biodiversity weakens ecosystem resilience."
        ),
        metadata={"source": "impact_biodiversity"}
    ),
    Document(  # Similar to Document 5
        page_content=(
            "Climate change affects human health by increasing heat-related illnesses and worsening air quality. "
            "Rising temperatures enable the spread of diseases like malaria and dengue, putting vulnerable populations at risk."
        ),
        metadata={"source": "impact_human_health_1"}
    ),
    Document(  # Similar to Document 4
        page_content=(
            "Human health is under growing threat from climate change. "
            "Prolonged heatwaves cause heat stress, while warmer conditions spread vector-borne diseases such as dengue and Zika. "
            "Poor air quality further intensifies respiratory issues."
        ),
        metadata={"source": "impact_human_health_2"}
    ),
]
vectorstore=Chroma(
    embedding_function=embeddings,
    persist_directory="chroma_db2",
    collection_name="sample2"
)
vectorstore.add_documents(documents)
retreiver=vectorstore.as_retriever(search_type="mmr",search_kwargs={"k":3,"lambda_mult":0.7})
query="what are the impacts of climate change?"
result=retreiver.invoke(query)
for item in result:
    print(item.page_content)