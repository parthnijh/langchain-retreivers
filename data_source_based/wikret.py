from langchain_community.retrievers import WikipediaRetriever
retriever=WikipediaRetriever(top_k_results=2,lang="en")
query="geopolitical history of india and pakistan"
docs=retriever.invoke(query)
print(len(docs))