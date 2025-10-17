from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
doc_sachin = Document(
    page_content=(
        "Sachin Tendulkar is regarded as the God of Cricket. "
        "He is the highest run scorer in international cricket with 100 centuries "
        "and played a key role in India's 2011 World Cup win."
    ),
    metadata={"name": "Sachin Tendulkar", "role": "Batsman", "era": "1989-2013"}
)

doc_kohli = Document(
    page_content=(
        "Virat Kohli is one of India's greatest modern batsmen. "
        "Known for his aggression and fitness, he has over 70 international centuries "
        "and has led India to major Test victories overseas."
    ),
    metadata={"name": "Virat Kohli", "role": "Batsman", "era": "2008-Present"}
)

doc_dhoni = Document(
    page_content=(
        "MS Dhoni is India's most successful captain, famous for his calm nature and finishing ability. "
        "He led India to the 2007 T20 World Cup, 2011 ODI World Cup, and 2013 Champions Trophy victories."
    ),
    metadata={"name": "MS Dhoni", "role": "Wicketkeeper-Batsman", "era": "2004-2019"}
)

doc_rohit = Document(
    page_content=(
        "Rohit Sharma, known as the Hitman, holds the record for the highest ODI score (264). "
        "He has multiple double centuries and currently leads the Indian national team."
    ),
    metadata={"name": "Rohit Sharma", "role": "Batsman", "era": "2007-Present"}
)

doc_bumrah = Document(
    page_content=(
        "Jasprit Bumrah is India's premier fast bowler, known for his yorkers and unorthodox action. "
        "He is a key performer in Tests, ODIs, and T20Is, especially in death overs."
    ),
    metadata={"name": "Jasprit Bumrah", "role": "Bowler", "era": "2016-Present"}
)
docs=[doc_sachin,doc_bumrah,doc_dhoni,doc_kohli,doc_rohit]
vectorstore=Chroma(
    embedding_function=embeddings,

    persist_directory="chroma_db1",
    collection_name="sample"
)
vectorstore.add_documents(docs)
retriever=vectorstore.as_retriever(search_kwargs={"k":2})
query="who is virat kohli"
result=retriever.invoke(query)
print(result)