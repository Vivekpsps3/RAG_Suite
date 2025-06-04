from src import tf_tools, pdf_processor, rag_retriever, csv_processor


LLM = tf_tools.LMAnswerModel()
PDF_Processor = pdf_processor.PDFProcessor(pdf_directory="documents/pdf")
RAG_Retriever = rag_retriever.NaiveRAGRetriever(embedder=LLM.embedder)
CSV_Processor = csv_processor.CSVProcessor(csv_directory="documents/csv") # Added CSV_Processor

if __name__ == "__main__":
    # Load and process PDF documents
    pdf_doc_infos = PDF_Processor.get_documents_for_rag()

    print(f"Processed {len(pdf_doc_infos)} documents from PDF files.")

    # Add documents to the RAG retriever
    if pdf_doc_infos:
        added_documents = RAG_Retriever.add_documents(
            documents=[doc['text'] for doc in pdf_doc_infos],
            metadatas=[doc['metadata'] for doc in pdf_doc_infos],
            ids=[doc['id'] for doc in pdf_doc_infos]
        )
        print(f"Added {len(added_documents)} documents to the RAG retriever.")

    # Load and process CSV documents
    csv_doc_infos = CSV_Processor.get_documents_for_rag()
    print(f"Processed {len(csv_doc_infos)} documents from CSV files.")

    if csv_doc_infos: # Check if there are any CSV documents to add
        # Add CSV documents to the RAG retriever
        added_csv_documents = RAG_Retriever.add_documents(
            documents=[doc['text'] for doc in csv_doc_infos],
            metadatas=[doc['metadata'] for doc in csv_doc_infos],
            ids=[doc['id'] for doc in csv_doc_infos]
        )
        print(f"Added {len(added_csv_documents)} CSV documents to the RAG retriever.")


    # Create a query to find out who Vivek is
    query = "Rajul Garg"
    print(f"Query: {query}")


    # Generate get relevant content embeddings for the query
    relevant_content = RAG_Retriever.query(query_text=query, top_k=5)
    print(f"Retrieved contents for the query: {relevant_content}")

    # Generate response using the LLM with the relevant content as context
    # Extract the document text from the relevant_content dictionary
    context = ""
    if relevant_content and 'documents' in relevant_content and relevant_content['documents']:
        context = "\n".join(relevant_content['documents'])
    else:
        context = "No relevant information found."

    prompt_with_context = f"Based on the following information:\n{context}\n\nAnswer the question: Tell me everything about {query}. Where are they based out of?"
    response = LLM.generate(prompt_with_context)
    print(f"Response to '{query}': {response}")
