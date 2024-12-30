import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


class ChatHandler:
    def __init__(self, vector_db_path,api_token,open_api_key):
        self.vector_db_path = vector_db_path
        # Initialize the embedding model using Hugging Face
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"token": api_token},
        )
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            api_key=open_api_key,
            max_tokens=500,
            temperature=0.2,
        )

    def answer_question(self, question):
        # Generate embedding for the question
        responses = []
        for root, dirs, files in os.walk(self.vector_db_path):
            for dir in dirs:
                index_path = os.path.join(root, dir, "index.faiss")
                if os.path.exists(index_path):
                    vector_store = FAISS.load_local(
                        os.path.join(root, dir), self.embeddings, allow_dangerous_deserialization=True
                    )
                    response_with_scores = vector_store.similarity_search_with_relevance_scores(question, k=100)
                    filtered_responses = [doc.page_content for doc, score in response_with_scores]
                    responses.extend(filtered_responses)

        if responses:
            prompt = self._generate_prompt(question, responses)
            response = self.llm.invoke(prompt)

            # Debugging: Check response structure
            # print("Response structure:", response)

            # Safely access the content attribute of AIMessage
            if hasattr(response, "content"):
                return response.content.strip()  # Ensure clean output
            else:
                return "Error: 'content' attribute not found in the AI's response."


        return "No relevant documents found or context is insufficient to answer your question."

    def _generate_prompt(self, question, documents):
        """
        Generate a structured prompt tailored to analyze government energy consumption data
        and answer questions effectively using the provided documents.
        """
        context = "\n".join(
            [f"Document {i + 1}:\n{doc.strip()}" for i, doc in enumerate(documents[:5])]
        )

        prompt = f"""
            You are an advanced AI assistant with expertise in 5G network optimization, deployment strategies, 
            and resource allocation. Your role is to analyze network datasets to identify inefficiencies, 
            propose actionable deployment and optimization strategies, and quantify potential improvements.

            ### Data Provided:
            The following documents contain detailed information about 5G network deployment, resource utilization, 
            and operational metrics:
            {context}

            ### Question:
            {question}

            ### Instructions:
            1. **Highlight Areas of Network Inefficiencies**:
               - Identify inefficiencies such as underutilized network nodes, high latency areas, or 
                 imbalanced resource allocation.
               - Use data points from the documents to back your observations.

            2. **Suggest Strategies for Network Optimization**:
               - Recommend actionable steps such as adjusting network configurations, deploying additional nodes, 
                 or reallocating bandwidth.
               - Ensure suggestions are feasible and aligned with the provided datasets.

            3. **Quantify Cost-Saving and Performance Benefits**:
               - Provide quantitative estimates of potential cost savings from the suggested strategies.
               - Highlight the performance benefits, such as improved latency, higher throughput, or enhanced user experience.

            4. **Present the Response Clearly**:
               - Organize your findings in a step-by-step format.
               - Use tables, bullet points, or concise paragraphs for clarity.

            ### Example Output Format:
            - **Network Inefficiencies Identified**:
              1. ...
              2. ...

            - **Optimization Strategies**:
              1. ...
              2. ...

            - **Cost-Saving and Performance Benefits**:
              - Cost Savings: $...
              - Performance Improvements: ...

            Please ensure the response is data-driven, actionable, and easy to understand.
        """
        return prompt
