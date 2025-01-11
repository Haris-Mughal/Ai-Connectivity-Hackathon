import os
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from pymilvus import connections, Collection

class ChatHandler:
    def __init__(self,api_token,grok_api_token,logger):
        self.logger = logger
        self.logger.info("Initializing ChatHandler...")
        self.groq_client = Groq(api_key=grok_api_token)
        # Initialize the embedding model using Hugging Face
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"token": api_token},
        )

    def _query_groq_model(self, prompt):
        """
        Query Groq's Llama model using the SDK.
        """
        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.1-8b-instant",  # Ensure the model name is correct
            )
            # Return the assistant's response
            return chat_completion.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error querying Groq API: {e}")
            return f"Error querying Groq API: {e}"

    def answer_question(self, question):
        # Generate embedding for the question
        self.logger.info(f"Received question: {question}")
        collections = connections._fetch_handler().list_collections()
        responses = []

        for collection_name in collections:
            collection = Collection(name=collection_name)
            embeddings = self.embeddings.embed_query(question)

            search_params = {
                "metric_type": "IP",
                "params": {"nprobe": 10},
            }

            results = collection.search(
                data=[embeddings],
                anns_field="embedding",
                param=search_params,
                limit=5,
            )
            # Extract the embeddings or metadata (if needed)
            for res in results[0]:
                # Store the ID or use res.distance if needed for similarity score
                responses.append({"id": res.id, "distance": res.distance,"content":res.entity})

        if responses:
            sorted_responses = sorted(responses, key=lambda x: x["distance"], reverse=True)
            prompt = self._generate_prompt(question, sorted_responses[:5])
            response = self._query_groq_model(prompt)
            return response


        return "No relevant documents found or context is insufficient to answer your question."

    def _generate_prompt(self, question, documents):
        """
        Generate a structured prompt tailored to analyze government energy consumption data
        and answer questions effectively using the provided documents.
        """
        context = "\n".join(
            [
                f"Document {i + 1}:\nID: {doc['id']}\nSimilarity: {doc['distance']:.4f}\nContent: {doc['content']}"
                for i, doc in enumerate(documents[:5])
            ]
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
