import os
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

class ChatHandler:
    def __init__(self, vector_db_path,open_api_key):
        self.vector_db_path = vector_db_path
        self.embeddings = OpenAIEmbeddings(api_key=open_api_key)
        self.llm = ChatOpenAI(
            model_name="gpt-4",
            api_key=open_api_key,
            max_tokens=500,
            temperature=0.2,
        )


    def answer_question(self, question):
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
            return self.llm.invoke(prompt)

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
    You are an advanced AI assistant with expertise in energy data analysis, resource optimization, 
    and sustainability practices. Your role is to analyze government energy consumption data 
    to identify inefficiencies, propose actionable strategies, and quantify potential impacts.

    ### Data Provided:
    The following documents contain detailed information about energy productivity, consumption trends, 
    and inefficiencies in various sectors:
    {context}

    ### Question:
    {question}

    ### Instructions:
    1. **Highlight Areas of Energy Waste**:
       - Identify inefficiencies such as underutilized facilities, overconsumption in specific sectors, or
         energy system losses.
       - Use data points from the documents to back your observations.

    2. **Suggest Strategies for Optimization**:
       - Recommend actionable steps like upgrading equipment, adopting renewable energy sources,
         or optimizing resource allocation.
       - Ensure suggestions are feasible and tailored to the identified inefficiencies.

    3. **Demonstrate Cost-Saving and Environmental Benefits**:
       - Provide quantitative estimates of potential cost savings from the suggested strategies.
       - Highlight the environmental benefits, such as reductions in CO2 emissions or energy waste.

    4. **Present the Response Clearly**:
       - Organize your findings in a step-by-step format.
       - Use tables, bullet points, or concise paragraphs for clarity.

    ### Example Output Format:
    - **Energy Waste Identified**:
      1. ...
      2. ...

    - **Optimization Strategies**:
      1. ...
      2. ...

    - **Cost-Saving and Environmental Benefits**:
      - Savings: $...
      - Environmental Impact: ...

    Please ensure the response is data-driven, actionable, and easy to understand.
    """
        return prompt
