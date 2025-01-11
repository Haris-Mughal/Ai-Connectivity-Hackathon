import os
import hashlib
import io
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from langchain_huggingface import HuggingFaceEmbeddings
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection
import json

class FileHandler:
    def __init__(self,api_token,logger):
        self.logger = logger
        self.logger.info("Initializing FileHandler...")
        # Initialize the embedding model using Hugging Face
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"token": api_token},
        )

    def handle_file_upload(self, file, document_name, document_description):
        try:
            content = file.read()
            file_hash = hashlib.md5(content).hexdigest()
            collection_name = f"collection_{file_hash}"

            # Check if the collection exists
            if connections._fetch_handler().has_collection(collection_name):
                self.logger.info(f"Collection '{collection_name}' already exists.")
                return {"message": "File already processed."}

            # Process file based on type
            if file.name.endswith(".pdf"):
                texts, metadatas = self.load_and_split_pdf(file)
            elif file.name.endswith(".docx"):
                texts, metadatas = self.load_and_split_docx(file)
            elif file.name.endswith(".txt"):
                texts, metadatas = self.load_and_split_txt(content)
            elif file.name.endswith(".xlsx"):
                texts, metadatas = self.load_and_split_table(content)
            elif file.name.endswith(".csv"):
                texts, metadatas = self.load_and_split_csv(content)
            else:
                self.logger.info("Unsupported file format.")
                raise ValueError("Unsupported file format.")


            if not texts:
                return {"message": "No text extracted from the file. Check the file content."}

            # self._store_vectors(collection_name, texts, metadatas)
            filename = file.name
            filelen = len(content)
            self._store_vectors(collection_name, texts, metadatas, document_name, document_description,filename,filelen)
            self.logger.info(f"File processed successfully. Collection name: {collection_name}")

            return {"message": "File processed successfully."}
        except Exception as e:
            self.logger.error(f"Error processing file: {str(e)}")
            return {"message": f"Error processing file: {str(e)}"}

    def _store_vectors(self, collection_name, texts, metadatas, document_name, document_description,file_name,file_len):
        fields = [
            FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
            FieldSchema(name="file_name_hash", dtype=DataType.INT64),  # Hash of file name
            FieldSchema(name="document_name_hash", dtype=DataType.INT64),  # Hash of document name
            FieldSchema(name="document_description_hash", dtype=DataType.INT64),  # Hash of document description
            FieldSchema(name="file_meta_hash", dtype=DataType.INT64),
            FieldSchema(name="file_size", dtype=DataType.INT64),
        ]
        schema = CollectionSchema(fields, description="Document embeddings with metadata")
        collection = Collection(name=collection_name, schema=schema)
        # Generate embeddings
        embeddings = [self.embeddings.embed_query(text) for text in texts]

        # Convert metadata to hashed values
        file_name_hash = int(hashlib.md5(file_name.encode('utf-8')).hexdigest(), 16) % (10 ** 12)
        document_name_hash = int(hashlib.md5((document_name or "Unknown Document").encode('utf-8')).hexdigest(), 16) % (
                    10 ** 12)
        document_description_hash = int(
            hashlib.md5((document_description or "No Description Provided").encode('utf-8')).hexdigest(), 16) % (
                                                10 ** 12)
        # Convert metadata list to JSON string and hash it
        metadata_string = json.dumps(metadatas, ensure_ascii=False)
        file_meta_hash = int(hashlib.md5(metadata_string.encode('utf-8')).hexdigest(), 16) % (10 ** 12)

        # Prepare data for insertion
        data = [
            embeddings,
            [file_name_hash] * len(embeddings),
            [document_name_hash] * len(embeddings),
            [document_description_hash] * len(embeddings),
            [file_meta_hash] * len(embeddings),
            [file_len or 0] * len(embeddings),
        ]

        # Insert data into collection
        collection.insert(data)
        collection.load()
    def load_and_split_pdf(self, file):
        reader = PdfReader(file)
        texts = []
        metadatas = []
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text:
                texts.append(text)
                metadatas.append({"page_number": page_num + 1})
        return texts, metadatas

    def load_and_split_docx(self, file):
        doc = Document(file)
        texts = []
        metadatas = []
        for para_num, paragraph in enumerate(doc.paragraphs):
            if paragraph.text:
                texts.append(paragraph.text)
                metadatas.append({"paragraph_number": para_num + 1})
        return texts, metadatas

    def load_and_split_txt(self, content):
        text = content.decode("utf-8")
        lines = text.split('\n')
        texts = [line for line in lines if line.strip()]
        metadatas = [{}] * len(texts)
        return texts, metadatas

    def load_and_split_table(self, content):
        excel_data = pd.read_excel(io.BytesIO(content), sheet_name=None)
        texts = []
        metadatas = []
        for sheet_name, df in excel_data.items():
            df = df.dropna(how='all', axis=0).dropna(how='all', axis=1)
            df = df.fillna('N/A')
            for _, row in df.iterrows():
                row_dict = row.to_dict()
                # Combine key-value pairs into a string
                row_text = ', '.join([f"{key}: {value}" for key, value in row_dict.items()])
                texts.append(row_text)
                metadatas.append({"sheet_name": sheet_name})
        return texts, metadatas

    def load_and_split_csv(self, content):
        csv_data = pd.read_csv(io.StringIO(content.decode('utf-8')))
        texts = []
        metadatas = []
        csv_data = csv_data.dropna(how='all', axis=0).dropna(how='all', axis=1)
        csv_data = csv_data.fillna('N/A')
        for _, row in csv_data.iterrows():
            row_dict = row.to_dict()
            row_text = ', '.join([f"{key}: {value}" for key, value in row_dict.items()])
            texts.append(row_text)
            metadatas.append({"row_index": _})
        return texts, metadatas

