import re
import pymupdf 
from pinecone import Pinecone , ServerlessSpec
from together import Together
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
# Load environment variables from CONFIG.env
config_path = os.path.join(os.path.dirname(__file__), '..','CONFIG.env')
load_dotenv(config_path)
print(config_path)

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
TOGETHER_AI_KEY = os.getenv("TOGETHER_AI_KEY")

class Complaince_RAG:

    def __init__(self,db):
        self.db=db
        self.Index_NAME= "compliance-embeddings" #index name for pinecone db
        self.pc= Pinecone(api_key= PINECONE_API_KEY)
        if self.Index_NAME not in self.pc.list_indexes().names():
                self.pc.create_index(
                    name=self.Index_NAME, 
                    dimension=384, 
                    metric='cosine',
                    spec=ServerlessSpec(
                        cloud='aws',
                        region='us-east-1'
                    )
                )
        self.index = self.pc.Index('compliance-embeddings')
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = Together(api_key=TOGETHER_AI_KEY)  #LLM access through together AI

    def _extract_text_from_pdf(self,pdf_path):
        """Extracting text data from the given pdf."""
        doc = pymupdf.open(pdf_path)
        text = "\n".join([page.get_text("text") for page in doc])
        return text.strip()
    
    def _chunk_text(self,text, chunk_size=500):
        """Splits the given document text into chunks so that number of tokens utilized during the query to LLM is less."""
        sentences = re.split(r'(?<=[.!?])\s+', text)  # Split by sentence
        chunks, current_chunk = [], ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence
            else:
                chunks.append(current_chunk.strip())
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk.strip())  

        return chunks
    
    def store_chunks_in_pinecone(self,pdf_path, compliance_label):
            """
            Stores the created chunk of the pdf in the pinecode db index.
            """
            extracted_text = self._extract_text_from_pdf(pdf_path)
            chunks = self._chunk_text(extracted_text)
            print("----Storing pdf chunks in Database--\n")
            # Embed and store chunks in Pinecone
            for i, chunk in enumerate(chunks):
                embedding = self.model.encode(chunk).tolist()  # Convert embedding to list
                metadata = {"text": chunk, "compliance": compliance_label}  # Store label
                self.index.upsert(vectors=[(f"{compliance_label}_{i}", embedding, metadata)])

            print(f"Stored {len(chunks)} chunks from {pdf_path} ({compliance_label}) in Pinecone!")

    def _retrieve_top_k(self,qna_pair, k=3):
        """For a given QnA pair retrieve k similar chunks from pinecode db based on cosine similarity."""
        query_embedding = self.model.encode(qna_pair).tolist()
        results = self.index.query(vector=query_embedding, top_k=k, include_metadata=True)
        # Check if matches exist in the response
        if "matches" not in results or not results["matches"]:
            print("No matches found.")
            return []
        # Extract metadata from matches
        retrieved_chunks = [(match["metadata"]["text"], match["metadata"]["compliance"]) for match in results["matches"]]
        return retrieved_chunks

    def classify_with_rag(self,q_id,qna_pair, top_k=10):
        """Uses retrieved compliance chunks in a RAG pipeline to classify QnA."""
        top_chunks = self._retrieve_top_k(qna_pair, k=top_k)
        print("Succesfully Fetched chunk \n")
       
        # Format retrieved context into label and document text
        context_text = "\n".join([f"- {label}: {chunk}" for chunk, label in top_chunks])


       
        response = self.client.chat.completions.create(
                model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
                messages=[
                    {"role": "system", "content": "You are an AI compliance expert."},
                    {"role": "user", "content": f"""
                        Given the following compliance standards content:

                        {context_text}

                        Classify the following question into one of these compliance standards based on the compliance standard content.

                        Only return the compliance name.

                        Q: {qna_pair}
                    """}
                ]
            )

        # Validate Response
        if not response or not response.choices:
            return {"error": "No response from AI model"}, 500

        compliance = response.choices[0].message.content.strip()

        # Handle Empty or Unexpected Response
        if not compliance:
            return {"error": "Model did not return a compliance category"}, 400


        try:
            query = """
                MERGE (q:QnAPair {id: $qna_id})
                MERGE (sc:Compliance {name: $compliance})
                MERGE (q)-[:FALLS_UNDER]->(sc)
                RETURN q, sc
            """
            result = self.db.query(query, parameters={"qna_id": q_id, "compliance": compliance})
        except Exception as e:
            print(f"Error linking QnA Pair to SubCategory: {e}")
        return compliance
    def count_each_compliance(self):
        try:
            query = """
                MATCH (q:QnAPair)-[:FALLS_UNDER]->(sc:Compliance)
                RETURN sc.name AS compliance, COUNT(q) AS total_questions
                ORDER BY total_questions DESC
            """
            result = self.db.query(query)
            return [{ "compliance": record["compliance"], "total_questions": record["total_questions"]} for record in result]
        except Exception as e:
            print(f"Error linking QnA Pair to SubCategory: {e}")


if __name__ =="__main__":
    pdf_compliance_map = {
        "iso_27001.pdf": "ISO 27001",
        "gdpr.pdf": "GDPR",
        "hippa.pdf": "HIPAA"
    }
    cr= Complaince_RAG()
    print("Created a new RAG \n")
    # Store all PDFs with compliance labels
    for pdf, label in pdf_compliance_map.items():
        pdf= "./Backend/Compliance/Policy/"+pdf
        cr.store_chunks_in_pinecone(pdf, label)
    print("Succesfully Created chunk \n")