import numpy as np
import nltk
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score
import torch

class Taxonomy:
    def __init__(self, db, file_path="./Backend/Taxonomy/data.csv"):
        self.db=db
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('punkt_tab')
        self.model = BertModel.from_pretrained("bert-base-uncased")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        # temporary before database setup
        self.file_path = file_path
        # Possible Cybersec Categories
        self.CYBERSEC_CATEGORIES = {
            "Access Control": ["Authentication", "Authorization", "Role-Based Access"],
            "Data Encryption": ["Symmetric Encryption", "Asymmetric Encryption", "Key Management"],
            "Incident Response": ["Threat Detection", "Breach Management", "Forensics"],
            "Network Security": ["Firewalls", "Intrusion Detection", "VPNs"],
            "Application Security": ["Secure Coding", "Vulnerability Management", "Penetration Testing"],
            "Cloud Security": ["Identity Management", "Data Protection", "Compliance"],
            "Endpoint Security": ["Antivirus", "Device Control", "Patch Management"],
            "Identity and Access Management": ["Single Sign-On", "Multi-Factor Authentication", "Biometric Security"],
            "Compliance and Governance": ["ISO 27001", "GDPR", "NIST Framework"],
            "Threat Intelligence": ["Indicators of Compromise", "Threat Feeds", "Adversary Tracking"],
            "Security Operations": ["SOC", "SIEM", "Incident Handling"],
            "Physical Security": ["CCTV Monitoring", "Access Cards", "Security Guards"],
            "Risk Management": ["Threat Modeling", "Risk Assessment", "Business Continuity"],
            "DevSecOps": ["CI/CD Security", "Secure Infrastructure as Code", "Automated Security Testing"],
            "Zero Trust Security": ["Microsegmentation", "Least Privilege", "Continuous Verification"],
            "Security Awareness Training": ["Phishing Simulations", "Cyber Hygiene", "User Education"],
            "Mobile Security": ["App Sandboxing", "Device Encryption", "Secure Containers"],
            "IoT Security": ["Device Authentication", "Secure Firmware", "Anomaly Detection"],
            "Supply Chain Security": ["Third-Party Risk", "Software Bill of Materials", "Vendor Security Audits"],
            "Cybersecurity Frameworks": ["MITRE ATTACK", "NIST Cybersecurity Framework", "CIS Controls"]
        }

        # Model and submodel for clustering to create taxonomy
        self.clustering = None
        self.new_qna_count =0 
        self.text_embeddings={}
        # Read dataset
        
        self.silhouette_scores=[]
        self.davies_bouldin_scores=[]
        self.sub_silhouette_scores=[]
        self.sub_davies_bouldin_scores=[]

    def _read_data(self,file_path=None):
        """
        From the given path/database read the data and then store it as necessary.
        """
        qna_pairs = {}
        if file_path:
            self.file_path=file_path
        df = pd.read_csv(self.file_path)
        df["Combined Text"] = df["Question Text"].fillna("") + " " + df["Answer"].fillna("") + " " + df["Notes/Comment"].fillna("")
        qna_pairs = df.set_index("Original ID")["Combined Text"].to_dict()
        return qna_pairs
    def _remove_all_db(self):
        try:
            query = """
                Match(n) Delete(n)
              """
            self.db.query(query)
            
        except Exception as e:
            print(f"Error Deteced while deleting existing relationship: {e}")
    def _create_category_db(self, category_name):
            try:
                query = """
                    MERGE (c:Category {name: $category_name})
                    RETURN c
                """
                result = self.db.query(query, parameters={"category_name": category_name})
                return result
            except Exception as e:
                print(f"Error creating category: {e}")
    def _create_sub_category_db(self, category_name,sub_category):
            
            try:
                query = """
                        MERGE (c:Category {name: $category_name})
                        MERGE (sc:SubCategory {name: $sub_category})
                        MERGE (sc)-[:BELONGS_TO]->(c)
                        RETURN sc, c
                    """
                result = self.db.query(query, parameters={"sub_category": sub_category, "category_name": category_name})
                return result
            except Exception as e:
                print(f"Error creating category: {e}")
    def _create_qna_sub_category_db(self, sub_category, qid):
        try:
            query = """
                MERGE (q:QnAPair {id: $qna_id})
                MERGE (sc:SubCategory {name: $sub_category})
                MERGE (q)-[:CATEGORIZES_TO]->(sc)
                RETURN q, sc
            """
            result = self.db.query(query, parameters={"qna_id": qid, "sub_category": sub_category})
            return result
        except Exception as e:
            print(f"Error linking QnA Pair to SubCategory: {e}")
    def generate_category_name(self, embedding):
        """Generates category name using the most frequent assigned category."""
        # categories = [self.assign_category(qna) for _, qna in qna_list[:2]]
        first_embedding = embedding[0][1]
        categories = self.assign_category(first_embedding.reshape(1, -1))
        # return Counter(categories).most_common(1)[0][0] 
        return categories
    
    def generate_sub_category_name(self, category_name, embedding):
        """Generates sub category name using the most frequent assigned sub category."""
        # categories = [self.assign_sub_category(category_name,qna) for _, qna in qna_list[0]]
        first_embedding = embedding[0][1]
        categories = self.assign_sub_category(category_name,first_embedding.reshape(1, -1).reshape(1, -1))
        # return Counter(categories).most_common(1)[0][0] 
        return categories

    def get_bert_embedding(self, text):
        """Converts text into BERT embeddings."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()
    
    def get_bert_embedding_batched(self, texts, batch_size=128):
            """Converts multiple texts into BERT embeddings in batches."""
            all_embeddings = {}
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, return_tensors='pt', truncation=True, padding=True)
                with torch.no_grad():  
                    outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :].detach().numpy()
                for j,text in enumerate(batch):
                        all_embeddings[text]=embeddings[j]
            return all_embeddings
    
    def assign_category(self, text_embedding):
        """Assigns a category to a given text based on cosine similarity with predefined cybersecurity categories."""
        category_similarities = {
            category: np.max([cosine_similarity(text_embedding, self.get_bert_embedding(sub_cat)).flatten()[0] 
                               for sub_cat in sub_cats])
            for category, sub_cats in self.CYBERSEC_CATEGORIES.items()
        }
        return max(category_similarities, key=category_similarities.get)
    def assign_sub_category(self, category, text_embedding):
        """Assigns a subcategory to a given category and text based on cosine similarity with predefined cybersecurity categories."""
        sub_categories = self.CYBERSEC_CATEGORIES[category]
        category_similarities = {
            sub_cat: cosine_similarity(text_embedding, self.get_bert_embedding(sub_cat)).flatten()[0]  
            for sub_cat in sub_categories  
        }

        return max(category_similarities, key=category_similarities.get) 

    def create_taxonomy(self,f):
        """
        Creates a taxonomy based on clustering methods. Firstly, embeddings are generated and a top-level clustering is formed. 
        Then, each cluster is divided into further sub-clusters. Each cluster is assigned a category name and sub-category name.
        Returns a dictionary representing the taxonomy.
        """
        self.qna_pairs= self._read_data(f)
        self._remove_all_db()
        self.category_map = defaultdict(list)
        self.label_to_category_name = {}
        self.sub_nodes_map = defaultdict(lambda: defaultdict(list))
        self.sub_label_to_sub_category = defaultdict(dict)

        print("----Creating Taxonomy-----\n")
        
        qna_ids, qna_pairs = zip(*self.qna_pairs.items()) 
        if not self.text_embeddings:
            self.text_embeddings = self.get_bert_embedding_batched(list(qna_pairs))
        embeddings = np.vstack(list(self.text_embeddings.values()))
        embeddings = normalize(embeddings)

        print("----Embeddings Created-----\n")

        # Perform top-level clustering
        self.clustering = AgglomerativeClustering(distance_threshold=0.15, n_clusters=None, metric="cosine", linkage="average")
        cluster_labels = self.clustering.fit_predict(embeddings)


        for qna_id,embedding, label in zip(qna_ids,embeddings,cluster_labels):
            self.category_map[label].append((qna_id,embedding))


        # Compute clustering scores
        unique_top_clusters = len(set(cluster_labels))
        if 1 < unique_top_clusters < len(self.qna_pairs):
            self.top_silhouette = silhouette_score(embeddings, cluster_labels, metric="cosine")
            self.top_davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        else:
            self.top_silhouette, self.top_davies_bouldin = None, None
        
        # Assign category names to labels
        for label in self.category_map.keys():
            category_name = self.generate_category_name(self.category_map[label]) if self.category_map[label] else f"Category {label}"
            self.label_to_category_name[label] = category_name

        for label,  embeddings in self.category_map.items():
                if len(embeddings) < 2:
                  best_similarity = 0.0
                  best_label = None
                  for other_label, other_embeddings in self.category_map.items():
                    if len(other_embeddings) >1:
                        if other_label == label or self.label_to_category_name[label] ==self.label_to_category_name[other_label]:
                            continue
                          
                            
                        # Compute similarity between the single-QnA and the first QnA in the other cluster
                        similarity = cosine_similarity(
                            embeddings[0][1].reshape(1, -1),
                            other_embeddings[0][1].reshape(1, -1)
                        ).flatten()[0]

                        if similarity > best_similarity:  # Keep track of the best match
                            best_similarity = similarity
                            best_label = other_label
                        elif similarity > 0.6:
                          self.label_to_category_name[label] = self.label_to_category_name[other_label]
                          best_label=None
                          break

                    # Merge into the best cluster if similarity is high enough
                  if best_label :
                        self.label_to_category_name[label] = self.label_to_category_name[other_label]
                        
        print("----First Level Nodes Categorization Successful-----\n")

        for label, category_name in self.label_to_category_name.items():
            self._create_category_db(category_name)
        # Perform sub-clustering within each top-level category
        self.sub_silhouette_scores = []
        self.sub_davies_bouldin_scores = []
        print("----Starting Second Level Nodes Categorization -----\n")

        for label, sub_embeddings in self.category_map.items():
            if len(sub_embeddings) < 2:
                continue   
            embeddings=[embedding for _, embedding in sub_embeddings]
            sub_qna_ids=[qid for qid,_ in sub_embeddings]
            sub_clustering = AgglomerativeClustering(distance_threshold=0.1, n_clusters=None, metric="cosine", linkage="average")
            sub_cluster_labels = sub_clustering.fit_predict(embeddings)

            unique_sub_clusters = len(set(sub_cluster_labels))
            if 1 < unique_sub_clusters < len(sub_embeddings):
                self.sub_silhouette_scores.append(silhouette_score(embeddings, sub_cluster_labels, metric="cosine"))
                self.sub_davies_bouldin_scores.append(davies_bouldin_score(embeddings, sub_cluster_labels))
            
            
            sub_label_to_qnas = defaultdict(list)
            for sub_qna_ids,sub_embedding, sub_label in zip(sub_qna_ids,embeddings, sub_cluster_labels):
                    sub_label_to_qnas[sub_label].append((sub_qna_ids,sub_embedding))
                

            # Generate sub-category names
            for sub_label, sub_embeddings in sub_label_to_qnas.items():
                category_name = self.label_to_category_name[label]
                if sub_label not in self.sub_label_to_sub_category[label]:
                    sub_category_name = self.generate_sub_category_name(category_name, sub_embeddings)
                    self.sub_label_to_sub_category[label][sub_label] = sub_category_name
                self.sub_nodes_map[label][sub_label] = sub_embeddings  
            category_name = self.label_to_category_name[label]
            for sub_label, sub_category_name in self.sub_label_to_sub_category[label].items():
                self._create_sub_category_db(category_name,sub_category_name)

            for sub_label, sub_embeddings in sub_label_to_qnas.items():
                sub_category_name=self.sub_label_to_sub_category[label][sub_label] 
                for q_id , _ in sub_embeddings:
                    self._create_qna_sub_category_db( sub_category_name, q_id)
        print("----Subcategory Assignment Successful-----\n")
        # Construct the final taxonomy dictionary
        self.taxonomy_dict = {}

        for label, category_name in self.label_to_category_name.items():
            if category_name not in self.taxonomy_dict:
                self.taxonomy_dict[category_name] = []  # Initialize as a list instead of a dictionary

            if label in self.sub_nodes_map:
                for sub_category_label, _ in self.sub_nodes_map[label].items():
                    sub_category_name = self.sub_label_to_sub_category[label][sub_category_label]

                    # Add sub-category name only if it's not already in the list
                    if sub_category_name not in self.taxonomy_dict[category_name]:
                        self.taxonomy_dict[category_name].append(sub_category_name)

        return self.taxonomy_dict 


    def evaluate_taxonomy(self):
        """
        Computes the average silhouette and Davies-Bouldin scores for sub-categories.
        """
        if not self.clustering:
            print("Error: Taxonomy has not been created. Call create_taxonomy() first.")
            return None

        # Compute average sub-category scores
        avg_sub_silhouette = (
            sum(self.sub_silhouette_scores) / len(self.sub_silhouette_scores) if self.sub_silhouette_scores else None
        )
        avg_sub_davies_bouldin = (
            sum(self.sub_davies_bouldin_scores) / len(self.sub_davies_bouldin_scores) if self.sub_davies_bouldin_scores else None
        )

        return {
            "Top-Level": {
                "Silhouette Score": self.top_silhouette,
                "Davies-Bouldin Index": self.top_davies_bouldin
            },
            "Sub-Categories": {
                "Average Silhouette Score": avg_sub_silhouette,
                "Average Davies-Bouldin Index": avg_sub_davies_bouldin
            }
        }
    def predict_category( self,new_q_id, new_text):
            """
            Predicts the most suitable category and sub-category for a given QnA pair.
            """
            if not self.clustering:
                print("Taxonomy has not been created yet. Run create_taxonomy() first.")
                return "Uncategorized", "Uncategorized"
            self.new_qna_count += 1 
            if self.new_qna_count >= 100 :
                print("Threshold reached. Rebuilding taxonomy...")

                self.create_taxonomy()
                self.new_qna_count = 0 
            #Embed the new QnA pair
            self.qna_pairs[new_q_id]=new_text
            new_embedding_un = self.get_bert_embedding(new_text)
            new_embedding = normalize(new_embedding_un.reshape(1, -1))

            # Find the best matching category
            category_similarities = {}
            for label, embedding in self.category_map.items():
                avg_similarity = np.mean(cosine_similarity(new_embedding, embedding))
                category_similarities[label] = avg_similarity
            if not category_similarities:
                return "Uncategorized", "Uncategorized"
            best_category_label = max(category_similarities, key=category_similarities.get)
            if best_category_label not in self.sub_nodes_map:
              best_category_name = self.label_to_category_name[best_category_label]

              # Find all labels that have the same category name
              matching_labels = [
                  label for label, name in self.label_to_category_name.items()
                  if name == best_category_name
              ]
              for label in matching_labels:
                if label in self.sub_nodes_map:
                  best_category_label = label
                  break 
            if best_category_label in self.sub_nodes_map:
                sub_category_similarities = {}
                for sub_category_label, sub_embedding in self.sub_nodes_map[best_category_label].items():
                    avg_similarity = np.mean(cosine_similarity(new_embedding, sub_embedding))
                    sub_category_similarities[sub_category_label] = avg_similarity
               
                best_sub_category_label = max(sub_category_similarities, key=sub_category_similarities.get)
                
                similarity= sub_category_similarities[max(sub_category_similarities)]
                self.category_map[best_category_label].append((new_q_id, new_text))
                if similarity < 0.4:
                  self.create_taxonomy()
                  return self.predict(new_q_id,new_text)
            self.text_embeddings[new_text] = new_embedding_un
            self.qna_pairs[new_q_id]= new_text
            if best_category_label in self.sub_nodes_map:
                self.sub_nodes_map[best_category_label][best_sub_category_label].append(new_embedding)
                best_category_name= self.label_to_category_name[best_category_label]
                best_sub_category_name = self.sub_label_to_sub_category[best_category_label][best_sub_category_label]
                return best_category_name, best_sub_category_name       
            return best_category_name, "Uncategorized"
            

   
