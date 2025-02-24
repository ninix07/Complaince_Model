import numpy as np
import nltk
from transformers import BertTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from anytree import Node, RenderTree
from collections import defaultdict, Counter
import pandas as pd
import spacy 
from sklearn.metrics import silhouette_score, davies_bouldin_score
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
spacy.load("en_core_web_sm")

class Taxonomy:
    def __init__(self, file_path="./Backend/Taxonomy/QnApairs_updated.csv"):
        self.nlp = spacy.load("en_core_web_sm")
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
        self.top_nodes = {}
        self.sub_nodes_map = defaultdict(lambda: defaultdict(list))
        self.taxonomy_root = Node("Root")

        # Read dataset
        self.qna_pairs, self.df = self._read_data()
        self.silhouette_scores=[]
        self.davies_bouldin_scores=[]
        self.sub_silhouette_scores=[]
        self.sub_davies_bouldin_scores=[]
    def _read_data(self):
        """
        From the given path/database read the data and then store it as necessary.
        """
        df = pd.read_csv(self.file_path)
        df["Combined Text"] = df["Question Text"].fillna("") + " " + df["Answer"].fillna("") + " " + df["Notes/Comment"].fillna("")
        qna_pairs = df.set_index("Original ID")["Combined Text"].to_dict()
        return qna_pairs, df

    
    def generate_category_name(self, qna_list):
        """Generates category name using the most frequent assigned category."""
        categories = [self.assign_category(qna) for _, qna in qna_list[:5]]
        return Counter(categories).most_common(1)[0][0] 
    def generate_sub_category_name(self, category_name, qna_list):
        """Generates sub category name using the most frequent assigned sub category."""
        categories = [self.assign_sub_category(category_name,qna) for _, qna in qna_list[:5]]
        return Counter(categories).most_common(1)[0][0] 

    def get_bert_embedding(self, text):
        """Converts text into BERT embeddings."""
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state[:, 0, :].detach().numpy()

    def assign_category(self, text):
        """Assigns a category to a given text based on cosine similarity with predefined cybersecurity categories."""
        text_embedding = self.text_embeddings[text]
        category_similarities = {
            category: np.mean([cosine_similarity(text_embedding, self.get_bert_embedding(sub_cat)).flatten()[0] 
                               for sub_cat in sub_cats])
            for category, sub_cats in self.CYBERSEC_CATEGORIES.items()
        }
        return max(category_similarities, key=category_similarities.get)
    def assign_sub_category(self, category, text):
        """Assigns a subcategory to a given category and text based on cosine similarity with predefined cybersecurity categories."""
        sub_categories = self.CYBERSEC_CATEGORIES[category]
        text_embedding = self.text_embeddings[text]

        category_similarities = {
            sub_cat: cosine_similarity(text_embedding, self.get_bert_embedding(sub_cat)).flatten()[0]  
            for sub_cat in sub_categories  
        }

        return max(category_similarities, key=category_similarities.get) 


    


    def create_taxonomy(self):
        """
        Creates a taxonomy based on clusetring method. Firstly, the embeddings are generated and a top level clustering is formed. 
        And then each clusters are divided into further sub-clusters. Each cluster is then assigned with a category name and sub-category name.
        Returns a dictionary for the taxonomy.
        """
        
        # Compute BERT embeddings for semantic preservation
        self.text_embeddings = {text: self.get_bert_embedding(text) for text in self.qna_pairs.values()}
        embeddings = np.vstack(list(self.text_embeddings.values()))
        embeddings = normalize(embeddings)
        
        # Perform top-level clustering
        self.clustering = AgglomerativeClustering(distance_threshold=0.1, n_clusters=None, metric="cosine", linkage="average")
        cluster_labels = self.clustering.fit_predict(embeddings)
        
        
        self.category_map = defaultdict(list)
        for (q_id, text), label in zip(self.qna_pairs.items(), cluster_labels):
            self.category_map[label].append((q_id, text))
        
        # Compute clustering scores
        unique_top_clusters = len(set(cluster_labels))
        if 1 < unique_top_clusters < len(self.qna_pairs):
            self.top_silhouette = silhouette_score(embeddings, cluster_labels, metric="cosine")
            self.top_davies_bouldin = davies_bouldin_score(embeddings, cluster_labels)
        else:
            self.top_silhouette, self.top_davies_bouldin = None, None
        
        # Initialize storage for taxonomy
        self.top_nodes = {}
        self.sub_nodes_map = defaultdict(lambda: defaultdict(list))
        
        # Create top-level category nodes
        label_to_category_name = {}
        for label, qna_list in self.category_map.items():
            category_name = self.generate_category_name(qna_list) if qna_list else f"Category {label}"
            self.top_nodes[label] = Node(category_name, parent=self.taxonomy_root)
            label_to_category_name[label] = category_name
        
        # Perform sub-clustering within each top-level category
        self.sub_silhouette_scores = []
        self.sub_davies_bouldin_scores = []
        
        for label, qna_list in self.category_map.items():
            if len(qna_list) < 2:
                continue  
            
            sub_embeddings = np.vstack([self.get_bert_embedding(text) for _, text in qna_list])
            sub_embeddings = normalize(sub_embeddings)
            
            sub_clustering = AgglomerativeClustering(distance_threshold=0.05, n_clusters=None, metric="cosine", linkage="average")
            sub_cluster_labels = sub_clustering.fit_predict(sub_embeddings)
            
            unique_sub_clusters = len(set(sub_cluster_labels))
            if 1 < unique_sub_clusters < len(qna_list):
                self.sub_silhouette_scores.append(silhouette_score(sub_embeddings, sub_cluster_labels, metric="cosine"))
                self.sub_davies_bouldin_scores.append(davies_bouldin_score(sub_embeddings, sub_cluster_labels))
            
            sub_label_to_qnas = defaultdict(list)
            for (q_id, text), sub_label in zip(qna_list, sub_cluster_labels):
                sub_label_to_qnas[sub_label].append((q_id, text))

            # Generate sub-category names and assign QnA pairs
            sub_label_to_name = {}
            for sub_label, qna_list in sub_label_to_qnas.items():
                category_name = label_to_category_name[label]
    
                sub_label_to_name[sub_label] = self.generate_sub_category_name(category_name, qna_list)
            
                # Assign QnA pairs to the sub-category
                sub_category_name = sub_label_to_name[sub_label]
                self.sub_nodes_map[label][sub_category_name].extend(qna_list)
        
        # Construct the final taxonomy dictionary
        taxonomy_dict = {}

        for label in self.top_nodes:
            category_name = self.top_nodes[label].name
            if category_name not in taxonomy_dict:
                taxonomy_dict[category_name] = {}
            if label in self.sub_nodes_map:
                for sub_category_name, sub_qna_list in self.sub_nodes_map[label].items():
                    if sub_category_name not in taxonomy_dict[category_name]:
                        taxonomy_dict[category_name][sub_category_name] = []
                    taxonomy_dict[category_name][sub_category_name].extend([q_id for q_id, _ in sub_qna_list])
        return taxonomy_dict


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

            
    
    def predict_category(self, new_text):
            """
            Predicts the most suitable category and sub-category for a given QnA pair.
            """
            if not self.top_nodes:
                print("Taxonomy has not been created yet. Run create_taxonomy() first.")
                return "Uncategorized", "Uncategorized"

            # Embed the new QnA pair
            new_embedding = self.get_bert_embedding(new_text)
            new_embedding = normalize(new_embedding.reshape(1, -1))

            # Find the best matching category
            category_similarities = {}
            for label, qna_list in self.category_map.items():
                category_embeddings = np.vstack([self.text_embeddings[text] for _, text in qna_list])
                avg_similarity = np.mean(cosine_similarity(new_embedding, category_embeddings))
                category_similarities[label] = avg_similarity

            best_category_label = max(category_similarities, key=category_similarities.get)
            best_category_name = self.top_nodes[best_category_label].name

            # Find the best sub-category 
            if best_category_label in self.sub_nodes_map:
                sub_category_similarities = {}

                for sub_category_name, sub_qna_list in self.sub_nodes_map[best_category_label].items():
                    sub_category_embeddings = np.vstack([self.text_embeddings[text] for _, text in sub_qna_list])
                    avg_similarity = np.mean(cosine_similarity(new_embedding.reshape(1, -1), sub_category_embeddings))
                    sub_category_similarities[sub_category_name] = avg_similarity

                if sub_category_similarities:
                    best_sub_category_name = max(sub_category_similarities, key=sub_category_similarities.get)
                else:
                    best_sub_category_name = "Uncategorized"

                return best_category_name, best_sub_category_name
            else:
                return best_category_name, "Uncategorized"

   
if __name__== "__main__":

    tax= Taxonomy()
    print(tax.create_taxonomy())
    test_text = "How do we secure cloud data?"
    print(tax.predict_category(test_text))
    print(tax.evaluate_taxonomy())