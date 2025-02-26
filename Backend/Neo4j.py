from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import atexit

config_path = os.path.join(os.path.dirname(__file__),'CONFIG.env')
load_dotenv(config_path)
print(config_path)

class ConnectNeo4j:
    _instance = None

    def __new__(cls):
        URI = os.getenv("NEO4j_URI")
        username = os.getenv("NEO4j_USERNAME")
        password = os.getenv("NEO4j_PASSWORD")
        if cls._instance is None:
            cls._instance = super(ConnectNeo4j, cls).__new__(cls)
            cls._instance._driver = GraphDatabase.driver(URI, auth=(username, password))
            print(f"Successfully connected to Neo4j at {URI}")
            atexit.register(cls._instance.close)  
        return cls._instance
    
    def close(self):
        if self._driver:
            self._driver.close()
    
    def query(self, query,parameters=None):
        with self._driver.session() as session:
            return session.run(query, parameters).data()
        

if __name__ =="__main__":
    db= ConnectNeo4j()