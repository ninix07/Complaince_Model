from flask import Flask, jsonify,request
from Compliance.Compliance import Complaince_RAG
from Taxonomy.Taxonomy import Taxonomy
from Neo4j import ConnectNeo4j
app = Flask(__name__)

db= ConnectNeo4j()
taxonomy= Taxonomy(db)
complaince= Complaince_RAG()

@app.route('/')
def home():
    return "<h1>Flask Application Running</p>"

@app.route('/api/createTaxonomy',methods=["POST"])
def createTaxonomy():
    f = request.files.get('file')

    try:
        taxonomy.create_taxonomy(f)  
        return jsonify({"message": "Taxonomy created successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/getTaxonomy')
def get_taxonomy():
    print("API Hit \n")
    return jsonify(taxonomy.taxonomy_dict)

@app.route('/api/predictCategory',methods=["POST"])
def get_category():
    data = request.get_json()  
    if not data or "question" not in data or"q_id" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    question = data["question"]
    q_id= data["q_id"]
    category,sub_category = taxonomy.predict_category( q_id, question)
    response = {"Category": category, "Sub-Category": sub_category}
    return jsonify(response), 200 

@app.route('/api/getComplaince',methods=["POST"])
def get_Complaince():
    data = request.get_json()  
    if not data or "question" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    question = data["question"]
    classification = complaince.classify_with_rag(question)
    response = {"Compliance": classification}
    
    return jsonify(response), 200 



if __name__ == '__main__':
    app.run(debug=True,port=8080)
