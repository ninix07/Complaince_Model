from flask import Flask, jsonify,request
from Compliance.Compliance import Complaince_RAG
from Taxonomy.Taxonomy import Taxonomy
from Neo4j import ConnectNeo4j
from flask_cors import CORS
app = Flask(__name__)
CORS(app)
try:
    db= ConnectNeo4j()
except Exception as err:
    print(f"Database error: {err}")
taxonomy= Taxonomy(db)
complaince= Complaince_RAG(db)

@app.route('/')
def home():
    return "<h1>Flask Application Running</p>"

@app.route('/api/createTaxonomy',methods=["POST"])
def createTaxonomy():
    f = request.files.get('file')
    if f==None:
        return jsonify({"error": "No file attacheed."}), 500
    try:
        taxonomy.create_taxonomy(f)  
        return jsonify({"message": "Taxonomy created successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/getTaxonomy')
def get_taxonomy():
    print("API Hit \n")
    return jsonify({"Data":taxonomy.taxonomy_dict})

@app.route('/api/predictCategory',methods=["POST"])
def get_category():
    data = request.get_json()  
    if not data or "question" not in data or"q_id" not in data:
        return jsonify({"error": "Missing 'question' in request body"}), 400
    question = data["question"]
    q_id= data["q_id"]
    prediction= taxonomy.predict_category( q_id, question)
    if prediction=="error":
        return jsonify({"error": "Taxonomy has not been created yet. Run create_taxonomy() first."}), 400
    
    response = {"Category": prediction[0], "Sub_Category": prediction[1]}
    return jsonify(response), 200 

@app.route('/api/getComplaince',methods=["POST"])
def get_Complaince():
    data = request.get_json() 
    print(data)
    if not data or "question" not in data or "q_id" not in data or not data["question"] or not data["q_id"]:
        return jsonify({"error": "Missing 'question or q_id' in request body"}), 400
    question = data["question"]
    q_id= data["q_id"]
    classification = complaince.classify_with_rag(q_id,question)
    response = {"Compliance": classification}
    
    return jsonify(response), 200 

@app.route('/api/countCompliance')
def count_Compliance():
    required_compliances = {"HIPAA", "ISO 27001", "GDPR"}
    count = complaince.count_each_compliance()
    if not count:
        return jsonify({"error": "No complaince map data in the database"}), 400
    existing_compliances = {entry["compliance"] for entry in count}
    for compliance in required_compliances:
        if compliance not in existing_compliances:
            count.append({"compliance": compliance, "total_questions": 0})
    return jsonify(count)




if __name__ == '__main__':
    app.run(debug=True,port=8080)
