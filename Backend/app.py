from flask import Flask, jsonify,request
from Compliance.Compliance import Complaince_RAG
from Taxonomy.Taxonomy import Taxonomy

app = Flask(__name__)
taxonomy= Taxonomy()
# taxonomy.create_taxonomy()
complaince= Complaince_RAG()
@app.route('/')
def home():
    return "<h1>Welcome to My Flask App</h1><p>This is a simple Flask application.</p>"

@app.route('/api/getTaxonomy')
def get_taxonomy():
    print("API Hit \n")
    return jsonify(taxonomy.taxonomy_dict)

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
    app.run(debug=True)
