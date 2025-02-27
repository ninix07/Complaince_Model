## Ways to Recreate the software

---

### Backend

---

1. Create a virtual environment and run
   ` pip install -r requirements.txt`
2. Setup a CONFIG.env file with `Together AI`, `Pinecone DB` API and `Neo4j` connection with auth (username and password) in following foprmat :
   `PINECONE_API_KEY=<Your_API_KEY>
TOGETHER_AI_KEY=<YOUR_API_KEY>
NEO4j_USERNAME=<username>
NEO4j_PASSWORD=<password>
NEO4j_URI=<uri>`
3. If you haven't setup `Pinecone` chunks initially run:
   `python ./Backend/Compliance/Compliance
4. Once the setup is over run the backend as:
   `python ./Backend/app.py`

---

### Frontend

---

1. Go into the frontend folder.
2. Run following to install the dependencies:
   `npm i`
3. Start the frontend as:
   `npm run dev`

---

Make sure that your neo4j connection is running and is available.
