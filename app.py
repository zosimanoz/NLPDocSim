from flask import Flask
from document_similarity.routes import doc_sim
import config

app = Flask(__name__)
app.config.from_object(config)

app.register_blueprint(doc_sim)

if __name__ == "__main__":
    app.run(debug=True)
