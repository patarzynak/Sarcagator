from flask import Flask, render_template, request, jsonify
from sarcagator import reply
app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact")
def contact():
    return render_template("contact.html")

@app.route("/getsarky")
def getsarky():
    # calculating much sarcastic response...
    app.logger.info("user entered <%s>" % request.args["sentence"])
    sentence = request.args["sentence"]
    # ,___,
    # {O,o}
    # |)``)
    response = reply(sentence)
    return jsonify({"status": "OWL_GIVEN", "response": response})

if __name__ == "__main__":
    app.run(debug=True)

