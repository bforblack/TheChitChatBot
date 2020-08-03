from flask import  Flask,request,jsonify
from NLP import  NLPEngine as engine

app = Flask(__name__)


@app.route('/summary', methods=['GET','POST'])
def summary():
    dataJson=request.json
    c=engine.NlpEngineStart(dataJson['data'])
    return c.summary()

app.run()
