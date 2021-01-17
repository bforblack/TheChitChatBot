from flask import  Flask,request,jsonify
from NLP import  NLPEngine as engine
import facebook

app = Flask(__name__)


@app.route('/summary', methods=['GET','POST'])
def summary():
    dataJson=request.json
    c=engine.NlpEngineStart(dataJson['data'])
    return jsonify("responce",c.summary())

@app.route('/getData', methods=['GET','POST'])
def fetchData():
    token='EAA1X4k123sUBAAG6i75kuwRTUBhEHfisOnh5vNZCpYPw90NttAi0pte7cCFZAoIpEXlalFABe789hx4xprvCladCF9iJ7UUu8nn9P9rHrQg5HXUr9zDgxyWZCVYPoZAxn8xgALAllNmYTpBoP5mepiJzxA0ke9549LGNNZCo9yMma2oMCWjPN85Qgw7gVww9MeWm5NTKwPgZDZD'
    dataJson=request.json
    graph=facebook.GraphAPI(token)
    profile=graph.get_object('me',fields='friends')
    return  jsonify('respnce',profile)


app.run()
