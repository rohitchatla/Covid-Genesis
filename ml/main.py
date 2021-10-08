from flask import * 
from flask import jsonify
# import sentiment_mod as s
import os
# import main_ocr_aadhar as ocr
import covid_19_symptom_analysis as sym
# import symptoms-covid-19-using-7-machine-learning-98 as sym2
import symptoms_covid_19_using_7_machine_learning_98 as sym2
import xray as xy

app = Flask(__name__)  
#cors = CORS(app) #,resources={r"/api/*": {"origins": "*"}}

@app.route('/symsroutine',methods=['POST', 'GET']) #/1:routinez
def symptoms(): #sentiment
	if request.method == 'POST':
		# data = request.get_json()
		# print(data["text"])
		# return jsonify(s.sentiment(data["text"]))
		data = request.get_json()
		# print(type(data))
		# print(sym2.check(data))
		return jsonify(sym2.check(data))
	if request.method == 'GET':
		# return sym.routi()
		# return sym.check()
		# print(sym2.check())
		return jsonify(sym2.check())
		# return sym2.check()
@app.route('/xray',methods=['POST', 'GET'])  #/2
def xrayfunc(): #aadhar_ocr
	if request.method == 'POST':
		# data = request.get_json()
		# im_b64=data["payload"]
		# return jsonify(ocr.main(data["text"],im_b64))
		data = request.get_json()
		return xy.check(data["b64url"])
	if request.method == 'GET':
		# return xray.routi()
		return xy.check()

port = int(os.environ.get("PORT", 5002))
if __name__ == '__main__':  
   app.run(debug = True, host='0.0.0.0', port=port)  