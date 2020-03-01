from flask import Flask, render_template, request, redirect, send_from_directory, url_for
from PIL import Image
from algorithm import create_model
import numpy as np
import tensorflow as tf
import os
from werkzeug.utils import secure_filename
from keras.applications.resnet50 import preprocess_input


app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

UPLOAD_FOLDER = os.path.basename('static')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

global model
model = create_model()
global graph
graph = tf.get_default_graph()


def pre_process(img):
	img = Image.open(img)
	img = img.resize((224,224), Image.ANTIALIAS)
	img = np.expand_dims(np.array(img), 0)
	return img


@app.route('/')
@app.route('/index')
def index():
	return render_template("index.html")


@app.route('/uploads/<filename>')
def upload_file(filename):
	return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/result', methods=['POST'])
def result():
	if not os.path.exists(app.config['UPLOAD_FOLDER']):
		os.mkdir(app.config['UPLOAD_FOLDER'])

	file = request.files['image']
	if (file.filename == ""):
		return redirect("/")

	filename = file.filename
	filename = secure_filename(filename)
	file.save(os.path.join(app.config['UPLOAD_FOLDER'], file.filename))
	name, ext = os.path.splitext(filename)
	
	
	src = 'static/' + filename
	dst = 'static/testIMG.jpg'
	os.rename(src, dst)
	#image = pre_process(dst)
	
	img = Image.open("static/testIMG.jpg")
	validation_batch = np.stack([preprocess_input(np.array(img.resize((224, 224), Image.ANTIALIAS)))])
	plant = [
		"BrownSpot",
		"Healthy",
		"Hispa",
		"LeafBlast"
	]

	with graph.as_default():
		pred = model.predict(validation_batch)

	d = {}
	print(pred[0])
	

	# print(type(pred))

	for index, value in enumerate(pred[0]):
		d[plant[index]] = value
		#print(type(d), " ")

	treatment = {
		"BrownSpot": "Use resistant varieties. Use fungicides (e.g., iprodione, propiconazole, azoxystrobin, trifloxystrobin, and carbendazim) as seed treatments. Treat seeds with hot water (53−54°C) for 10−12 minutes before planting, to control primary infection at the seedling stage. To increase effectiveness of treatment, pre-soak seeds in cold water for eight hours.",
		"Healthy": "Your plant is healthy",
		"Hispa": "Always consider an integrated approach with preventive measures together with biological treatments if available. If infestations are high (>50%) during booting stage, spray flubendiamide @ 0.1ml or chlorantraniliprole @ 0.3ml/l of water. Other insecticides based on chlorpyriphos, chlorantraniliprole, indoxacarb, azadirachtin, gamma- or lamda-cyhalothrin are also helpful, particularly if infestation is severe. Other insecticides include alpha-cypermethrin, abamectin 2% to kill the larvae. Care should be taken not to use chemicals causing resurgence of the insect.",
		"LeafBlast": "Apply fungicides during the time frame predicted by the DD50 program,(opens in new window) which is about 5 to 7 days before heading (late boot stage). Fungicides are especially needed if blast symptoms have been observed in the field and the variety is very susceptible. Fungicides should be applied a second time about two days after 50 percent heading (90 percent head exsertion). In uniform stands, 90 percent heading will occur in 4 to 5 days after the first heads are visible."
	}
	treatment_technique = treatment[plant[np.argmax(pred)]]
	diseasePred = plant[np.argmax(pred)]
	plant_name = 'Rice'
	
	#print(plant_name)
	plant_img = dst
	
	return render_template('result.html', plant_name=plant_name, treatment_technique=treatment_technique,
						   dictionary=d, plant_img=plant_img, using_algo = False, disease = diseasePred)

	# return render_template('index.html')


if __name__ == '__main__':
	#app.run(debug=True)
	#app.run(host='0.0.0.0')
	app.run(extra_files=['templates/index.html'])
	app.config['TEMPLATES_AUTO_RELOAD'] = True

if app.config["DEBUG"]:
	@app.after_request
	def after_request(response):
		response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, public, max-age=0"
		response.headers["Expires"] = 0
		response.headers["Pragma"] = "no-cache"
		return response
