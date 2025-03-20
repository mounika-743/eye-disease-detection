from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from model import load_model, predict

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

model = load_model('models/cnn_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        prediction = predict(model, filepath)
        return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)