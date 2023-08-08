from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from imageCaptioning.pipeline.dataset_02 import DatasetPipeline
from imageCaptioning.components.dataset import ImageCaptionDataset
from imageCaptioning.pipeline.train_03 import TrainPipeline
from imageCaptioning import logger
import os
app = Flask(__name__)

# Specify the directory where uploaded files will be stored
app.config['UPLOAD_FOLDER'] = 'artifacts/predict'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Check if the uploaded file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/', methods=['GET', 'POST'])
def test():
    dataset_pipeline = DatasetPipeline([])
    train_pipeline = TrainPipeline(None, ImageCaptionDataset('', [], 32, -1, dataset_pipeline.val_transform, 'validation'))
    caption = ''
    if request.method == 'POST':
        if 'image' in request.files:
            image = request.files['image']
            if image and allowed_file(image.filename):
                filename = secure_filename(image.filename)
                image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                caption = train_pipeline.predict(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'model-20.pth')
    return render_template('index.html', caption=caption)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)