# Imports
from flask import Flask, request, render_template, jsonify, Markup
from numpy import uint8, array, argsort, fliplr, asarray
from cv2 import imdecode, cvtColor, COLOR_BGR2RGB, IMREAD_COLOR
from io import BytesIO
from PIL import Image
from os import environ, path
from googleapiclient import discovery
from json import load as load_json
from re import sub


# CONSTANTS

# GCP CONFIG
AUTH_FILE_PATH = path.join('app', 'auth.json')
PROJECT_ID = 'know-your-fruit-283323'
MODEL_NAME = 'prod'
VERSION_NUM = 'v7'

# CLASS_NAMES MUST MATCH ORDER USED IN MODEL 
CLASS_NAMES = ['Apple', 'Avocado', 'Avocado-cut', 'Banana', 'Bell Peppers', 'Blackberries', 'Blueberries', 'Cherries (Sweet)',
    'Coconut', 'Cucumber','Cucumber-cut', 'Grapes', 'Kiwifruit', 'Kiwifruit-cut', 'Lemon', 'Lime', 'Lychee', 'Mango', 'Honeydew Melon',
    'Orange','Orange-cut',' Papaya','Papaya-cut', 'Peach', 'Pear','Pineapple', 'Plum','Pomegranate','Pomegranate-cut', 'Raspberries',
    'Strawberries', 'Tomato', 'Watermelon', 'Watermelon-cut']

# TOP K RESULTS
K = 7


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    ''' If the submit button was pushed, the model will be sent to the cloud and a new page will be presented.
        Otherwise, the front page will be displayed.'''

    if request.method == 'POST':
        img_bytes = request.files['file']
        prediction_names = _get_prediction_names(img_bytes)
        return render_template('choose.html', fruit_names=list(map(_get_noncut_fruit_name, prediction_names)))
    else:
        return render_template('upload.html')


@app.route('/fruits/<fruit_name>')
def get_fruit_data(fruit_name):
    'Renders the page for the fruit data.'''

    # Fix '...-cut' in fruit_name if it is a cut fruit
    fruit_name = _get_noncut_fruit_name(fruit_name)

    fruit_data = _get_fruit_data(fruit_name)
    
    # Make it HTML safe
    fruit_data = Markup(fruit_data)

    return render_template('data.html', name=fruit_name, pic_path='/static/'+fruit_name+'.jpg', info=fruit_data)


def _get_prediction_names(img_bytes):
    '''Returns the top K class names as a list in order of highest likelihood, given the img bytes from the JSON request.'''

    # Process the bytes and convert to numpy array
    byte_io = BytesIO()
    byte_io.write(img_bytes.read())
    byte_io.seek(0)
    file_bytes = asarray(bytearray(byte_io.read()), dtype=uint8)

    # Open as OpenCV image
    img = imdecode(file_bytes, IMREAD_COLOR)
    img = cvtColor(img, COLOR_BGR2RGB)
    img = Image.fromarray(img)

    # Perform some preprocessing before sending to AI Platform Prediction. GCP handles the rest of the processing.
    img = img.resize((299, 299))
    in_img = array(img).reshape(1, 299, 299, 3).tolist()

    # Sends JSON Request to GCP Model. Response stored in preds
    preds = _predict_json(PROJECT_ID, MODEL_NAME, in_img, version=VERSION_NUM)

    # Extracts the vector of probabilities corresponding to the class names
    p_vec = array([preds[0]['sequential']]) 
    
    # Convert probability vector into ordered (by likelihood) list of top K class names
    prediction_names = _get_top_k_class_predictions(p_vec, K, CLASS_NAMES)[0]

    return prediction_names


def _get_fruit_data(fruit_name):
    '''Returns the data corresponding to the correct fruit_name.'''

    # Opens the JSON stored in fruit_file. Names in this file must match those of the CLASS_NAMES variable, aside from ...-cut
    fruit_dict = None
    with open(path.join('app', 'dict_fruit.txt'), 'r', encoding='utf8') as fruit_file:
        fruit_dict = load_json(fruit_file)
    
    data = fruit_dict[fruit_name]
    
    # Removing some of the unnecessary info 
    data = data.split('SERVE')[0]

    lowercase_heading_lst = ['Varieties to Explore', 'Nutrient Content Claims', 'Health Claims', 
    'STORE', 'SELECT', 'Storage', 'Selection' 'Nutrition Benefits']
    for heading in lowercase_heading_lst:
        data = data.replace(heading, '<h3>'+heading.upper()+'</h3>')
        data = data.replace('<h3>'+heading.upper()+'</h3><br>', '<h3>'+heading.upper()+'</h3>')
        data = data.replace('<h3>'+heading.upper()+'</h3> <br>', '<h3>'+heading.upper()+'</h3>')
    return data


def _get_noncut_fruit_name(fruit_name):
    '''Get name of fruit without cut, if it was, otherwise unchanged.'''

    last_4_chars = ""
    try:
        last_4_chars = fruit_name[-4:]
    except:
        pass
    if last_4_chars == '-cut':
        fruit_name = fruit_name[:-4]
    
    return fruit_name


def _predict_json(project, model, instances, version=None):
    '''Creates the JSON request to the model given the instance, and returns the response.'''

    environ['GOOGLE_APPLICATION_CREDENTIALS'] = AUTH_FILE_PATH
    service = discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def _get_top_k_class_predictions(predictions, k, class_names):
    '''Returns a np.array of the top k class predictions corresponding to the array of prediction vectors.
       Currently the application only supports one prediction, but this function would work if there was more.'''

    sorted_indices = argsort(predictions, axis=1)
    flipped = fliplr(sorted_indices)
    top_k_indices = flipped[:, :k]
    res = []
    for row in top_k_indices:
        r = []
        for elem in row:
            r.append(class_names[elem])
        res.append(r)
    return res


if __name__ == '__main__':
    app.run('0.0.0.0')
