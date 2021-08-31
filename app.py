import pandas as pd
import pickle
from flask import Flask, jsonify, request


# load model
path = './model.pkl'
model = pickle.load(open(path, 'rb'))

# app
app = Flask(__name__)

# routes
@app.route('/', methods=['POST'])

def predict():
    # get data
    data = request.get_json(force=True)

    # convert data into dataframe
    data.update( (x, [y]) for x, y in data.items() )
    data_df = pd.DataFrame.from_dict(data)

    # predictions
    result = model.predict(data_df)

    # send back to browser
    output = {
        'result': int(result[0])
    }

    # return data
    return jsonify(result=output)


if __name__ == '__main__':
    app.run(port=5000, debug=True)