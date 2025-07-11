from flask import Flask, request, render_template, jsonify
from classify import predict_waste
import json

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('/about.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    image = request.files['image']
    print("Received image:", image.filename)
    
    try:
        category = predict_waste(image)
        print("Predicted category:", category)
    except Exception as e:
        print("Error during prediction:", e)
        return jsonify({"category": "error", "info": str(e)})

    with open("recycle_info.json", "r") as f:
        data = json.load(f)
    
    info = data.get(category, "Info not available.")
    return jsonify({"category": category, "info": info})

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host='0.0.0.0', port=port)