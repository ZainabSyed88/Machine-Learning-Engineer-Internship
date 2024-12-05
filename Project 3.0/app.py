from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
# Initialize Flask app
app = Flask(__name__)
# Load model and dataset
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))
@app.route('/')
def home():
    return render_template('index.html', 
                           companies=df['Company'].unique(),
                           types=df['TypeName'].unique(),
                           cpus=df['CPU Brand'].unique(),
                           gpus=df['GPU Brand'].unique(),
                           os_list=df['os'].unique())
@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    company = request.form.get('company')
    type_ = request.form.get('type')
    ram = int(request.form.get('ram'))
    weight = float(request.form.get('weight'))
    touchscreen = 1 if request.form.get('touchscreen') == 'Yes' else 0
    ips = 1 if request.form.get('ips') == 'Yes' else 0
    screen_size = float(request.form.get('screen_size'))
    resolution = request.form.get('resolution')
    cpu = request.form.get('cpu')
    hdd = int(request.form.get('hdd'))
    ssd = int(request.form.get('ssd'))
    gpu = request.form.get('gpu')
    os = request.form.get('os')
    # Compute PPI (Pixels Per Inch)
    x_res, y_res = map(int, resolution.split('*'))
    ppi = ((x_res**2 + y_res**2)**0.5) / screen_size
    # Create input array for prediction
    query = np.array([company, type_, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os]).reshape(1, -1)
    predicted_price = int(np.exp(pipe.predict(query)[0]))
    return jsonify({'predicted_price': f"₹{predicted_price}"})

if __name__ == '__main__':
    app.run(debug=True)