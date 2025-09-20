from flask import Flask, render_template, request
from utils.recommend import get_recommendations
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")

# Load colleges CSV once at startup
colleges_df = pd.read_csv("data/colleges.csv")

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        location = request.form.get('location')
        interests = request.form.get('interests')
        field = request.form.get('field')

        # Get top 10 recommendations using your BERT logic
        recommendations = get_recommendations(colleges_df, location, interests, field)

        return render_template('results.html', recommendations=recommendations)
    
    return render_template('index.html')

if __name__ == '__main__':
    # Run app on port 5000
    app.run(debug=True, host='0.0.0.0', port=5000)
