from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/calculate', methods=['POST'])
def calculate():
    criteria_count = int(request.form['criteria_count'])
    alternatives_count = int(request.form['alternatives_count'])
    method = request.form['method']

    criteria = request.form.getlist('criteria')
    weights = list(map(float, request.form.getlist('weights')))
    alternatives = request.form.getlist('alternatives')

    scores = []
    for i in range(alternatives_count):
        alt_scores = list(map(float, request.form.getlist(f'scores_{i}')))
        scores.append(alt_scores)

    scores = np.array(scores)

    if method == 'saw':
        final_scores = calculate_saw(scores, weights)
    elif method == 'wp':
        final_scores = calculate_wp(scores, weights)
    elif method == 'topsis':
        final_scores = calculate_topsis(scores, weights)
    elif method == 'ahp':
        final_scores = calculate_ahp(scores, weights)

    best_alternative = alternatives[np.argmax(final_scores)]

    # Buat list dari dictionary untuk memudahkan pengiriman data ke template
    results = [{'alternative': alternatives[i], 'score': final_scores[i]} for i in range(len(alternatives))]

    return render_template('results.html', results=results, best_alternative=best_alternative)

def calculate_saw(scores, weights):
    normalized_scores = scores / np.linalg.norm(scores, axis=0)
    final_scores = np.dot(normalized_scores, weights)
    return final_scores

def calculate_wp(scores, weights):
    weighted_scores = scores ** weights
    final_scores = np.prod(weighted_scores, axis=1)
    return final_scores

def calculate_topsis(scores, weights):
    norm_scores = scores / np.sqrt(np.sum(scores**2, axis=0))
    weighted_scores = norm_scores * weights
    ideal_best = np.max(weighted_scores, axis=0)
    ideal_worst = np.min(weighted_scores, axis=0)

    distance_to_best = np.sqrt(np.sum((weighted_scores - ideal_best) ** 2, axis=1))
    distance_to_worst = np.sqrt(np.sum((weighted_scores - ideal_worst) ** 2, axis=1))

    final_scores = distance_to_worst / (distance_to_best + distance_to_worst)
    return final_scores

def calculate_ahp(scores, weights):
    final_scores = np.dot(scores, weights)
    return final_scores

if __name__ == '__main__':
    app.run(debug=True)
