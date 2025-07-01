from flask import Flask, render_template, request
import pandas as pd
import pickle
import os
import difflib
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load model.pkl
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Extract components
df = model["df"]
tfidf = model["tfidf"]

# Ensure Dataset_name is string
df["Dataset_name"] = df["Dataset_name"].fillna("").astype(str)

# Transform all dataset names
tfidf_matrix = tfidf.transform(df["Dataset_name"])

# Index mapping for title lookup
indices = pd.Series(df.index, index=df["Dataset_name"]).drop_duplicates()

# Recommendation function
def recommend(title, num=5):
    if title in indices:
        idx = indices[title]
        matched_title = title
    else:
        matches = difflib.get_close_matches(title, indices.index, n=1, cutoff=0.3)
        if not matches:
            return [], None
        matched_title = matches[0]
        idx = indices[matched_title]

    # Compute similarity with all items
    sim_scores = list(enumerate(cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:num+1]
    dataset_indices = [i[0] for i in sim_scores]

    # Return selected columns
    return df[['Dataset_name', 'Dataset_link', 'Type_of_file', 'size', 'No_of_files', 'Medals', 'Date', 'Time']].iloc[dataset_indices].to_dict(orient='records'), matched_title

# Home route
@app.route('/', methods=['GET', 'POST'])
def index():
    recommendations = []
    query = ""
    matched_title = ""

    if request.method == 'POST':
        query = request.form['dataset_name'].strip()
        recommendations, matched_title = recommend(query)

    return render_template(
        "index.html",
        recommendations=recommendations,
        query=query,
        matched_title=matched_title
    )

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render sets this PORT variable
    app.run(host='0.0.0.0', port=port, debug=True)
