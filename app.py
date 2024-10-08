from flask import (
    Flask,
    request,
    render_template,
)
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import string
import pandas as pd
import json

app = Flask(__name__, static_url_path="/static")

from sklearn.metrics.pairwise import cosine_similarity


def preprocess(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text


def tfidf_search(query: str) -> list[int]:
    with open("./outputs/tfidf_vectorizer.pkl", "rb") as pickle_file:
        tfidf_vectorizer = pickle.load(pickle_file)

    with open("./outputs/tfidf_matrix.pkl", "rb") as pickle_file:
        tfidf_matrix = pickle.load(pickle_file)

    query_vec = tfidf_vectorizer.transform([query])
    similarity_scores = cosine_similarity(query_vec, tfidf_matrix)[0]

    score_df = pd.DataFrame(
        data={"score": similarity_scores, "index": list(range(len(similarity_scores)))}
    )

    score_df = score_df.sort_values(by="score", ascending=False)
    score_df = score_df[score_df["score"] > 0]

    return list(score_df["index"])


def index_to_story(indices: list[int]):
    stories = pd.read_csv("./outputs/stories.csv")

    stories = (
        stories[stories.index.isin(indices)]
        .reset_index()
        .rename(columns={"index": "id"})
    )

    with open("./outputs/images.json", "r") as file:
        images = json.load(file)

    stories["image_location"] = stories["title"].apply(lambda x: images[x])
    stories["id"] = indices

    return stories


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    if request.method == "POST":
        query_org = request.form["query-text"]
        query = preprocess(query_org)

        best_results_index = tfidf_search(query)

        stories = index_to_story(best_results_index)

        return render_template("results.html", stories=stories, query=query_org)


@app.route("/story", methods=["GET"])
def story():
    story = request.args.get("story")
    title = request.args.get("title")
    image_location = request.args.get("image_location")
    return render_template(
        "story.html", story=story, title=title, image_location=image_location
    )


if __name__ == "__main__":
    app.run(debug=True)
