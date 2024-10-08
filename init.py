from stories import Stories
from tf_idf import TfIdfVectorizer
import pandas as pd


class InitialieData:

    def __init__(self) -> None:
        self.__save_stories()
        self.__save_tfidf()

    def __save_stories(self):
        stories = Stories(
            "./static/stories",
        "./static/images"
        )
        stories.save_stories_and_images()

    def __load_stories(self):
        return pd.read_csv("./outputs/stories.csv")

    def __save_tfidf(self):
        stories = self.__load_stories()
        stories["combined"] = stories["title"] + " " + stories["story"]
        tfidf = TfIdfVectorizer(stories, "combined")
        tfidf.save_vectorizer_and_matrix()


InitialieData()
