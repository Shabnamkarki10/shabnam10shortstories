from sklearn.feature_extraction.text import TfidfVectorizer
import string
import pickle


class TfIdfVectorizer:

    def __init__(self, df, col) -> None:
        self.__df = df
        self.__col = col

    def save_vectorizer_and_matrix(self):
        self.__preprocess()
        self.__vectorize_and_save_tfidf_matrix()

    def __preprocess(self):
        # Convert to lowercase
        self.__df[self.__col] = self.__df[self.__col].str.lower()
        # # Remove punctuations and special characters
        edited = []
        for idx, row in self.__df.iterrows():
            edited.append(
                row[self.__col].translate(str.maketrans("", "", string.punctuation))
            )

        self.__df[self.__col] = edited

    def __vectorize_and_save_tfidf_matrix(self):
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(self.__df[self.__col])
        with open("./outputs/tfidf_vectorizer.pkl", "wb") as file:
            pickle.dump(vectorizer, file)

        with open("./outputs/tfidf_matrix.pkl", "wb") as file:
            pickle.dump(tfidf_matrix, file)
