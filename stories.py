import pandas as pd
from os import listdir
from os.path import isfile, join
import json


class Stories:

    def __init__(self, stories_path: str, images_path: str) -> None:
        self.__df: pd.DataFrame = None
        self.__images = self.__read_stories_and_images(stories_path, images_path)

    def __read_stories_and_images(
        self, stories_path: str, images_path: str
    ) -> pd.DataFrame:
        stories = [f for f in listdir(stories_path) if isfile(join(stories_path, f))]
        images = [f for f in listdir(images_path) if isfile(join(images_path, f))]

        images_json = dict()

        self.__df = pd.DataFrame(columns=["title", "story"])
        for story, image in zip(stories, images):
            with open(stories_path + "/" + story, "r") as f:
                data = f.read()
                data = data.split("\n")
                title = data[0]
                data = " ".join(data[1:])
                images_json[title] = "/static/images/" + image

                self.__df.loc[len(self.__df)] = [title, data]

        return images_json

    def save_stories_and_images(self):
        self.__df.to_csv("./outputs/stories.csv", index=None)
        with open("./outputs/images.json", "w") as file:
            file.write(json.dumps(self.__images, indent=4))
