import re

from emoji.core import demojize
from nltk import TweetTokenizer
from transformers import pipeline
import contractions


class RelevanceScore:

    def __init__(self):
        tokenizer_kwargs = {'padding': True, 'truncation': True}
        self.personal_story_pipeline = pipeline("text-classification", model="falkne/storytelling-LM-europarl-mixed-en", **tokenizer_kwargs)
        self.argument_pipeline = pipeline("text-classification", model="TomatenMarc/WRAP", **tokenizer_kwargs)
        self.tokenizer = TweetTokenizer()

    def compute_relevance_score(self, comment: str) -> float:
        return (self.compute_reason_statement_score(comment) + self.compute_source_score(comment) + self.compute_personal_story_score(comment) +
                self.compute_example_score(comment))

    def compute_reason_statement_score(self, comment: str) -> float:
        try:
            normalized_comment = self.normalizeTweet(comment)
            pred = self.argument_pipeline(normalized_comment)[0]
            if pred["label"] == "Reason":
                result = 2 * pred["score"]
            elif pred["label"] == "Statement":
                result = pred["score"]
            else:
                result = 0
            return round(result, 3)
        except Exception as e:
            print(e)

    # Code from: https://anonymous.4open.science/r/TACO/notebooks/classifier_cv.ipynb
    def normalizeToken(self, token):
        lowercased_token = token.lower()
        if token.startswith("@"):
            return "@USER"
        elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
            return "HTTPURL"
        elif len(token) == 1:
            return demojize(token)
        else:
            if token == "’":
                return "'"
            elif token == "…":
                return "..."
            else:
                return contractions.fix(token)  # Change in code, the original version does not resolve all contractions

    def normalizeTweet(self, tweet):
        tokens = self.tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
        normTweet = " ".join([self.normalizeToken(token) for token in tokens])

        normTweet = (
            normTweet.replace("cannot ", "can not ")
            .replace("n't ", " n't ")
            .replace("n 't ", " n't ")
            .replace("ca n't", "can't")
            .replace("ai n't", "ain't")
        )
        normTweet = (
            normTweet.replace("'m ", " 'm ")
            .replace("'re ", " 're ")
            .replace("'s ", " 's ")
            .replace("'ll ", " 'll ")
            .replace("'d ", " 'd ")
            .replace("'ve ", " 've ")
        )
        normTweet = (
            normTweet.replace(" p . m .", "  p.m.")
            .replace(" p . m ", " p.m ")
            .replace(" a . m .", " a.m.")
            .replace(" a . m ", " a.m ")
        )

        return " ".join(normTweet.split())

    @staticmethod
    def compute_source_score(comment):
        url_pattern = re.compile(r'http[s]?://|www(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        urls = re.findall(url_pattern, comment)

        return 1 if len(urls) > 0 else 0

    def compute_personal_story_score(self, comment: str) -> int:
        sentences = comment.split(".")
        prediction = self.personal_story_pipeline.predict(comment)[0]

        if len(sentences) > 3:
            step = int(len(sentences) / 3)
            for i in range(0, len(sentences) - 1, step):
                partial_comment = " ".join(sentences[i:i + 1])
                prediction = self.personal_story_pipeline.predict(partial_comment)[0]

        if prediction["label"] == "LABEL_1":
            return prediction["score"]
        return 0

    def compute_example_score(self, comment):
        sentences = comment.split(".")

        if len(sentences) > 3:
            step = int(len(sentences) / 3)

            for i in range(0, len(sentences) - 1, step):
                partial_comment = " ".join(sentences[i:i + 1])
                prediction = self.argument_pipeline.predict(self.normalizeTweet(partial_comment))[0]
        else:
            prediction = self.argument_pipeline.predict(self.normalizeTweet(comment))[0]

        if prediction["label"] == "Notification":
            return prediction["score"]

        return 0
