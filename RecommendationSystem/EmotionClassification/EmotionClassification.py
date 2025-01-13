import json
import logging
import random

import numpy as np
import torch
from pandas import DataFrame

from transformers.utils import logging

import torch.nn.functional as F

from RecommendationSystem.EmotionClassification.EmotionEnum import Emotion
from RecommendationSystem.SentimentClassification.SentimentEnum import Sentiment
from RecommendationSystem.StanceDetection.DistillationModel.utils import modeling
from RecommendationSystem.StanceDetection.DistillationModel.utils.model_helper import load_model
import RecommendationSystem.StanceDetection.DistillationModel.utils.preprocessing as pp
import RecommendationSystem.StanceDetection.DistillationModel.utils.data_helper as dh
from RecommendationSystem.StanceDetection.Stance import Stance
from transformers import AutoTokenizer


class EmotionClassification:
    def __init__(self):
        logging.set_verbosity_error()
        self.directory_path = "/code/RecommendationSystem/static/EmotionClassification/"
        self.normalization_dict = self.init_normalization_dict()
        self.model = modeling.stance_classifier(num_labels=6, model_select="Bertweet")
        self.tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(device)

        load_model(self.model, self.directory_path + 'EmotionClassificationModel.pt')
        self.model.eval()

    def compute_emotion(self, user_comment: str):
        emotion = None
        try:
            emotion = self.compute(user_comment)
        except Exception as e:
            print(e)
        return emotion

    def compute(self, user_comment: str):
        data = DataFrame.from_dict({
            "Tweet": [user_comment],
            "Target": ['None'],
            "Stance": ["FAVOR"]
        })

        x_val, y_val, x_val_target = pp.clean_dataframe(data, self.normalization_dict)

        batch_size = 32

        x_val_all = [x_val, y_val, x_val_target]
        seed = 1
        model_select = 'Bertweet'
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        x_val_all = dh.data_tokenizer(self.tokenizer, x_val_all, model_select)

        x_val_input_ids, x_val_seg_ids, x_val_atten_masks, y_val, x_val_len, valloader = \
            dh.data_loader(x_val_all, batch_size, 'val', "test", model_select)

        val_preds = []
        with torch.no_grad():
            for input_ids, seg_ids, atten_masks, target, length in valloader:
                pred1 = self.model(input_ids, seg_ids, atten_masks, length)
                val_preds.append(pred1)
            preds = torch.cat(val_preds, 0)
            rounded_preds = F.softmax(preds, dim=1)
            _, indices = torch.max(rounded_preds, 1)
            y_pred = np.array(indices.cpu().numpy())

            if y_pred[0] == 0:
                return Emotion.SADNESS
            elif y_pred[0] == 1:
                return Emotion.JOY
            elif y_pred[0] == 2:
                return Emotion.LOVE
            elif y_pred[0] == 3:
                return Emotion.ANGER
            elif y_pred[0] == 4:
                return Emotion.FEAR
            elif y_pred[0] == 5:
                return Emotion.SURPRISE

    def init_normalization_dict(self):
        with open(self.directory_path + "noslang_data.json", "r") as f:
            data1 = json.load(f)
        data2 = {}
        with open(self.directory_path + "emnlp_dict.txt", "r") as f:
            lines = f.readlines()
            for line in lines:
                row = line.split('\t')
                data2[row[0]] = row[1].rstrip()
        normalization_dict = {**data1, **data2}
        return normalization_dict
