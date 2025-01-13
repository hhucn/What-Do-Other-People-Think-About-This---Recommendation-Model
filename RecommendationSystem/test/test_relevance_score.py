import unittest
from unittest.mock import patch

from RecommendationSystem.relevance_score.RelevanceScore import RelevanceScore


class MyTestCase(unittest.TestCase):

    def setUp(self):
        self.relevance_score = RelevanceScore()
        self.standard_statement_comment = "Men shouldn’t be making laws about women’s bodies."
        self.standard_reason_comment = "Opinion: As the draconian (and then some) abortion law takes effect in #Texas, this is not an idle question for millions of Americans. A slippery slope towards more like-minded Republican state legislatures to try to follow suit. #abortion #F24 HTTPURL"
        self.empty_comment = ""
        self.very_long_comment = "Can't we just get Sean Spicer, Ms. Conway and all of Trump's other regular TV defenders--Hugh Hewitt et al.--before TV cameras for a daily 15 minute segment in which they all spout their deflections and misdirections without interruption? Why let reporters questions slow the flow of misinformation? Just let the deflectors spout it all out in a burst. Then we would not have to be plagued by their inanities throughout the day.Fifteen minutes should be plenty of time for FOX News to get all the sound bites they need for any given day.Fifteen minutes should also be plenty of time for those who wish to be misinformed to receive all the disinformation they could possibly want."
        self.personal_story = "My daughter has been tested 4 times for her allergy to peanuts. She is in the highest category of reactivity which means if peanuts are being ingested in her vicinity, she could die. A buffer zone simply doesn’t work in a confined space such as an airline"
        self.example = "...a more progressive inheritance tax and tax capital gains as ordinary income...In reverse order:1. For better or worse, taxing long-term capital gains as ordinary income (short-term gains already are) isn't going to happen. Already, investors go to great lengths to avoid US taxes, since most countries have lower taxes on investment gains. That change would make investors even more inclined to invest elsewhere than they already are. 2. For those who (understandably) don't know the difference, an estate tax is imposed on the person who dies, while an inheritance tax is imposed on the people who inherit. There is no federal inheritance tax, and the federal estate tax has such a large deductible that few decedents pay it. Some states have inheritance taxes; others don't. Those that have them often use the decedent's state to determine whether the inheritance tax applies. Since inheritance taxes are imposed on those who inherit, they do (or at least should) take account of the inheritor's situation. If, for example, a decedent leaves $100,000 each to a rich person and a poor person, the two inheritors shouldn't pay the same tax -- just as those two inheritors would pay different rates on income."
        self.one_sentence_comment = "This is just one sentence"
        
        self.notification_comment_with_source_without_http = "The OASIS Initiative, which I started with Prof. Malcolm Potts at UC Berkeley is focusing on building local leadership and the evidence base necessary to help people in the Sahel face Africa's greatest development challenge: unprecedented population growth and effects of climate change in an already fragile region. Please consider supporting our work - the donate button is on the top right. Gifts to UC Berkeley have only about 3% indirect cost rate. www.oasisinitiative.berkeley.edu "

        self.comment_with_source = self.standard_reason_comment + "https://www.google.de" + self.standard_statement_comment

    def test_detect_source_without_http(self):
        relevance_score = self.relevance_score.compute_source_score(self.notification_comment_with_source_without_http)

        self.assertEqual(relevance_score, 1)

    def test_compute_relevance_score_for_statement_comment(self):
        relevance_score = self.relevance_score.compute_reason_statement_score(self.standard_statement_comment)

        self.assertAlmostEqual(relevance_score, 0.986, delta=0.001)

    def test_compute_relevance_score_for_reason_comment(self):
        relevance_score = self.relevance_score.compute_reason_statement_score(self.standard_reason_comment)

        self.assertAlmostEqual(relevance_score, 1.964, delta=0.001)

    def test_compute_relevance_score_for_empty_comment(self):
        relevance_score = self.relevance_score.compute_reason_statement_score(self.empty_comment)

        self.assertEqual(relevance_score, 0)

    def test_compute_relevance_score_for_very_long_comment(self):
        relevance_score = self.relevance_score.compute_reason_statement_score(self.very_long_comment)

        self.assertIsNotNone(relevance_score)

    def test_compute_source_score_with_source(self):
        relevance_score = self.relevance_score.compute_source_score(self.comment_with_source)

        self.assertEqual(relevance_score, 1)

    def test_compute_source_score_without_source(self):
        relevance_score = self.relevance_score.compute_source_score(self.standard_statement_comment)

        self.assertEqual(relevance_score, 0)

    def test_compute_source_score_for_empty_comment(self):
        relevance_score = self.relevance_score.compute_source_score(self.empty_comment)

        self.assertEqual(relevance_score, 0)

    def test_compute_personal_story_score_with_personal_story(self):
        relevance_score = self.relevance_score.compute_personal_story_score(self.very_long_comment + self.personal_story)

        self.assertAlmostEqual(relevance_score, 0.685, delta=0.001)

    def test_compute_personal_story_score_with_comment_without_personal_story(self):
        relevance_score = self.relevance_score.compute_personal_story_score(self.standard_statement_comment)

        self.assertEqual(relevance_score, 0)

    def test_compute_personal_story_with_empty_comment(self):
        relevance_score = self.relevance_score.compute_personal_story_score(self.empty_comment)

        self.assertEqual(relevance_score, 0)

    def test_compute_personal_story_with_comment_with_length_less_than_sliding_window(self):
        relevance_score = self.relevance_score.compute_personal_story_score(self.one_sentence_comment)

        self.assertIsNotNone(relevance_score)

    def test_compute_example_score_with_comment_with_example(self):
        relevance_score = self.relevance_score.compute_example_score(self.example)

        self.assertAlmostEqual(relevance_score, 0.97, delta=0.001)

    def test_compute_example_score_with_comment_without_example(self):
        relevance_score = self.relevance_score.compute_example_score(self.standard_reason_comment)

        self.assertEqual(relevance_score, 0)

    def test_compute_example_score_with_comment_with_length_less_than_sliding_window(self):
        relevance_score = self.relevance_score.compute_example_score(self.one_sentence_comment)

        self.assertIsNotNone(relevance_score)

    @patch.object(RelevanceScore, "compute_reason_statement_score")
    @patch.object(RelevanceScore, "compute_personal_story_score")
    @patch.object(RelevanceScore, "compute_example_score")
    @patch.object(RelevanceScore, "compute_source_score")
    def test_all_scores_are_called_by_compute_relevance_score(self, mock_compute_reason_statement_score, mock_compute_personal_story_score,
                                                              mock_compute_example_score, mock_compute_source_score):
        self.relevance_score.compute_relevance_score(self.standard_reason_comment)
        mock_compute_reason_statement_score.assert_called_once()
        mock_compute_personal_story_score.assert_called_once()
        mock_compute_source_score.assert_called_once()
        mock_compute_example_score.assert_called_once()


if __name__ == '__main__':
    unittest.main()
