import unittest

from simulate.survey import Survey


class SurveyBranchRecodeTests(unittest.TestCase):
    def make_survey(self):
        return Survey(
            {
                "scales": [
                    {
                        "name": "attention1",
                        "type": "multiple choice",
                        "question": "Select Somewhat disagree.",
                        "items": {"attention1": "Select Somewhat disagree."},
                        "options": {
                            "Strongly disagree": 1,
                            "Disagree": 2,
                            "Somewhat disagree": 3,
                        },
                    }
                ],
                "flow": {"Type": "Root", "Flow": []},
                "questions": {
                    "QID_attention": {
                        "scale_name": "attention1",
                        "raw_payload": {
                            "RecodeValues": {
                                "7": 1,
                                "8": 2,
                                "9": 3,
                            }
                        },
                    }
                },
            }
        )

    def test_not_selected_branch_compares_against_recoded_choice_value(self):
        survey = self.make_survey()
        expression = {
            "LogicType": "Question",
            "QuestionID": "QID_attention",
            "ChoiceLocator": "q://QID_attention/SelectableChoice/9",
            "Operator": "NotSelected",
            "Type": "Expression",
        }
        state = {"question_values": {"QID_attention": 3}}

        self.assertFalse(survey._evaluate_expression(expression, state))

    def test_selected_branch_compares_against_recoded_choice_value(self):
        survey = self.make_survey()
        expression = {
            "LogicType": "Question",
            "QuestionID": "QID_attention",
            "ChoiceLocator": "q://QID_attention/SelectableChoice/9",
            "Operator": "Selected",
            "Type": "Expression",
        }
        state = {"question_values": {"QID_attention": 3}}

        self.assertTrue(survey._evaluate_expression(expression, state))

    def test_branch_choice_without_recode_preserves_raw_choice_id_behavior(self):
        survey = Survey(
            {
                "scales": [],
                "flow": {"Type": "Root", "Flow": []},
                "questions": {"QID_plain": {"raw_payload": {}}},
            }
        )
        expression = {
            "LogicType": "Question",
            "QuestionID": "QID_plain",
            "ChoiceLocator": "q://QID_plain/SelectableChoice/9",
            "Operator": "Selected",
            "Type": "Expression",
        }
        state = {"question_values": {"QID_plain": 9}}

        self.assertTrue(survey._evaluate_expression(expression, state))


if __name__ == "__main__":
    unittest.main()
