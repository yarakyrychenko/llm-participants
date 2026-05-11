import json
import tempfile
import unittest
from pathlib import Path

from simulate.qsf import parse_qsf


def _question(payload):
    return {
        "SurveyID": "SV_test",
        "Element": "SQ",
        "PrimaryAttribute": payload["QuestionID"],
        "Payload": payload,
    }


class QSFRecodeTests(unittest.TestCase):
    def parse_payloads(self, *payloads):
        data = {
            "SurveyEntry": {"SurveyID": "SV_test", "SurveyName": "Recode test"},
            "SurveyElements": [
                {
                    "SurveyID": "SV_test",
                    "Element": "BL",
                    "PrimaryAttribute": "Survey Blocks",
                    "Payload": {
                        "1": {
                            "Type": "Standard",
                            "Description": "Main",
                            "ID": "BL_main",
                            "BlockElements": [
                                {"Type": "Question", "QuestionID": payload["QuestionID"]}
                                for payload in payloads
                            ],
                            "Options": {"RandomizeQuestions": "false"},
                        }
                    },
                },
                {
                    "SurveyID": "SV_test",
                    "Element": "FL",
                    "PrimaryAttribute": "Survey Flow",
                    "Payload": {"Type": "Root", "Flow": [{"Type": "Block", "ID": "BL_main"}]},
                },
                *[_question(payload) for payload in payloads],
            ],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            qsf_path = Path(temp_dir) / "survey.qsf"
            qsf_path.write_text(json.dumps(data), encoding="utf-8")
            return parse_qsf(str(qsf_path))

    def test_mc_uses_recode_values_and_choice_order(self):
        parsed = self.parse_payloads(
            {
                "QuestionID": "QID_donation",
                "QuestionType": "MC",
                "Selector": "SAVR",
                "DataExportTag": "donation",
                "QuestionText": "How much would you donate?",
                "Choices": {
                    "25": {"Display": "10$"},
                    "1": {"Display": "0$"},
                    "16": {"Display": "1$"},
                },
                "ChoiceOrder": [1, "16", "25"],
                "RecodeValues": {"1": "0", "16": "1", "25": "10"},
            }
        )

        scale = parsed["scales"][0]
        self.assertEqual(scale["name"], "donation")
        self.assertEqual(list(scale["options"]), ["0$", "1$", "10$"])
        self.assertEqual(scale["options"], {"0$": 0, "1$": 1, "10$": 10})

    def test_matrix_uses_recode_values_export_tags_and_declared_orders(self):
        parsed = self.parse_payloads(
            {
                "QuestionID": "QID_alien_info",
                "QuestionType": "Matrix",
                "Selector": "Likert",
                "SubSelector": "SingleAnswer",
                "DataExportTag": "alien_info",
                "QuestionText": "How often do you use these sources?",
                "Choices": {
                    "11": {"Display": "Personal conversations"},
                    "1": {"Display": "Traditional media"},
                    "12": {"Display": "In-person events"},
                },
                "ChoiceOrder": [1, "11", "12"],
                "ChoiceDataExportTags": {
                    "1": "alien_info_1",
                    "11": "alien_info_5",
                    "12": "alien_info_6",
                },
                "Answers": {
                    "9": {"Display": "Very frequently"},
                    "1": {"Display": "Never"},
                    "8": {"Display": "Frequently"},
                },
                "AnswerOrder": [1, "8", "9"],
                "RecodeValues": {"1": "1", "8": "4", "9": "5"},
            }
        )

        scale = parsed["scales"][0]
        self.assertEqual(
            scale["items"],
            {
                "alien_info_1": "Traditional media",
                "alien_info_5": "Personal conversations",
                "alien_info_6": "In-person events",
            },
        )
        self.assertEqual(list(scale["options"]), ["Never", "Frequently", "Very frequently"])
        self.assertEqual(scale["options"], {"Never": 1, "Frequently": 4, "Very frequently": 5})

    def test_non_numeric_recode_values_are_preserved(self):
        parsed = self.parse_payloads(
            {
                "QuestionID": "QID_status",
                "QuestionType": "MC",
                "Selector": "SAVR",
                "DataExportTag": "status",
                "QuestionText": "Choose status",
                "Choices": {"1": {"Display": "Control"}, "2": {"Display": "Treatment"}},
                "RecodeValues": {"1": "control", "2": "treatment"},
            }
        )

        self.assertEqual(
            parsed["scales"][0]["options"],
            {"Control": "control", "Treatment": "treatment"},
        )

    def test_mc_prefers_qualtrics_advanced_fixed_choice_order(self):
        parsed = self.parse_payloads(
            {
                "QuestionID": "QID_religion",
                "QuestionType": "MC",
                "Selector": "SAVR",
                "DataExportTag": "religion",
                "QuestionText": "What is your religion, if any?",
                "Choices": {
                    "14": {"Display": "Catholic"},
                    "15": {"Display": "Mormon"},
                    "16": {"Display": "Protestant"},
                    "17": {"Display": "Orthodox Christian"},
                    "18": {"Display": "Muslim"},
                    "19": {"Display": "Jewish"},
                    "20": {"Display": "Buddhist"},
                    "21": {"Display": "Hindu"},
                    "22": {"Display": "Other religion (please specify)"},
                    "23": {"Display": "I am not religious"},
                },
                "ChoiceOrder": ["14", "15", "16", "17", "18", "19", "20", "21", "22", "23"],
                "RecodeValues": {
                    "14": "3",
                    "15": "4",
                    "16": "2",
                    "17": "5",
                    "18": "7",
                    "19": "6",
                    "20": "8",
                    "21": "9",
                    "22": "10",
                    "23": "1",
                },
                "Randomization": {
                    "Type": "Advanced",
                    "Advanced": {
                        "FixedOrder": ["23", "16", "14", "15", "17", "19", "18", "20", "21", "22"],
                        "RandomizeAll": [],
                        "Undisplayed": [],
                    },
                },
            }
        )

        self.assertEqual(
            list(parsed["scales"][0]["options"]),
            [
                "I am not religious",
                "Protestant",
                "Catholic",
                "Mormon",
                "Orthodox Christian",
                "Jewish",
                "Muslim",
                "Buddhist",
                "Hindu",
                "Other religion (please specify)",
            ],
        )
        self.assertEqual(
            list(parsed["scales"][0]["options"].values()),
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        )


if __name__ == "__main__":
    unittest.main()
