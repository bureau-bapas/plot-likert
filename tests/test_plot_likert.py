import unittest

import matplotlib

matplotlib.use("Agg")

import pandas as pd

import plot_likert


class TestLikertResponse(unittest.TestCase):
    def test_replaces_numeric_responses_with_scale_values(self):
        scale = ["Disagree", "Neutral", "Agree"]
        df = pd.DataFrame({"Q1": ["0", "2"], "Q2": ["1", "0"]})

        result = plot_likert.likert_response(df, scale)

        expected = pd.DataFrame(
            {"Q1": ["Disagree", "Agree"], "Q2": ["Neutral", "Disagree"]}
        )
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
