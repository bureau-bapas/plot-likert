import unittest
import pandas as pd
import numpy as np
from plot_likert.plot_likert import _calculate_x_axis_ticks, PADDING_LEFT, PADDING_RIGHT


class TestPlotLikertLogic(unittest.TestCase):
    def setUp(self):
        self.counts = pd.DataFrame({"Q1": [10, 20, 30]}, index=["A", "B", "C"])
        self.padded_counts = pd.DataFrame(
            {"pad": [5, 5, 5], "Q1": [10, 20, 30]}, index=["A", "B", "C"]
        )
        self.center = 50.0

    def test_calculate_x_axis_ticks_fixed(self):
        use_fixed_axis = True
        fixed_max_label_percentage_left = 40.0
        fixed_max_label_percentage_right = 40.0
        counts_are_percentages = True
        xtick_interval = 20

        xvalues, xlabels_formatted, xlim = _calculate_x_axis_ticks(
            use_fixed_axis,
            self.counts,
            self.padded_counts,
            self.center,
            xtick_interval,
            fixed_max_label_percentage_left,
            fixed_max_label_percentage_right,
            counts_are_percentages,
        )

        # Expected labels: 40%, 20%, 0%, 20%, 40%
        expected_labels = ["40%", "20%", "0%", "20%", "40%"]
        self.assertEqual(list(xlabels_formatted), expected_labels)

        # Expected xvalues
        expected_xvalues = [10.0, 30.0, 50.0, 70.0, 90.0]
        np.testing.assert_array_equal(xvalues, expected_xvalues)

        # Expected xlim
        expected_min = 10 - (80 * PADDING_LEFT)
        expected_max = 90 + (80 * PADDING_RIGHT)
        self.assertEqual(xlim, (expected_min, expected_max))

    def test_calculate_x_axis_ticks_dynamic(self):
        use_fixed_axis = False
        counts_are_percentages = False
        xtick_interval = 10

        xvalues, xlabels_formatted, xlim = _calculate_x_axis_ticks(
            use_fixed_axis,
            self.counts,
            self.padded_counts,
            self.center,
            xtick_interval,
            None,
            None,
            counts_are_percentages,
            num_ticks_available=10,
            current_xlim=(0, 100),
        )

        expected_labels = ["0", "10", "20", "30", "", ""]
        self.assertEqual(list(xlabels_formatted), expected_labels)
        np.testing.assert_array_equal(xvalues, [50.0, 40.0, 30.0, 20.0, 10.0, 0.0])


if __name__ == "__main__":
    unittest.main()
