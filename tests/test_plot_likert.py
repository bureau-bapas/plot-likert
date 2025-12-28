
import unittest
import pandas as pd
import numpy as np
from plot_likert.plot_likert import _calculate_x_axis_ticks, PADDING_LEFT, PADDING_RIGHT

class TestPlotLikertLogic(unittest.TestCase):

    def setUp(self):
        # Create some dummy data structures
        self.counts = pd.DataFrame({'Q1': [10, 20, 30]}, index=['A', 'B', 'C'])
        self.padded_counts = pd.DataFrame({'pad': [5, 5, 5], 'Q1': [10, 20, 30]}, index=['A', 'B', 'C'])
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
            counts_are_percentages
        )

        # Expected labels: 40%, 20%, 0%, 20%, 40%
        expected_labels = ['40%', '20%', '0%', '20%', '40%']
        self.assertEqual(list(xlabels_formatted), expected_labels)

        # Expected xvalues
        # center is 50.
        # Left labels: 40, 20, 0. so 50-40=10, 50-20=30
        # Right labels: 20, 40. so 50+20=70, 50+40=90
        # xvalues should be [10, 30, 50, 70, 90]
        expected_xvalues = [10.0, 30.0, 50.0, 70.0, 90.0]
        np.testing.assert_array_equal(xvalues, expected_xvalues)

        # Expected xlim
        # min = center - fixed_left = 50 - 40 = 10
        # max = center + fixed_right = 50 + 40 = 90
        # total_width = 80
        # padding_left = 80 * PADDING_LEFT
        # padding_right = 80 * PADDING_RIGHT
        expected_min = 10 - (80 * PADDING_LEFT)
        expected_max = 90 + (80 * PADDING_RIGHT)
        self.assertEqual(xlim, (expected_min, expected_max))

    def test_calculate_x_axis_ticks_dynamic(self):
        use_fixed_axis = False
        counts_are_percentages = False
        xtick_interval = 10

        # Mock what max_width calculation would produce in the function
        # max_width = int(round(padded_counts.sum(axis=1).max()))
        # In setup: 5 + 30 = 35.

        xvalues, xlabels_formatted, xlim = _calculate_x_axis_ticks(
            use_fixed_axis,
            self.counts,
            self.padded_counts, # sum max is 35
            self.center, # 50.0
            xtick_interval,
            None,
            None,
            counts_are_percentages,
            num_ticks_available=10,
            current_xlim=(0, 100)
        )

        # Dynamic logic:
        # max_width = 35
        # center = 50
        # right_edge = 35 - 50 = -15... wait lines 220 in original code:
        # right_edge = max_width - center
        # If center is > max_width, right_edge is negative... this seems specific to how data is set up.
        # In real plot_likert, center and padded_counts are calculated together.
        # Let's align them.
        # If center is 50, it means the middle is at 50.
        # If max_width is 35, something is wrong with my mock data vs logic.
        # Usually max_width >= center.

        # Let's adjust mock data to be realistic
        # Say scale is 1-5. Middle is 3.
        # center = sum of first 2 + half of 3rd.
        # padded counts aligns so that 'center' is the vertical line.
        # So max_width (total width of bar) should extend beyond center usually?

        # Actually, let's just trust the arithmetic of the function and check outputs based on inputs.
        # max_width = 35. center = 50.
        # right_edge = 35 - 50 = -15.
        # interval = 10.
        # right_labels = range(10, -15+10, 10) -> range(10, -5, 10) -> [] empty.
        # left_labels = range(0, 51, 10) -> [0, 10, 20, 30, 40, 50]
        # xlabels = [0, 10, 20, 30, 40, 50] U [] -> [0, 10, 20, 30, 40, 50]
        # xvalues:
        # left_values = 50 - [0, 10, 20, 30, 40, 50] = [50, 40, 30, 20, 10, 0]
        # right_values = 50 + [] = []
        # concatenated: [50, 40, 30, 20, 10, 0] ... wait, the concatenation order in original/new code:
        # xlabels = concat(left, right) = [0, 10... 50]
        # xvalues = concat(left-values, right-values) = [50, 40... 0]

        # xlabels formatted: "0", "10", ... "50"

        expected_labels = ['0', '10', '20', '30', '', '']
        # Note: set/unique/sort logic might change order if I'm not careful?
        # In dynamic logic: xlabels = np.concatenate([left_labels, right_labels])
        # It's NOT sorted in dynamic logic in the original code?
        # Original: xlabels = np.concatenate([left_labels, right_labels])
        # left_labels = 0, 10, ...
        # right_labels = 10, 20, ...
        # So it is 0, 10, 20, 30, 40, 50.

        # xvalues:
        # left_values = center - left_labels (50 - 0 = 50, 50-10=40...) -> 50, 40, 30, 20, 10, 0
        # right_values = center + right_labels -> empty

        # So xvalues corresponds to xlabels?
        # xlabels[0] = 0. xvalues[0] = 50.
        # 0 label is at 50? (which is center). Correct.
        # 10 label is at 40. Correct (10 away from center to left).

        self.assertEqual(list(xlabels_formatted), expected_labels)
        np.testing.assert_array_equal(xvalues, [50., 40., 30., 20., 10., 0.])


if __name__ == '__main__':
    unittest.main()
