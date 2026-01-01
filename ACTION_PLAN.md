# Action Plan: `custom_xaxis` Branch Improvements

## Required Before PR

- [ ] **Expose new parameters in `plot_likert()` function**
  - Add `fixed_max_label_percentage_left` and `fixed_max_label_percentage_right` to `plot_likert()` signature
  - Pass them through to `plot_counts()`

## Recommended

- [x] **Add warning when only one fixed param is provided**
  - If user provides `fixed_max_label_percentage_left` but not `_right` (or vice versa), warn that the feature requires both

- [x] **Clean up test comments**
  - Remove verbose explanatory comments in `tests/test_plot_likert.py`

## Optional

- [ ] **Add integration test**
  - Create a test that actually generates a plot with `fixed_max_label_percentage_*` params and validates the output

- [ ] **Consider API simplification**
  - Could combine into single `fixed_axis_range` parameter (tuple or single value for symmetric)
