# Pair-level AUC by type pair: model_a vs model_b_masklr10 (test set)

Each item pair in a compatibility outfit is labelled real/fake by its outfit; AUC of -distance per type pair (pairs with ≥500 examples). Sorted by gain.

| type pair | pairs | model_a | model_b_masklr10 | gain | mask moved from 1s (1 − cos) |
|---|---|---|---|---|---|
| scarves–sunglasses | 622 | 0.746 | 0.755 | +0.008 | 0.225 |
| outerwear–sunglasses | 2022 | 0.717 | 0.722 | +0.005 | 0.143 |
| hats–sunglasses | 840 | 0.735 | 0.739 | +0.004 | 0.181 |
| all-body–tops | 1072 | 0.777 | 0.781 | +0.004 | 0.246 |
| bottoms–tops | 15946 | 0.729 | 0.733 | +0.004 | 0.113 |
| bags–hats | 2592 | 0.784 | 0.788 | +0.004 | 0.140 |
| shoes–sunglasses | 5292 | 0.735 | 0.739 | +0.004 | 0.109 |
| accessories–shoes | 3608 | 0.765 | 0.768 | +0.003 | 0.162 |
| bottoms–shoes | 15636 | 0.749 | 0.751 | +0.002 | 0.072 |
| jewellery–sunglasses | 4264 | 0.721 | 0.722 | +0.001 | 0.119 |
| jewellery–jewellery | 9386 | 0.810 | 0.811 | +0.001 | 0.184 |
| bags–tops | 15170 | 0.757 | 0.757 | +0.000 | 0.064 |
| bags–shoes | 23360 | 0.794 | 0.794 | -0.000 | 0.064 |
| jewellery–shoes | 21078 | 0.750 | 0.750 | -0.001 | 0.068 |
| hats–jewellery | 2262 | 0.744 | 0.743 | -0.001 | 0.170 |
| jewellery–outerwear | 8076 | 0.734 | 0.734 | -0.001 | 0.141 |
| shoes–tops | 17712 | 0.768 | 0.767 | -0.001 | 0.068 |
| outerwear–shoes | 10560 | 0.771 | 0.770 | -0.001 | 0.092 |
| accessories–bags | 2894 | 0.751 | 0.750 | -0.001 | 0.176 |
| bottoms–hats | 1958 | 0.763 | 0.762 | -0.001 | 0.172 |
| bottoms–jewellery | 11652 | 0.708 | 0.707 | -0.001 | 0.109 |
| bags–bottoms | 13336 | 0.742 | 0.740 | -0.002 | 0.067 |
| bags–sunglasses | 5022 | 0.741 | 0.739 | -0.002 | 0.084 |
| accessories–sunglasses | 708 | 0.729 | 0.727 | -0.002 | 0.238 |
| accessories–tops | 2832 | 0.729 | 0.727 | -0.003 | 0.188 |
| bags–outerwear | 9760 | 0.767 | 0.764 | -0.003 | 0.105 |
| jewellery–scarves | 2192 | 0.737 | 0.734 | -0.003 | 0.183 |
| jewellery–tops | 13096 | 0.720 | 0.716 | -0.004 | 0.106 |
| bags–jewellery | 19200 | 0.756 | 0.753 | -0.004 | 0.062 |
| all-body–jewellery | 8950 | 0.760 | 0.756 | -0.004 | 0.125 |
| hats–shoes | 3014 | 0.819 | 0.814 | -0.004 | 0.154 |
| bags–scarves | 2414 | 0.786 | 0.781 | -0.005 | 0.163 |
| accessories–hats | 588 | 0.782 | 0.777 | -0.005 | 0.232 |
| all-body–bags | 9072 | 0.797 | 0.792 | -0.005 | 0.116 |
| bottoms–sunglasses | 3508 | 0.704 | 0.699 | -0.005 | 0.144 |
| outerwear–tops | 7020 | 0.753 | 0.747 | -0.006 | 0.181 |
| all-body–outerwear | 3110 | 0.784 | 0.778 | -0.006 | 0.179 |
| bags–bags | 780 | 0.757 | 0.750 | -0.007 | 0.340 |
| accessories–all-body | 1154 | 0.791 | 0.784 | -0.007 | 0.203 |
| sunglasses–tops | 3902 | 0.717 | 0.710 | -0.007 | 0.111 |
| hats–outerwear | 1384 | 0.789 | 0.781 | -0.008 | 0.202 |
| accessories–outerwear | 1756 | 0.729 | 0.720 | -0.009 | 0.237 |
| tops–tops | 1810 | 0.753 | 0.744 | -0.010 | 0.273 |
| bottoms–outerwear | 7002 | 0.738 | 0.727 | -0.010 | 0.145 |
| all-body–shoes | 10038 | 0.825 | 0.814 | -0.010 | 0.116 |
| hats–tops | 2286 | 0.783 | 0.772 | -0.011 | 0.179 |
| all-body–hats | 1024 | 0.826 | 0.813 | -0.014 | 0.184 |
| scarves–shoes | 2666 | 0.769 | 0.754 | -0.015 | 0.173 |
| all-body–sunglasses | 1728 | 0.754 | 0.736 | -0.018 | 0.121 |
| accessories–jewellery | 3344 | 0.726 | 0.708 | -0.018 | 0.195 |
| accessories–bottoms | 2464 | 0.720 | 0.701 | -0.019 | 0.217 |
| outerwear–scarves | 1518 | 0.734 | 0.713 | -0.021 | 0.256 |
| scarves–tops | 2066 | 0.752 | 0.730 | -0.021 | 0.224 |
| all-body–scarves | 746 | 0.814 | 0.792 | -0.022 | 0.247 |
| all-body–bottoms | 732 | 0.781 | 0.758 | -0.023 | 0.248 |
| shoes–shoes | 502 | 0.756 | 0.732 | -0.024 | 0.347 |
| bottoms–scarves | 1802 | 0.739 | 0.715 | -0.024 | 0.203 |
| accessories–scarves | 656 | 0.759 | 0.727 | -0.031 | 0.277 |

Pair-weighted mean gain: -0.0029; 12/58 type pairs improve.
