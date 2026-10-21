# Pair-level AUC by type pair: model_a vs model_b (test set)

Each item pair in a compatibility outfit is labelled real/fake by its outfit; AUC of -distance per type pair (pairs with ≥500 examples). Sorted by gain.

| type pair | pairs | model_a | model_b | gain | mask moved from 1s (1 − cos) |
|---|---|---|---|---|---|
| hats–jewellery | 2262 | 0.744 | 0.757 | +0.014 | 0.043 |
| all-body–tops | 1072 | 0.777 | 0.790 | +0.013 | 0.036 |
| outerwear–sunglasses | 2022 | 0.717 | 0.727 | +0.011 | 0.036 |
| tops–tops | 1810 | 0.753 | 0.763 | +0.010 | 0.037 |
| hats–sunglasses | 840 | 0.735 | 0.745 | +0.010 | 0.040 |
| jewellery–shoes | 21078 | 0.750 | 0.758 | +0.008 | 0.030 |
| all-body–scarves | 746 | 0.814 | 0.822 | +0.008 | 0.029 |
| hats–tops | 2286 | 0.783 | 0.789 | +0.006 | 0.043 |
| jewellery–jewellery | 9386 | 0.810 | 0.816 | +0.006 | 0.037 |
| accessories–shoes | 3608 | 0.765 | 0.770 | +0.005 | 0.036 |
| shoes–sunglasses | 5292 | 0.735 | 0.740 | +0.005 | 0.040 |
| all-body–bottoms | 732 | 0.781 | 0.785 | +0.004 | 0.025 |
| bags–bags | 780 | 0.757 | 0.761 | +0.004 | 0.032 |
| scarves–tops | 2066 | 0.752 | 0.755 | +0.003 | 0.033 |
| accessories–jewellery | 3344 | 0.726 | 0.729 | +0.003 | 0.042 |
| all-body–jewellery | 8950 | 0.760 | 0.763 | +0.003 | 0.036 |
| bags–jewellery | 19200 | 0.756 | 0.759 | +0.002 | 0.026 |
| shoes–shoes | 502 | 0.756 | 0.758 | +0.002 | 0.027 |
| scarves–sunglasses | 622 | 0.746 | 0.748 | +0.002 | 0.031 |
| jewellery–tops | 13096 | 0.720 | 0.721 | +0.002 | 0.037 |
| bottoms–jewellery | 11652 | 0.708 | 0.709 | +0.001 | 0.036 |
| bags–scarves | 2414 | 0.786 | 0.787 | +0.001 | 0.040 |
| bags–shoes | 23360 | 0.794 | 0.795 | +0.001 | 0.032 |
| bottoms–tops | 15946 | 0.729 | 0.729 | +0.001 | 0.040 |
| sunglasses–tops | 3902 | 0.717 | 0.717 | +0.001 | 0.039 |
| all-body–outerwear | 3110 | 0.784 | 0.785 | +0.001 | 0.042 |
| bottoms–hats | 1958 | 0.763 | 0.763 | +0.000 | 0.047 |
| jewellery–outerwear | 8076 | 0.734 | 0.734 | -0.000 | 0.039 |
| all-body–hats | 1024 | 0.826 | 0.826 | -0.000 | 0.037 |
| bottoms–scarves | 1802 | 0.739 | 0.739 | -0.001 | 0.031 |
| accessories–hats | 588 | 0.782 | 0.781 | -0.001 | 0.024 |
| hats–outerwear | 1384 | 0.789 | 0.788 | -0.001 | 0.034 |
| all-body–shoes | 10038 | 0.825 | 0.824 | -0.001 | 0.040 |
| shoes–tops | 17712 | 0.768 | 0.767 | -0.001 | 0.035 |
| accessories–bags | 2894 | 0.751 | 0.750 | -0.001 | 0.034 |
| all-body–bags | 9072 | 0.797 | 0.795 | -0.001 | 0.038 |
| outerwear–tops | 7020 | 0.753 | 0.752 | -0.001 | 0.050 |
| accessories–outerwear | 1756 | 0.729 | 0.727 | -0.002 | 0.036 |
| hats–shoes | 3014 | 0.819 | 0.817 | -0.002 | 0.043 |
| bags–tops | 15170 | 0.757 | 0.755 | -0.002 | 0.031 |
| bags–hats | 2592 | 0.784 | 0.782 | -0.003 | 0.038 |
| bottoms–sunglasses | 3508 | 0.704 | 0.701 | -0.003 | 0.040 |
| jewellery–scarves | 2192 | 0.737 | 0.734 | -0.004 | 0.035 |
| accessories–tops | 2832 | 0.729 | 0.726 | -0.004 | 0.040 |
| outerwear–shoes | 10560 | 0.771 | 0.766 | -0.005 | 0.041 |
| accessories–sunglasses | 708 | 0.729 | 0.724 | -0.005 | 0.031 |
| bags–bottoms | 13336 | 0.742 | 0.737 | -0.006 | 0.032 |
| bags–sunglasses | 5022 | 0.741 | 0.736 | -0.006 | 0.037 |
| bags–outerwear | 9760 | 0.767 | 0.761 | -0.006 | 0.034 |
| outerwear–scarves | 1518 | 0.734 | 0.727 | -0.006 | 0.033 |
| jewellery–sunglasses | 4264 | 0.721 | 0.714 | -0.007 | 0.039 |
| scarves–shoes | 2666 | 0.769 | 0.762 | -0.007 | 0.035 |
| bottoms–outerwear | 7002 | 0.738 | 0.730 | -0.008 | 0.042 |
| bottoms–shoes | 15636 | 0.749 | 0.741 | -0.009 | 0.037 |
| accessories–scarves | 656 | 0.759 | 0.749 | -0.009 | 0.019 |
| accessories–all-body | 1154 | 0.791 | 0.781 | -0.010 | 0.032 |
| all-body–sunglasses | 1728 | 0.754 | 0.741 | -0.014 | 0.044 |
| accessories–bottoms | 2464 | 0.720 | 0.701 | -0.019 | 0.036 |

Pair-weighted mean gain: -0.0004; 27/58 type pairs improve.
