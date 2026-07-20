# Ensemble feature ablation (4-seed holdout means)

| config          |   pooled_mcc |   macro_mcc |
|:----------------|-------------:|------------:|
| full            |       0.3146 |      0.2193 |
| - kiwi score    |       0.318  |      0.222  |
| - xl score      |       0.3083 |      0.2134 |
| - span mass     |       0.3163 |      0.2184 |
| - log length    |       0.3069 |      0.1868 |
| - lp one-hots   |       0.2074 |      0.2185 |
| kiwi only (+lp) |       0.2827 |      0.1709 |
| xl only (+lp)   |       0.3066 |      0.1854 |
