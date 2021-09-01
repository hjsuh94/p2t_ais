Plans for experiments. 

## Carrots

# Prediction of Rewards on Equation Error

1. Hypothesis 1: We should penalize equation error z more harshly then reward. 
    - Required Evidence: 
        - lambda = 1e-1, 1e-2, 1e-3
    - Method: compare equation error on rewards.
    - Conclusion: No meaningful difference has been found. Hypothesis disproved.

2. Hypothesis 2: In general, higher dimensions of z lead to lower equation error.
    - Required Evidence:
        - z : 10, 20, 40, 60 on MLPs.
    - Method: compare equation error on rewards.
    - Conclusion: 10 \approx 20 > 40 > 60. Bias-Variance? Larger values of z led to overfitting?

3. Hypotehsis 3: In the linear setting, higher dimensions have more impact on reward prediction accuracy.
    - Required Evidence:
        - z: 10, 20, 40, 60 on linear nets.
    - Method: compare equation error on rewards.

4. Hypothesis 4: In general, using MLPS lead to better reward prediction compared to linear.
    - Required Evidence:
        - Model: Comparison on linear vs. MLP on z = 20
    - Method: compare equation error on rewards.
    - Conclusion: Yes. MLPs are better. Hypothesis Proven.

5. Hypotehsis 5: Predicting pixels can lead to better perforamnce in reward prediction. 
    - Required Evidence:
        - Model: Comparison with / without decompression.
    - Method: compare equation error on rewards.

# Getting good performance closed-loop / combatting with distribution shift.

1. Predicting pixels can lead to better perforamnce in combating distribution shift (better generalization)
    - Required Evidence:
        - Model: : Comparison with / without decompression on closed loop models. 

## Single Integrator Problem

# Interpretation of Latent States 

1. Hypothesis 1: Reward symmetry causes latent states to collapse, draws in two states with equal rewards but different dynamics.
    - Required evidence:
        - z: Plot the latent state to compare within reward level-set (circular) vs. on the gradient.



# Prediction of Rewards on Equation Error 




## Pendulum Problem
