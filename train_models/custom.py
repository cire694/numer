"""
Group features by correlation, train a tree on each group, then
use a feedforward NN to weigh the regression results (kind of like MoE)

Two stages: 
1. Feature grouping + trees: 
    - Grouping correlated features and training separate trees forces each tree to specialize
    - highly correlated (magnitude) -> measuring same underlying thing

2. Dynamic weights: era-level context (which features were post predictive recently)
    - learning dynamic weights that adapt to market regime

Could potentially be a bad idea
1. perhaps a combination signals uncorrelated with each other is actually meaningful -> splitting
    into features that are correlated with each other is a bad idea
2. How do I make the weights dynamic? Some sort of attention mechanism?
    - mixture of experts with learned gating network
3. overfitting is a high risk, two stage model has more params than one stage model
4. NN might perform worse than equal weighting -> should test in stages

Two opposing philosophies: 
1. group by correlation: each tree has a specialized "theme"
2. group by decorrelation: deliberately put uncorrelated features in the same tree. 
    Assumption is that feature interactions across themes are what drives prediction
Neither are obviously correct; this is an empirical question
Alternative: don't split at all, let the tree learn itself

Numerai has its own thematic groupings:

feature_groups.keys(): 
    dict_keys(['small', 'medium', 'all', 'v2_equivalent_features', 'v3_equivalent_features', 'fncv3_features', 'intelligence', 'charisma', 'strength', 'dexterity', 'constitution', 'wisdom', 'agility', 'serenity', 'sunshine', 'rain', 'midnight', 'faith'])
    - I think each key (beyond small, medium, all) is a feature group (not sure about the v2, v3)

Idea: use attention to weigh era + tree output: 
- era embedding: learned representation of era (linear model)
- tree's keys: learned representation of each expert (using nn.Embedding)

scores = era_embed @ key.weight.T / sqrt(d_model)
then take softmax on the last dim

We'll need heavy regularization: 
- heavy dropout on gates (I don't have gates anymore though ...?)
- entropy regularization: penalize gates for being too confident (assigning all weights to one expert)
    - doubts about this though ...
- 
"""

"""
Stage 1: feature grouping (no NN)
    Each feature group gets a lightGBM -> take the equal weight average, find Sharpe
        - also explore what the feature groups are; do they have high/low correations? are they disjoint?

Stage 2: static weights
    Same trees as stage 1, but add ridge regression 
    Q: does leared weighting beat equal weighting?
        - if not, NN won't help either

Stage 3: dynamic weights
    same trees as stage 1, but add the attention mechanism
    Q: does era-conditioning help over static weights?
"""

"""
['intelligence', 'charisma', 'strength', 'dexterity', 'constitution', 'wisdom', 'agility', 'serenity', 'sunshine', 'rain', 'midnight', 'faith']

These are the thematic groups

"""

#what target should i do? num_trees 41 targets * these 12 feature groups? seems dtm

# Stage 1 is done using lightGBM.
# Add ridge ridge regression