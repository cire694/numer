"""
Idea: trees are better for tabular data (recall paper), but we want to weigh each prediction differently depending on the era
- Mixture of Experts

Two stage model
- Neural Network on top of trees, to learn how to weigh the votes
    - we'll need to add some "temporal" element (e.g. but what exactly?)
    - One idea: weights = softmax(era_embed @ key.weights.T / sqrt(d_model))
        - era embedding: learned representation of era (linear model)
            - should there be a updatable state? or should this just be a function of the current features
                - e.g. momentum, etc (might already be a feature)
            - or mean + std for each feature groups across all rows
        - tree's keys: learned representation of each expert (using nn.Embedding)
            - pass in the tree's prediction
        - How many layers? I don't have any intuition for this. Should it just be linear?
- Trees will have heavy regularization: 
    - lr: 0.001
    - estimators: 3000-10_000+
    - colsample_bytree: 0.1-0.2
    - min_data_in_leaf: 1000+

"""