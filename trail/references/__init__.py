"""Reference-artifact builders (method-independent, cached below the metric layer).

References are the evaluation's own models — NOT unlearning methods (the scope
firewall forbids ``methods/``). The gold retrained model (L2) and the LiRA
shadow ensemble (L3) are the two members. Each is keyed without a checkpoint
hash, so a single build amortizes across every method evaluated on the same
data.
"""
