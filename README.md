# RL Counterfactual Policy Explanations

This repository is contains code for the paper unpublished paper titled "Explaining Black Box Reinforcement Learning
Agents Through Counterfactual Policies".

The code has experiments on generating counterfactual explanations for RL policies via counterfactual policies.

## Managing dependencies

This repository uses `pip-tools` to manage dependencies.
Requirements are specified in input files, e.g. [requirements.in](requirements.in).
To compile them, install `pip-tools` (`pip install pip-tools`) and run

```
pip-compile requirements.in
```

It will produce a `requirements.txt` file.
