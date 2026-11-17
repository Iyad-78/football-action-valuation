# Colab workflow

Colab is an optional compute worker, not the source of truth. GitHub remains the
canonical repository.

After the repository is published:

```python
!git clone <REPOSITORY_URL>
%cd football-action-valuation
!pip install -e ".[boosting]"
!fav data download
!fav run-actions --models logistic lightgbm mlp gru transformer --epochs 6
```

Develop and test data preparation locally on a small sample first. Use Colab
for GRU/Transformer training runs, then export configuration,
metrics and small model artifacts back to the project. Never commit raw
StatsBomb event files to Git. The current implementation explicitly uses CPU
with two threads; it does not automatically accelerate on Colab's GPU. PyTorch
must be available (normally preinstalled in Colab).

