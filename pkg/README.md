Generate 10 random regression data sets with Gaussian errors, fit and predict an initial model, then create model summary documents that get saved to a Models MongoDB collection.

Steps to simulate model summaries and save to MongoDB
```
conda env create -f environment.yml
python ./summarize.py
```