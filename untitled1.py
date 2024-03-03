# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 17:07:46 2024

@author: kunal
"""

from datasets import load_dataset

# Print all the available datasets
from huggingface_hub import list_datasets
print([dataset.id for dataset in list_datasets()])