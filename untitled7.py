# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 09:02:14 2024

@author: kunal
"""

from haystack.nodes import PDFToTextConverter
from pathlib import Path
converter = PDFToTextConverter(
    remove_numeric_tables=True,
    valid_languages=["de","en"]
)
docs = converter.convert(file_path=Path("sample1.pdf"), meta=None)