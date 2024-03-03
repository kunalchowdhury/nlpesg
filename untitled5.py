# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 20:19:39 2024

@author: kunal
"""
import transformers
from transformers import pipeline

text="""ESG investing exists within a broader spectrum of investing based on financial and social returns. On one
side of the spectrum, pure financial investment is pursued to maximise shareholder and debtholder value
through financial returns based on absolute or risk-adjusted measures of financial value. At best, it
assumes the efficiency of capital markets will effectively allocate resources to parts of the economy that
maximise benefits, and contributes more broadly to economic development. On the other side of the
spectrum, pure social “investing, such as philanthropy, seeks only social returns, such that the investor
gains from confirming evidence of benefits to segments or all of society, in particular related to
environmental or social benefits, including with regard to human and worker rights, gender equality. Social
impact investing seeks a blend of social return and financial return – but the prioritisation of social or
financial returns depends on the extent to which the investors are willing to compromise one for the other
in alignment with their overall objectives."""

#pipe=pipeline("summarization", model="t5-large")
pipe=pipeline("summarization", model="facebook/bart-large-cnn")
pipe_out=pipe(text)
print(pipe_out[0]['summary_text'])