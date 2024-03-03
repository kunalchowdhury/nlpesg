# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 15:58:42 2024

@author: kunal
"""
import spacy
nlp = spacy.load('en_core_web_sm')

def is_question(sent):
    d = nlp(sent)
    token = d[0] # gets the first token in a sentence
    if token.tag_ == "VERB" or token.tag_ == "VB" or token.tag_ == "VBD" or token.tag_ == "VBG" or token.tag_ == "VBN" or token.tag_ == "VBP" or token.tag_ == "VBZ" and token.dep_ == "ROOT": # checks if the first token is a verb and root or not
        return True
    elif token.dep_ == "aux" and token.tag_ == "VBZ":
        return True

    for token in d: # loops through the sentence and checks for WH tokens
        if token.tag_ == "WDT" or token.tag_ == "WP" or token.tag_ == "WP$" or token.tag_ == "WRB":
            return True
    return  False
#Has the firm ever been suspended from PRI signatory list?
#print(is_question("Is the firm a PRI signatory?"))
#print(is_question("Has the firm ever been suspended from PRI signatory list?"))
print(is_question("Please list the year the firm first signed the PRI is a question"))