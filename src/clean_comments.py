from patterns import SUBSTITUTIONS
import re
import string


### data cleaner function
def clean(input_str):
    input_str = input_str.lower()
    
    for sub in SUBSTITUTIONS:
        input_str = re.sub(sub[0], sub[1], input_str)
    
    input_str = input_str.translate(str.maketrans('','', string.punctuation))		# Eliminate punchuation
    input_str = input_str.strip()
    
    return input_str

# print(SUBSTITUTIONS)