SUBSTITUTIONS = [
    (r'\d+', ''),       # Delete digits
    (r"n't", " not "),      # Replace pattern n't -> not
    (r"can't", "cannot "),      # Replace pattern can't -> cannot
    (r"what's", "what is "),        # Replace pattern what's -> what is
    (r"\'s", " "),              # Delete pattern 's
    (r"\'ve", " have "),        # Replace pattern 've -> have
    (r"\'re", " are "),         # Replace pattern 're -> are
    (r"\'d", " would "),        # Replace pattern 'd -> would
    (r"\'ll", " will "),        # Replace pattern 'll -> will
    (r"\'scuse", " excuse "),       # Replace pattern 'scuse -> excuse
    (r"i'm", "i am"),               # Replace pattern i'm -> i am
    (r" m ", " am "),               # Replace pattern m -> am
    ('\s+', ' '),               # Eliminate duplicate whitespaces using wildcards
    ('\W', ' ')             # Delete non word characters
]