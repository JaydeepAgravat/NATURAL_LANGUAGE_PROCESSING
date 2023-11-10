# Text Preprocessing

## What is Text Preprocessing ?

- Text preprocessing is a crucial step in NLP that involves cleaning and transforming raw text data into a format that is suitable for analysis or further processing by machine learning algorithms.
- The goal is to prepare the text data in a way that makes it easier for computers to understand and extract meaningful information.
- Text preprocessing typically includes several tasks:

## 1. Lowercasing

Unifying the text by converting all characters to lowercase, ensuring consistent treatment of words regardless of case.

```py
Series.str.lower()
```

## 2. Removing HTML Tags

Eliminating HTML tags and special characters, particularly relevant when dealing with text sourced from web pages.

```py
def remove_html_tags(text):
    # Remove HTML tags
    cleaned_text = re.sub(r'<.*?>', '', text)
    return cleaned_text.strip()
```

## 3. Removing URLs

Discarding Uniform Resource Locators (URLs), a standard procedure in text preprocessing to enhance the cleanliness of the data.

```py
def remove_url(description):
    if bool(re.search(r'https?://\S+', description)):
        description = re.sub(r'https?://\S+', '', description)
    return description
```

```py
def contains_url(description):
    url_pattern = r'https?://\S+'
    return bool(re.search(url_pattern, description))
```

## 4. Removing Emojis

Extracting emojis from text, a common step in preprocessing, especially when dealing with informal or expressive content.

```py
def remove_emojis(text):
    # Define a regular expression pattern to match emoji characters
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emoticons
                               u"\U0001F300-\U0001F5FF"  # Symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # Transport & map symbols
                               u"\U0001F700-\U0001F77F"  # Alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               "]+", flags=re.UNICODE)

    # Use re.sub to replace emoji characters with an empty string
    cleaned_text = emoji_pattern.sub('', text)
    return cleaned_text
```

```py
def contains_emoji(description):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # Emojis in the emoticons block
                               u"\U0001F300-\U0001F5FF"  # Emojis in the symbols & pictographs block
                               u"\U0001F680-\U0001F6FF"  # Emojis in the transport & map symbols block
                               u"\U0001F700-\U0001F77F"  # Emojis in the alchemical symbols block
                               u"\U0001F780-\U0001F7FF"  # Emojis in the Geometric Shapes Extended block
                               u"\U0001F800-\U0001F8FF"  # Emojis in the Supplemental Arrows-C block
                               "]+", flags=re.UNICODE)
    return bool(emoji_pattern.search(description))
```

## 5. Removing Punctuation

Clearing the text of unnecessary punctuation marks that often contribute little to the meaningful analysis.

```py
def remove_punctuation(text):  
    # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)

    # Use the translate method to remove punctuation
    cleaned_text = text.translate(translator)

    return cleaned_text
```

## 6. Spelling Correction

Rectifying spelling errors, a crucial text processing task aimed at enhancing the accuracy of subsequent analyses.

```py
# Define a function to correct spelling using TextBlob
def get_correct_spelling(text):
    blob = TextBlob(text)
    corrected_text = blob.correct()
    return str(corrected_text)
```

```py
# Create a Speller instance
spell = Speller(lang='en',fast=True)

# Define a function to correct spelling using autocorrect
def get_correct_spelling(text):
    corrected_text = spell(text)
    return corrected_text
```

## 7. Chat Word Treatment

Managing chat-specific language, including abbreviations and informal expressions, to facilitate more effective handling of informal communication.

```py
def preprocess_chat_text(text):
    text = remove_html_tags(text)            # Remove HTML tags  
    text = remove_punctuation(text)          # Remove punctuation
    text = text.lower()                      # Lowercase the text
    text = expand_contractions(text)         # Expand contractions
    text = replace_chat_abbreviations(text)  # Replace chat abbreviations
    return text

def expand_contractions(text):
    expanded_text = contractions.fix(text)
    return expanded_text

def remove_html_tags(text):
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_extra_whitespaces(text):
    return ' '.join(text.split())

def replace_chat_abbreviations(text):
    new_text = []
    chat_words = {
        "u": "you",
        "r": "are",
        "brb": "be right back",
        "lol": "laugh out loud",
        "omg": "oh my god"
    }
    for w in text.split():
        if w in chat_words:
            new_text.append(chat_words[w])
        else:
            new_text.append(w)
    return ' '.join(new_text)


# Example chat-like text
chat_text = "It's there! <b>OMG</b> u r so funny lol brb!  "

# Preprocess the chat text
cleaned_text = preprocess_chat_text(chat_text)

print("Original text:", chat_text)
print("Cleaned text:", cleaned_text)
# Original text: It's there! <b>OMG</b> u r so funny lol brb!  
# Cleaned text: its there oh my god you are so funny laugh out loud be right back
```

## 8. Removing Stopwords

Eliminating frequently used but contextually unimportant words (e.g., "the," "is," "and") to focus on more meaningful content.

```py
'''i, me, my, myself, we, our, ours, ourselves, you, you're, you've, you'll, you'd, your, yours, 
yourself, yourselves, he, him, his, himself, she, she's, her, hers, herself, it, it's, its, itself,
they, them, their, theirs, themselves, what, which, who, whom, this, that, that'll, these, those, am,
is, are, was, were, be, been, being, have, has, had, having, do, does, did, doing, a, an, the, and,
but, if, or, because, as, until, while, of, at, by, for, with, about, against, between, into, through,
during, before, after, above, below, to, from, up, down, in, out, on, off, over, under, again, further,
then, once, here, there, when, where, why, how, all, any, both, each, few, more, most, other, some,
such, no, nor, not, only, own, same, so, than, too, very, s, t, can, will, just, don, don't, should,
should've, now, d, ll, m, o, re, ve, y, ain, aren, aren't, couldn, couldn't, didn, didn't, doesn,
doesn't, hadn, hadn't, hasn, hasn't, haven, haven't, isn, isn't, ma, mightn, mightn't, mustn,
mustn't, needn, needn't, shan, shan't, shouldn, shouldn't, wasn, wasn't, weren, weren't, won,
won't, wouldn, wouldn't"
'''

def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))  # Use English stop words
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text
```

## 9. Handling Numbers

Addressing numerical values by either converting them into words or removing them, depending on the analytical context.

```py
def remove_numbers(text):
    pattern = r'\d+'
    return re.sub(pattern, '', text)
```

## 10. Remove Multiple White Spaces

Condensing excessive white spaces to a single space, reducing distraction and potential interference with algorithm performance.

```py
def remove_whitespace(text):
    return re.sub(r'\s+', ' ', text)
```

## 11. Tokenization

Breaking down the text into smaller units, such as words or sentences, facilitating a better understanding of the text's structural components.

```py
# Define a function to tokenize a string into sentences
def get_tokenize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    return sentences

# Define a function to tokenize a sentence into words
def get_tokenize_words(sentence):
    words = nltk.word_tokenize(sentence)
    return words
```

## 12. Stemming

Reducing words to their base or root form, aiding in the normalization of different word variations (e.g., transforming "running" to "run").

```py
def get_stem_text(text):
    stemmer = PorterStemmer()              # Create a Porter stemmer object
    words = nltk.word_tokenize(text)       # Tokenize the text into individual words
    stemmed_words = [stemmer.stem(word) for word in words] # Stem each word in the list
    stemmed_text = " ".join(stemmed_words) # Join the stemmed words back into a string
    return stemmed_text
```

## 13. Lemmatization

Treating different forms of words as the same, for instance, converting "better" to its base form, "good." This step contributes to a more comprehensive and accurate analysis.

```py
def get_lemmatize_text(text):
    # Initialize the WordNet Lemmatizer
    wn = nltk.WordNetLemmatizer()

    # Lemmatize each word in the text
    lemmatized_words = []
    for word in text.split():
        lemmatized_words.append(wn.lemmatize(word))

    # Return the lemmatized text
    return " ".join(lemmatized_words)
```
