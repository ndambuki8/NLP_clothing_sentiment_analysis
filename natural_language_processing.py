import nltk as nl 
import pandas as pd
from nltk import FreqDist
import random
from nltk import classify
from nltk import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem.wordnet import WordNetLemmatizer
import re, string
from nltk.corpus import stopwords



##REMOVES NOISE BY GETTING RID OF UNWANTED PUNCTUATION MARKS AND GIVING ONE NAME TO RELATED WORDS - lemmatizing
def remove_noise(word_tokens, stop_words = ()):

    cleaned_tokens = []

    for token, tag in pos_tag(word_tokens):
        
        token = re.sub("(@[A-Za-z0-9_]+)","", token)

        if tag.startswith("NN"):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        lemmatizer = WordNetLemmatizer()
        token = lemmatizer.lemmatize(token, pos)

        if len(token) > 0 and token not in string.punctuation and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())
    return cleaned_tokens


### MY PROGRAM STARTS HERE BY IMPORTING A WOMEN CLOTHING REVIEW DATASET - AND PICKING ONLY RELEVANT/NEEDED COLUMNS

col_list = ["Unnamed: 0", "Clothing ID", "Age", "Title", "Review Text", "Rating", "Recommended IND", "Positive Feedback Count", "Division Name", "Department Name", "Class Name"]

f = pd.read_csv("text.csv", usecols=col_list, index_col=0)
n = 50
f = f.head(int(len(f)*(n/100)))   ##CHOOSING TO TRAIN X % OF DATASET


t = f["Recommended IND"]    #PICK CATEGORIZATION COLUMN, +VE OR -VE REVIEWS

y = f["Review Text"]   ## PICK GENERAL REVIEWS COLUMN
y = y.astype(str)     ## CONVERT THEM TO STRING
# t = t.astype(str)

negative_reviewSet = ""
positive_reviewSet = ""
bigram_sayingPos = list("")
bigram_sayingNeg = list("")
positiv =[]
negativ = []

##LOOPING THROUGH REVIEW COLUMN AND CATEGORY COLUMN, SEPARATING +VE AND -VE REVIEWS BY CATEGORY
##CREATE BIGRAMS FROM INDIVIDUAL REVIEW SENTENCES COMBINED..CREATE A LIST OF WORD FROM EACH SENTENCE REVIEW - NEGATIV AND POSITIV
for x,i in zip(y,t):   
   
    if i == 1:
        x = x.lower()
        positive_reviewSet = positive_reviewSet+" "+x
        bigram_sayingPos = list(nl.ngrams(x.split(),2)) + bigram_sayingPos
        positiv.append(x.split())
        # print(positive_reviewSet)

    else:
        x = x.lower()
        negative_reviewSet = negative_reviewSet+" "+x
        bigram_sayingNeg = list(nl.ngrams(x.split(),2)) + bigram_sayingNeg
        negativ.append(x.split())
        # print(negative_reviewSet)

# print(positiv)
# print(negativ)


### USE THE LIST OF STOP WORDS TO EXTRACT THE SAME FROM YOUR DATASET - FURTHER CLEANING
stop_words = stopwords.words('english')

positive_cleaned_review_list = []
negative_cleaned_review_list = []

for tokens in positiv:
    positive_cleaned_review_list.append(remove_noise(tokens, stop_words))

for tokens in negativ:
    negative_cleaned_review_list.append(remove_noise(tokens, stop_words))

# print(positiv[4])
# print(positive_cleaned_review_list[4])
# print(negativ[4])
# print(negative_cleaned_review_list[4])


#Checking frequency of words
def get_all_words(words):
    for token in words:
        for token in tokens:
         yield token

all_pos_words = get_all_words(positiv)
all_neg_words = get_all_words(negativ)


# freq_dist_pos = FreqDist(all_pos_words)
freq_dist_pos = FreqDist(all_neg_words)
# print(freq_dist_pos.most_common(10))


#Converting tokens to a dictionary - KEY "+VE OR -VE :   VALUES: X, Y, Z"
def get_reviews_for_model(cleaned_word_list):
    for word_tokens in cleaned_word_list:
        yield dict([token, True] for token in word_tokens)

positive_tokens_for_model = get_reviews_for_model(positiv)
negative_tokens_for_model = get_reviews_for_model(negativ)


#Splitting the dataset for TRaining and Testing the Model
positive_dataset = [(word_dict, "Positive")
                     for word_dict in positive_tokens_for_model]

negative_dataset = [(word_dict, "Negative")
                     for word_dict in negative_tokens_for_model]

dataset = positive_dataset + negative_dataset

random.shuffle(dataset)

train_data = dataset[:15000]
test_data = dataset[15000:]


classifier = NaiveBayesClassifier.train(train_data)
print("Accuracy is:", classify.accuracy(classifier, test_data))   ## PRINTS ACCURACY LEVEL OF MODEL

print(classifier.show_most_informative_features(10))   ### PRINTS LOGS FOR THE TRAINING NEG : POSITIV RATIO


###USER INPUTS A TEXT TO CLASSIFY
test = input("Enter phrase:\t")
test = test.lower()
bigram_test = list(nl.ngrams(test.split(), 2))

print(classifier.classify(dict([token, True] for token in test)))  