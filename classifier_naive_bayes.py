
# *********Let's interact with tool, enter question or statement and system will predict sentiment********
#
#
#
# Enter a question? (quit to exit):I am depressed, need help
# Sentiment:DEPRESSED
# Enter a question? (quit to exit):I am so happy today
# Sentiment:NON-DEPRESSED
# Enter a question? (quit to exit):quit


import pandas as pd
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from math import log
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
from numpy.random import RandomState
from nltk.corpus import stopwords
import csv
import sys
import nltk
nltk.download('punkt')
nltk.download('stopwords')



def process_message(message, lower_case = True, stem = True, stop_words = True, gram = 1):
    if lower_case:
        message = message.lower()
    words = word_tokenize(message)
    words = [w for w in words if len(w) > 1]
    if gram > 1:
        w = []
        for i in range(len(words) - gram + 1):
            w += [' '.join(words[i:i + gram])]
        return w
    if stop_words:
        sw = stopwords.words('english')
        words = [word for word in words if word not in sw]
    if stem:
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words]
    return words


#Corpus, it will store entire Corpus or it can store repective sentiment corpus i.e depressed or non-depressed.
class Corpus:

    #length of unique features in corpus, total unique features in Corpus
    vocab_len = 0
    #Matrix: total tf-idf weight per term in a document within Corpus
    sum_tf_idf_weights_all_terms = 0
    #Total rows in Corpus
    rows_topic = 0
    #tf-idf weight per term in document
    vocab_per_document = 0

    def __init__(self, csv_file, topic):
        #https://medium.com/@rnbrown/more-nlp-with-sklearns-countvectorizer-add577a0b8c8
        #Convert a collection of raw documents to a matrix of TF-IDF features.
        #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
        vectorizer = TfidfVectorizer(stop_words='english', analyzer='word', token_pattern=u'(?ui)\\b\\w*[a-z]+\\w*\\b',
                                     ngram_range=(1, 2))


        #Learn vocabulary and idf, return term-document matrix.
        term_document_matrix = vectorizer.fit_transform(self.doc_generator(csv_file, topic, textcol=1, skipheader=True))

        #Number of unique features
        self.vocab_len = len(vectorizer.get_feature_names())

        col = [i for i in vectorizer.get_feature_names()]
        vocab_per_document = pd.DataFrame(term_document_matrix.todense(), columns=col)
        self.rows_topic = vocab_per_document.shape[0]

        #total sum tf-idf per term in corpus
        self.tf_idf_per_term = vocab_per_document.sum(axis=0, skipna=True)
        sum_tf_idf_weights_all_terms_temp = 0
        for i in self.tf_idf_per_term:
            sum_tf_idf_weights_all_terms_temp += i

        #total tf-idf weight for all terms.
        self.sum_tf_idf_weights_all_terms = sum_tf_idf_weights_all_terms_temp


    #Returns tf-idf weight per term in Corpus
    def get_term_tf_idf(self,term):
        try:
            return self.tf_idf_per_term.loc[term]
        except:
            return 0



    #Document generator, reads CSV, skips header, reads a line in CSV, applies stemming & lamemmatixers
    #https://towardsdatascience.com/stemming-lemmatization-what-ba782b7c0bd8
    def doc_generator(self,filepath, topic, textcol=0, skipheader=True):

        porter = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        with open(filepath) as f:
            reader = csv.reader(f)
            if skipheader:
                next(reader, None)
            if (topic == '-1'):
                for row in reader:
                   stem = self.stemSentence(porter, row[textcol])
                   yield stem
            else:
                for row in reader:
                    if (topic == row[2]):
                        stem = self.stemSentence(porter, row[textcol])
                        yield stem


    def stemSentence(self,porter, sentence):
        token_words = word_tokenize(sentence)

        stem_sentence = []
        for word in token_words:
            stem_sentence.append(porter.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    def lemmatizeSentence(self,lemmatizer, sentence):
        token_words = word_tokenize(sentence)

        stem_sentence = []
        for word in token_words:
            stem_sentence.append(lemmatizer.lemmatize(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)


#Build training set. This class finds tf-idf weights per term in document and calculates unique features and length(size) of features in Corpus
class QuoraClassifier(object):
    Corpus = None
    Corpus_Non_Depressed = None
    Corpus_Depressed = None
    feature_names_size = 0
    total_docs = 0
    Non_Depressed_docs = 0
    Depressed_docs = 0
    p_non_depressed_topic = 0
    p_depressed_topic = 0

    def __init__(self, csv_file):
        print(
            'Converting a collection of ENTIRE Corpus to a matrix of TF-IDF features using TfidfVectorizer with ngram_range(1,2)..')
        #Build entire corpus that includes depressed and non-depressed topics.
        self.Corpus = Corpus(csv_file, '-1')
        #vocabulary lengh (feature names length) in corpus.
        self.feature_names_size = self.Corpus.vocab_len
        #total docs in Corups
        self.total_docs = self.Corpus.rows_topic

        print(
            'Converting a NON-DEPRESSED Corpus to a matrix of TF-IDF features using TfidfVectorizer with ngram_range(1,2)..')
        #non-depressed corpus
        self.Corpus_Non_Depressed = Corpus(csv_file, '0')
        #total non-depressed documents
        self.Non_Depressed_docs = self.Corpus_Non_Depressed.rows_topic
        # sum of all tf-idf term weights for non-depressed documents
        self.non_depressed_sum_tf_idf_weights_all_terms = self.Corpus_Non_Depressed.sum_tf_idf_weights_all_terms

        print(
            'Converting a DEPRESSED Corpus to a matrix of TF-IDF features using TfidfVectorizer with ngram_range(1,2)..')
        # depressed corpus
        self.Corpus_Depressed = Corpus(csv_file, '1')
        self.Depressed_docs = self.Corpus_Depressed.rows_topic
        self.depressed_sum_tf_idf_weights_all_terms = self.Corpus_Depressed.sum_tf_idf_weights_all_terms

        #probability of non-depressed documents
        self.p_non_depressed_topic = log(self.Non_Depressed_docs / self.total_docs)
        # probability of depressed documents
        self.p_depressed_topic = log(self.Depressed_docs / self.total_docs)


    def Naive_Bayes_Classify(self,question):

        #Calculate probability of non-depressed sentiment for given tweet
        probability_non_depressed = 0
        for term in question:
            # Apply Laplace smoothing
            tf_idf_per_term = self.Corpus_Non_Depressed.get_term_tf_idf(term)
            #Use log to preserve precision instead of multiplication
            probability_non_depressed += log((tf_idf_per_term + 1) / (self.non_depressed_sum_tf_idf_weights_all_terms + self.feature_names_size))
        probability_non_depressed += self.p_non_depressed_topic

        # Calculate probability of depressed sentiment for given tweet
        probability_depressed = 0
        for term in question:
            #Laplace smoothing
            tf_idf_per_term = self.Corpus_Depressed.get_term_tf_idf(term)
            probability_depressed += log((tf_idf_per_term + 1) / (self.depressed_sum_tf_idf_weights_all_terms + self.feature_names_size))
        probability_depressed += self.p_depressed_topic

        if (probability_non_depressed >= probability_depressed):
            return 0
        else:
            return 1

    #Run classification on test-data and let system predict sentiment and output it to a file.
    def predict(self,testData):
        result = []
        for i, r in testData.iterrows():
            processed_message = process_message(r['message'])
            result.append(int(self.Naive_Bayes_Classify(processed_message)))

        pd.options.mode.chained_assignment = None
        testData['prediction'] = result
        testData.to_csv('output_testdata_prediction.csv')
        print('Applied classification on test-data, results are in output_testdata_prediction.txt file.. ')
        print('Format of the output_testdata_prediction CSV file >> id (document ID),message (tweet),label (given sentiment), prediction')
        return testData

    #Run the metrics on system prediction on test-data and calculate metrics on human vs system prediction.
    def metrics(self,testData):
        print('Calculating precision, re-call, accurancy, F-score on test-data prediction. This will compare human prediction and system prediction')
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0
        for index, row in testData.iterrows():
            label = int(row['label'])
            prediction = int(row['prediction'])

            true_pos += int(label == 0 and prediction == 0)
            true_neg += int(label == 1 and prediction == 1)
            false_pos += int(label == 1 and prediction == 0)
            false_neg += int(label == 0 and prediction == 1)

            # true_pos += int(label == 1 and prediction == 1)
            # true_neg += int(label == 0 and prediction == 0)
            # false_pos += int(label == 0 and prediction == 1)
            # false_neg += int(label == 1 and prediction == 0)

        precision = true_pos / (true_pos + false_pos)
        recall = true_pos / (true_pos + false_neg)
        Fscore = 2 * precision * recall / (precision + recall)
        accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg)

        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F-score: ", Fscore)
        print("Accuracy: ", accuracy)


if __name__ == '__main__':
    csv_file_name = 'Book11'
    try:
        print('----------------------Train classifer with training data set------------------------------')
        print('Reading  dataset data in file, ' + csv_file_name)
        print('Format of the CSV file >> id (document ID),message (question),label (sentiment)')
        df = pd.read_csv(csv_file_name + '.csv')
        rng = RandomState()

        print('Split data into 98% training data and 2% test data..')
        #Split data set into 98% trining data set and remaining 2% for test data set.
        trainData = df.sample(frac=0.98, random_state=rng)
        testData = df.loc[~df.index.isin(trainData.index)]

        #create 2 seperate files one for training and other for test.
        trainData.to_csv(csv_file_name + '_train.csv', index=False)
        testData.to_csv(csv_file_name + '_test.csv', index=False)

        print('Train the classifier using training data set...')
        #Train the classifier using training data set.
        quoraClassifier = QuoraClassifier(csv_file_name + '_train.csv')

        print('----------------------Apply classifier on test data------------------------------')
        print('Apply Multinomial Naïve Bayes algorithm classifier to generate sentiment i.e. reading  (message) from test data, predict if it\'s non-depressed(0) or depressed(1)')
        #Predict sentiment using Multinomial Naïve Bayes algorithm classifier, generate sentiment i.e. reading tweet (message), predict if it's non-depressed(0) or depressed(1)
        results = quoraClassifier.predict(testData)


        #Calculate precision, recall, F-score and accuracy on prediction - This step will compare human prediction and system prediction and calculates precision, recall..
        print('----------------------------------------------------------')
        quoraClassifier.metrics(results)
        print('----------------------------------------------------------')
        print('\n\n\n*********Let\'s interact with tool, enter question and system will predict sentiment********\n\n\n')
        while True:
            question = input("Enter a question? (quit to exit):")
            if question == "quit":
                break
            else:
                processed_message = process_message(question)
                if (int(quoraClassifier.Naive_Bayes_Classify(processed_message))):
                    print('Sentiment:' + 'DEPRESSED question')
                else:
                    print('Sentiment:' + 'NON-DEPRESSED question')
    except FileNotFoundError:
        print('Reading Kaggle dataset data in file, ' + csv_file_name + ',not found!')
        sys.exit(1)




