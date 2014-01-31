# -*- coding: utf-8 -*-

"""
This module contains classes and methods for feature extraction.

Feature extraction is one essential part in the classification process and
is configured in the data and classifier selection form.

"""


from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer

from apps.auxiliary_data_app.tokenizers import ClassifierFeatureTokenizer


class ExtendedCountVectorizer(CountVectorizer):

    """
    This class inherits from the class

        sklearn.feature_extraction.text.CountVectorizer

    in order to add stemming support for German provided by the class

        nltk.stem.snowball.SnowballStemmer

    Stemming can be enabled in the data and classifier selection form.

    Scikit-learn (sklearn) is an open source toolkit for machine learning
    written in Python. The CountVectorizer converts a collection of text
    documents to a matrix of token counts. For details, see:

        http://scikit-learn.org/0.13/modules/generated
        /sklearn.feature_extraction.text.CountVectorizer.html
        #sklearn.feature_extraction.text.CountVectorizer

    It is used in very implemented algorithm to process the classifiers' input
    features.

    The SnowballStemmer class is a Python port of Martin Porter's stemmers
    for 12 different languages. It is named after "Snowball", Porter's own
    language for writing stemming algorithms.

    Porter's algorithms can be investigated here:

        http://snowball.tartarus.org/

    NLTK (Natural Language Toolkit) is an open source toolkit for language
    processing written in Python.

    The documentation of the Snowball class can be found under:

        http://nltk.org/api/nltk.stem.html#module-nltk.stem.snowball

    """

    def __init__(self, input='content', charset='utf-8',
                 charset_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=ur'(?u)\b\w\w+\b',
                 ngram_range=(1, 1),
                 min_n=None, max_n=None, analyzer='word',
                 max_df=1.0, min_df=2, max_features=None,
                 vocabulary=None, binary=False, dtype=long, stemming=False):

        super(ExtendedCountVectorizer, self).__init__(
            input=input,
            charset=charset,
            charset_error=charset_error,
            strip_accents=strip_accents,
            lowercase=lowercase,
            preprocessor=preprocessor,
            tokenizer=tokenizer,
            stop_words=stop_words,
            token_pattern=token_pattern,
            ngram_range=ngram_range,
            min_n=min_n,
            max_n=max_n,
            analyzer=analyzer,
            max_df=max_df,
            min_df=min_df,
            max_features=max_features,
            vocabulary=vocabulary,
            binary=binary,
            dtype=dtype)

        self.stemming = stemming

    def build_stemmer(self):
        """Return a callable that stems a list of tokens if each token matches
        one of the patterns defined in

            from apps.auxiliary_data_app.tokenizers import ClassifierFeatureTokenizer
        """
        if not self.stemming:
            return lambda tokens: tokens
        german_stemmer = SnowballStemmer('german')
        return lambda tokens: (german_stemmer.stem(token)
            for token in tokens if ClassifierFeatureTokenizer._pattern_url.match(token) or
                                   ClassifierFeatureTokenizer._pattern_word.match(token) or
                                   ClassifierFeatureTokenizer._pattern_entire_text.match(token))

    def build_analyzer(self):
        """Return a callable that handles preprocessing and tokenization."""
        if hasattr(self.analyzer, '__call__'):
            return self.analyzer

        preprocess = self.build_preprocessor()

        if self.analyzer == 'char':
            return lambda doc: self._char_ngrams(preprocess(self.decode(doc)))

        elif self.analyzer == 'char_wb':
            return lambda doc: self._char_wb_ngrams(
                preprocess(self.decode(doc)))

        elif self.analyzer == 'word':
            stop_words = self.get_stop_words()
            stem = self.build_stemmer()
            tokenize = self.build_tokenizer()

            return lambda doc: self._word_ngrams(
                stem(tokenize(preprocess(self.decode(doc)))), stop_words)

        else:
            raise ValueError('%s is not a valid tokenization scheme' %
                             self.analyzer)
