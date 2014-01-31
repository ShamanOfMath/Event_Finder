# -*- coding: utf-8 -*-

"""
This module contains the database models for the auxiliary_data_app.

Each model defines the essential fields, relations and behaviors of
the data that should be stored. Generally, each model maps to a
single database table.

For an overview about Django's models, see:
    https://docs.djangoproject.com/en/1.5/topics/db/models/

For a detailed explanation of field options and field types, see:
    https://docs.djangoproject.com/en/1.5/ref/models/fields/

"""


from django.db import models


class CommonInformation(models.Model):

    """
    The CommonInformation model is an abstract base class.

    It provides fields that will be shared by all other models.
    At the moment, these are two fields that register timestamps
    for the first creation and the last modification of a single
    row in a database table.

    """

    timestamp_first_created = models.DateTimeField(
        auto_now_add=True,
        verbose_name='First creation',
        help_text="The date of this instance's first creation")

    timestamp_last_modified = models.DateTimeField(
        auto_now=True,
        verbose_name='Last modification',
        help_text="The date of this instance's last modification")

    class Meta:

        """
        This class provides metadata options for the CommonInformation model.

        """

        abstract = True


class StopWord(CommonInformation):

    """
    The StopWord model provides fields for storing stop words.

    These stop words will be filtered out during the various classification
    processes.

    This model currently contains 909 stop words, 735 for German and 174 for
    English.

    The German stop words have been taken from

        http://solariz.de/wp-content/files/stopwords.txt

    However, only a subset from the total set was used since the total set
    contains some nouns and adjectives that I do not consider as a stop word.

    The English stop words headed by "Default English stopwords list"
    have been taken from

        http://www.ranks.nl/resources/stopwords.html

    """

    LANGUAGES = (
        ('ENG', 'English'),
        ('GER', 'German')
    )

    language = models.CharField(
        max_length=3,
        choices=LANGUAGES,
        default='GER',
        help_text='The language that this stop word belongs to')

    stop_word = models.CharField(
        max_length=25,
        help_text='The stop word itself')

    def __unicode__(self):
        """Return a unicode representation for a StopWord instance."""
        return u'{0}'.format(self.stop_word)


class LanguageIdentificationTrainingData(CommonInformation):

    """
    The LanguageIdentificationTrainingData model provides fields for
    storing large texts for various languages.

    Character n-gram models will be created out of these texts. These n-gram
    models are used for training a classifier whose only purpose at the moment
    is to detect whether a webpage contains data for one specific language.

    The texts are corpora taken from the Leipzig Corpora Collection available at

        http://corpora.informatik.uni-leipzig.de/download.html

    The files with name 'sentences.txt' from the following corpora have been
    used for a total of 22 languages, respectively:

    Arabic:
        http://corpora.uni-leipzig.de/downloads/ara_news_2005-2009_10K-text.tar.gz

    Chinese (Simplified):
        http://corpora.uni-leipzig.de/downloads/zho_news_2007-2009_10K-text.tar.gz

    Danish:
        http://corpora.uni-leipzig.de/downloads/dan_web_2002_10K-text.tar.gz

    Dutch:
        http://corpora.uni-leipzig.de/downloads/nld_web_2002_10K-text.tar.gz

    English:
        http://corpora.uni-leipzig.de/downloads/eng_news_2007_10K-text.tar.gz

    Estonian:
       http://corpora.uni-leipzig.de/downloads/est_newscrawl_2011_10K-text.tar.gz

    Finnish:
        http://corpora.uni-leipzig.de/downloads/fin_web_2002_10K-text.tar.gz

    French:
        http://corpora.uni-leipzig.de/downloads/fra_web_2002_10K-text.tar.gz

    German:
        http://corpora.uni-leipzig.de/downloads/deu_news_2010_10K-text.tar.gz

    Italian:
        http://corpora.uni-leipzig.de/downloads/ita_news_2010_10K-text.tar.gz

    Japanese:
        http://corpora.uni-leipzig.de/downloads/jpn_news_2005_10K-text.tar.gz

    Latvian:
        http://corpora.uni-leipzig.de/downloads/lav_wikipedia_2007_10K-text.tar.gz

    Lithuanian:
       http://corpora.uni-leipzig.de/downloads/lit_wikipedia_2007_10K-text.tar.gz

    Norwegian (Bokmal):
        http://corpora.uni-leipzig.de/downloads/nob_news_2007_10K-text.tar.gz

    Norwegian (Nynorsk):
        http://corpora.uni-leipzig.de/downloads/nno_wikipedia_2007_10K-text.tar.gz

    Persian:
        http://corpora.uni-leipzig.de/downloads/fas_newscrawl_2011_10K-text.tar.gz

    Portuguese:
        http://corpora.uni-leipzig.de/downloads/por-pt_newscrawl_2011_10K-text.tar.gz

    Russian:
        http://corpora.uni-leipzig.de/downloads/rus_news_2010_10K-text.tar.gz

    Spanish:
        http://corpora.uni-leipzig.de/downloads/spa_news_2001-2002_10K-text.tar.gz

    Swedish:
        http://corpora.uni-leipzig.de/downloads/swe_news_2007_10K-text.tar.gz

    Turkish:
        http://corpora.uni-leipzig.de/downloads/tur_news_2005_10K-text.tar.gz

    """

    language = models.CharField(max_length=50, unique=True)
    training_data = models.TextField()

    class Meta:
        verbose_name_plural = 'language identification training data'

    def __unicode__(self):
        """"""
        return u'{0}'.format(self.language)
