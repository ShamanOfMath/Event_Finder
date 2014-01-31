# -*- coding: utf-8 -*-

"""
This module contains the database models for the crawled_webpages_app.

Each model defines the essential fields, relations and behaviors of
the data that should be stored. Generally, each model maps to a
single database table.

For an overview about Django's models, see:
    https://docs.djangoproject.com/en/1.5/topics/db/models/

For a detailed explanation of field options and field types, see:
    https://docs.djangoproject.com/en/1.5/ref/models/fields/

"""

from django.db import models

from apps.auxiliary_data_app.models import CommonInformation


class CrawledWebpage(CommonInformation):

    """
    The CrawledWebpage model provides fields for storing textual data from
    previously crawled webpages. Additionally, it contains fields for saving
    annotation results output by the various classification methods.

    Foreign key fields
    ------------------
    - domain -
    The domain that this crawled webpage belongs to. Each crawled webpage
    is associated with exactly one domain.

    Fields for storing webpage data
    -------------------------------
    - html_emphs -
    Stores the textual content of emphasized elements represent by the tags
    <em> and <strong>.

    - html_headings -
    Stores the textual content of headings <h1> to <h6>.

    - html_lists -
    Stores the textual content of lists represented by the tags <dl>, <ol>, <ul>.

    - html_meta -
    Stores the webpage's entire <meta> information.

    - html_tables -
    Stores the textual content of tables (<table>).

    - html_title -
    Stores the textual content of the webpage's title (<title>).

    - html_text -
    Stores the textual cotent of the entire webpage body (<body>).

    - url -
    Stores the entire url of the webpage.

    Fields for storing annotation results
    -------------------------------------
    - em_nb_tagging -
    Stores the annotation for the EM-NB classifier.

    - is_1dnf_reliable_negative -
    Stores the annotation for the 1-DNF technique to determine reliable negative
    webpages.

    - is_cosine_rocchio_reliable_negative -
    Stores the annotation for the Cosine-Rocchio technique to determine reliable
    negative webpages.

    - is_eventpage -
    Stores the hand-made annotation of whether a webpage is labeled as positive,
    negative or unlabeled.

    - is_nb_reliable_negative -
    Stores the annotation for the Naive Bayes technique to determine reliable
    negative webpages.

    - is_rocchio_reliable_negative -
    Stores the annotation for the Rocchio technique to determine reliable
    negative webpages.

    - is_spy_reliable_negative -
    Stores the annotation for the Spy technique to determine reliable
    negative webpages.

    - svm_tagging -
    Stores the annotation for the SVM classifier.

    """

    EVENTPAGE_CLASSIFICATION_OPTIONS = (
        ('Y', 'Yes'),
        ('N', 'No'),
        ('-', 'Unknown'))

    RELIABLE_NEGATIVE_OPTIONS = (
        ('N', 'Negative'),
        ('-', 'Unlabeled')
    )

    TAGGING_OPTIONS = (
        ('P', 'Positive'),
        ('N', 'Negative'),
        ('-', 'Unlabeled')
    )

    domain = models.ForeignKey('Domain')

    # Unique constraint ensures that a webpage with exactly the same content
    # will be saved only once to the database.
    html_text = models.TextField(unique=True)
    html_meta = models.TextField(blank=True)
    html_title = models.CharField(max_length=100, blank=True)
    html_headings = models.TextField(blank=True)
    html_lists = models.TextField(blank=True)
    html_tables = models.TextField(blank=True)
    html_emphs = models.TextField(blank=True)

    # Unique constraint ensures that a webpage with exactly the same url
    # will be saved only once to the database.
    url = models.URLField(unique=True)

    is_eventpage = models.CharField(
        max_length=1,
        choices=EVENTPAGE_CLASSIFICATION_OPTIONS,
        default='-',
        verbose_name='is this an event page?')

    is_1dnf_reliable_negative = models.CharField(
        max_length=1,
        choices=RELIABLE_NEGATIVE_OPTIONS,
        default='-',
        verbose_name='1-DNF annotation')

    is_spy_reliable_negative = models.CharField(
        max_length=1,
        choices=RELIABLE_NEGATIVE_OPTIONS,
        default='-',
        verbose_name='Spy annotation')

    is_cosine_rocchio_reliable_negative = models.CharField(
        max_length=1,
        choices=RELIABLE_NEGATIVE_OPTIONS,
        default='-',
        verbose_name='CR annotation')

    is_rocchio_reliable_negative = models.CharField(
        max_length=1,
        choices=RELIABLE_NEGATIVE_OPTIONS,
        default='-',
        verbose_name='Rocchio annotation')

    is_nb_reliable_negative = models.CharField(
        max_length=1,
        choices=RELIABLE_NEGATIVE_OPTIONS,
        default='-',
        verbose_name='NB annotation')

    em_nb_tagging = models.CharField(
        max_length=1,
        choices=TAGGING_OPTIONS,
        default='-',
        verbose_name='EM-NB tagging')

    svm_tagging = models.CharField(
        max_length=1,
        choices=TAGGING_OPTIONS,
        default='-',
        verbose_name='SVM tagging')


    def __unicode__(self):
        return u'{0}'.format(self.url)


class Domain(CommonInformation):

    """
    The Domain model provides fields for storing domain names for a set
    of crawled webpages.

    A domain is associated with one or more crawled webpages.

    Fields
    ------
    - name -
    The name of this domain.

    """

    name = models.CharField(max_length=50, unique=True)

    def __unicode__(self):
        """"""
        return u'{0}'.format(self.name)
