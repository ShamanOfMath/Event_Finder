# -*- coding: utf-8 -*-

"""
This module contains all the forms displayed in this web application.

"""


from decimal import Decimal
from itertools import combinations

from django import forms

from apps.crawled_webpages_app.models import Domain


def _get_form_entries(feature_locations, return_single_elems=False):
    """Return all entry combinations for the unlabeled data source setting."""
    entries = []

    if return_single_elems:
        comb_range = xrange(1, 2)
    else:
        comb_range = xrange(2, len(feature_locations)+1)

    for i in comb_range:
        combs = combinations(feature_locations.keys(), i)
        for comb in combs:
            entry = (
                ''.join(comb),
                ' + '.join([feature_locations[elem] for elem in comb]))
            entries.append(entry)
    return entries


CRAWLED_WEBPAGE_FEATURE_LOCATIONS = {
    'Em': 'Emphs',
    'He': 'Headings',
    'Li': 'Lists',
    'Me': 'Meta',
    'Ta': 'Tables',
    'Ti': 'Title',
    'Ur': 'Url'
}

ADDITIONAL_FORM_ENTRIES = [
    ('TeMe', 'Text + Meta'),
    ('TeTi', 'Text + Title'),
    ('TeUr', 'Text + Url')
]

EMPTY_CHOICE = ('', '----------')

CRAWLED_WEBPAGE_FORM_ENTRIES = \
    sorted(
        _get_form_entries(
            CRAWLED_WEBPAGE_FEATURE_LOCATIONS,
            return_single_elems=True) + [('Te', 'Text')]) + \
    sorted(
        _get_form_entries(
            CRAWLED_WEBPAGE_FEATURE_LOCATIONS) + ADDITIONAL_FORM_ENTRIES)

CRAWLED_WEBPAGE_FORM_ENTRIES.insert(0, EMPTY_CHOICE)

VENYOO_FEATURE_FORM_ENTRIES = [
    EMPTY_CHOICE,
    ('S', 'Summary'),
    ('T', 'Title'),
    ('TS', 'Title + Summary')
]

DOMAIN_NAMES = sorted(list(Domain.objects.values_list('name')))
for i, domain_name in enumerate(DOMAIN_NAMES):
    DOMAIN_NAMES[i] = domain_name * 2


class WebPageCrawlerForm(forms.Form):

    """
    This is the webpage crawler form enabling the user to crawl new webpages
    from the given domain and save it to the database.

    """

    url_to_crawl = forms.URLField()
    maximum_number_of_pages = forms.IntegerField(min_value=1)
    delete_previous_pages = forms.BooleanField(
        required=False,
        label='Delete previously crawled pages from this domain?')


class DataAndClassifierSelectionForm(forms.Form):

    """
    This is the data and classifier selection form enabling the user to
    set parameters for both positive and unlabeled data, the RN technique
    to be used, the classifier for the final labeling and parameters for
    feature selection, stemming etc.

    """

    positive_data_source = forms.ChoiceField(
        choices=VENYOO_FEATURE_FORM_ENTRIES)

    unlabeled_data_source = forms.ChoiceField(
        choices=CRAWLED_WEBPAGE_FORM_ENTRIES)

    rn_method_domains = forms.MultipleChoiceField(
        choices=DOMAIN_NAMES,
        label='Domain data input for RN detection')

    classifier_domains = forms.MultipleChoiceField(
        choices=DOMAIN_NAMES,
        required=False,
        label='Domain data input for final labeling')

    hand_labeled_data = forms.BooleanField(
        required=False,
        label='Use hand-labeled data as input for final labeling instead?')

    save_statistics = forms.BooleanField(required=False, label='Save statistics?')

    rn_method = forms.ChoiceField(
        choices=(
            EMPTY_CHOICE,
            ('1dnf', '1-DNF'),
            ('cr', 'Cosine-Rocchio'),
            ('nb', 'Naive Bayes'),
            ('rocchio', 'Rocchio'),
            ('spy', 'Spy')),
        label='Step-1 RN technique')

    sampling_ratio = forms.DecimalField(
        required=False,
        min_value=Decimal('0.0'),
        max_value=Decimal('1.0'))

    iterations = forms.IntegerField(required=False, min_value=1)

    noise_level = forms.DecimalField(
        required=False,
        min_value=Decimal('0.0'),
        max_value=Decimal('1.0'))

    kmeans_clustering = forms.BooleanField(
        required=False,
        label='Enable K-means clustering?')

    classifier = forms.ChoiceField(
        choices=(
            EMPTY_CHOICE,
            ('emnb', 'Naive Bayes using EM'),
            ('svm', 'Support Vector Machine')),
        label='Step-2 classifier for final labeling')

    apply_em = forms.BooleanField(
        required=False,
        label='Apply Expectation-Maximization (EM)?')

    svm_iterative_mode = forms.BooleanField(
        required=False,
        label='Apply iterative mode until convergence?')

    positive_set_extension = forms.ChoiceField(
        choices=(
            EMPTY_CHOICE,
            ('-', 'No Extension'),
            ('emnb', 'Positive Set classified by EM-NB'),
            ('svm', 'Positive Set classified by SVM'),
            ('by_hand', 'Positive Set classified by Hand')))

    tokenizer = forms.ChoiceField(
        choices=(
            EMPTY_CHOICE,
            ('dates', 'Date Expressions'),
            ('numerics', 'Numerics'),
            ('words', 'Words'),
            ('entire_text', 'Entire Text')),
        label='Feature Selection')

    ngram_size = forms.ChoiceField(
        choices=(
            EMPTY_CHOICE,
            ('1', '1'),
            ('2', '2'),
            ('3', '3'),
            ('1,2', '1, 2'),
            ('1,3', '1, 3'),
            ('2,3', '2, 3')),
        label='N-Gram Range')

    apply_stemming = forms.BooleanField(required=False, label='Apply stemming?')


class HaystackSearchForm(forms.Form):

    """
    This is the haystack search form enabling the user to search both
    positive event pages from Venyoo.de and unlabeled webpages.

    """

    search_query = forms.CharField(
        label='Enter your search query')

    search_source = forms.ChoiceField(
        choices=(
            EMPTY_CHOICE,
            ('venyoo_events', 'Venyoo Events'),
            ('crawled_webpages', 'Crawled Webpages'),
            ('both', 'Venyoo Events + Crawled Webpages')),
        label='Where to search?')

    max_results = forms.IntegerField(
        min_value=1,
        label='How many results to show at maximum?')
