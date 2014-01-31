# -*- coding: utf-8 -*-

"""
This class contains utilities for the crawled_webpages_app.

"""


from itertools import chain
from urlparse import urlparse
import re

from apps.crawled_webpages_app.models import CrawledWebpage


class CrawledWebpageUtility(object):

    """
    This class provides one of the main functionalities of this application.
    It contains a method named webpage_generator that returns a Python
    generator for a specific Django queryset. This generator will be used
    as input to the classification methods.

    Generators are much more memory-efficient than usual lists, that is why
    they are preferred as the data structure for this web application.

    """

    @classmethod
    def webpage_generator(cls,
                           data_source='Te',

                           filter_positives=None,
                           exclude_positives=None,

                           filter_rn=None,
                           exclude_rn=None,

                           filter_domains=None,
                           filter_ids=None,

                           filter_hand_labeled_pages=False,
                           exclude_hand_labeled_pages=False):
        """
        Return a Python generator of all CrawledWebpage instances that
        fulfill the filter criteria stated by the method's keyword arguments.
        """

        # Return empty lists for arguments filter_domains and filter_ids if None
        if filter_domains is None or filter_domains == 'None':
            filter_domains = []
        if filter_ids is None or filter_ids == 'None':
            filter_ids = []

        # Take a standard queryset as the starting point for all further queries
        queryset = CrawledWebpage.objects.all()

        # Filter this standard queryset by ONLY ONE of the following ways
        if filter_hand_labeled_pages:
            queryset = queryset.filter(is_eventpage__in=['Y', 'N'])
        elif exclude_hand_labeled_pages:
            queryset = queryset.exclude(is_eventpage__in=['Y', 'N'])
            if filter_domains:
                queryset = queryset.filter(domain__name__in=filter_domains)
        elif filter_domains:
            queryset = queryset.filter(domain__name__in=filter_domains)
        elif filter_ids:
            queryset = queryset.filter(id__in=filter_ids)

        # Further constrain this queryset by RN annotations
        if filter_rn == '1dnf':
            queryset = queryset.filter(is_1dnf_reliable_negative='N')
        elif filter_rn == 'cr':
            queryset = queryset.filter(is_cosine_rocchio_reliable_negative='N')
        elif filter_rn == 'nb':
            queryset = queryset.filter(is_nb_reliable_negative='N')
        elif filter_rn == 'rocchio':
            queryset = queryset.filter(is_rocchio_reliable_negative='N')
        elif filter_rn == 'spy':
            queryset = queryset.filter(is_spy_reliable_negative='N')
        if exclude_rn == '1dnf':
            queryset = queryset.exclude(is_1dnf_reliable_negative='N')
        elif exclude_rn == 'cr':
            queryset = queryset.exclude(is_cosine_rocchio_reliable_negative='N')
        elif exclude_rn == 'nb':
            queryset = queryset.exclude(is_nb_reliable_negative='N')
        elif exclude_rn == 'rocchio':
            queryset = queryset.exclude(is_rocchio_reliable_negative='N')
        elif exclude_rn == 'spy':
            queryset = queryset.exclude(is_spy_reliable_negative='N')

        # Further constrain this queryset by positive annotations
        if filter_positives == 'emnb':
            queryset = queryset.filter(em_nb_tagging='P')
        elif filter_positives == 'svm':
            queryset = queryset.filter(svm_tagging='P')
        elif filter_positives == 'by_hand':
            queryset = queryset.filter(is_eventpage='Y')
        if exclude_positives == 'emnb':
            queryset = queryset.exclude(em_nb_tagging='P')
        elif exclude_positives == 'svm':
            queryset = queryset.exclude(svm_tagging='P')
        elif exclude_positives == 'by_hand':
            queryset = queryset.exclude(is_eventpage='Y')

        model_fields = {
            'Em': (webpage.html_emphs for webpage in queryset),
            'He': (webpage.html_headings for webpage in queryset),
            'Li': (webpage.html_lists for webpage in queryset),
            'Me': (webpage.html_meta for webpage in queryset),
            'Ta': (webpage.html_tables for webpage in queryset),
            'Te': (webpage.html_text for webpage in queryset),
            'Ti': (webpage.html_title for webpage in queryset),
            'Ur': (' '.join(
                (urlparse(webpage.url).path,
                 urlparse(webpage.url).query)) for webpage in queryset)
        }

        webpage_parts = [model_fields[field] for field in
                     re.findall(r'[A-Z][^A-Z]*', data_source)]

        webpage_ids = (webpage.id for webpage in queryset)

        return chain.from_iterable(webpage_parts), webpage_ids, queryset.count()
