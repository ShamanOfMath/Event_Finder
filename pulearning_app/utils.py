# -*- coding: utf-8 -*-

"""
This module contains utilities for the pulearning_app.

"""


from collections import OrderedDict
import sys

from apps.pulearning_app.forms import CRAWLED_WEBPAGE_FORM_ENTRIES, VENYOO_FEATURE_FORM_ENTRIES


class PULearningUtils(object):

    """
    This class contains utility methods for the pulearning_app.

    """

    @classmethod
    def get_user_input(cls, data_source='positive'):
        """
        Ask for the positive and unlabeled data sources, if one
        of the classification management commands is executed from the command
        line and not from the web interface.

        """
        if data_source not in ('positive', 'unlabeled'):
            raise ValueError(
                'attribute data_source must be "positive" or "unlabeled"')

        if data_source == 'positive':
            options = OrderedDict(VENYOO_FEATURE_FORM_ENTRIES[1:])

        elif data_source == 'unlabeled':
            options = OrderedDict(CRAWLED_WEBPAGE_FORM_ENTRIES[1:])

        keys_mapping = dict(
            zip(xrange(1, len(options) + 1), options.keys()))
        values_mapping = dict(
            zip(xrange(1, len(options) + 1), options.values()))

        for i in values_mapping:
            print i, ':', values_mapping[i]

        user_input = None
        while True:
            user_input = raw_input('Enter a number (or type "exit" to quit): ')
            if user_input == 'exit':
                sys.exit()
            try:
                user_input = int(user_input)
                if user_input in keys_mapping:
                    break
                else:
                    print 'This number is out of range. Please try again.'
            except ValueError:
                print 'This is not a valid number. Please try again.'

        return keys_mapping[user_input]
