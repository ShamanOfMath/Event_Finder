# -*- coding: utf-8 -*-

"""
This class contains utilities for the venyoo_events_app.

"""


from itertools import chain

from apps.venyoo_events_app.models import Event


class VenyooDocumentUtility(object):

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

                          data_source='TS',

                          ids_to_filter=None,
                          ids_to_exclude=None,

                          join_title_and_summary=False,
                          return_ids=False):
        """
        Return a Python generator of all Event instances that
        fulfill the filter criteria stated by the method's keyword arguments.
        """

        if ids_to_filter is None or ids_to_filter == 'None':
            ids_to_filter = []
        if ids_to_exclude is None or ids_to_exclude == 'None':
            ids_to_exclude = []

        if join_title_and_summary and data_source in ('T', 'S'):
            raise ValueError('chaining can only be used if data_source="TS"')

        if ids_to_filter and ids_to_exclude:
            raise ValueError(
                '"ids_to_filter" and "ids_to_exclude" cannot be defined at the same time')
        elif ids_to_filter:
            queryset = 'Event.objects.filter(id__in=' + str(ids_to_filter) + ')'
        elif ids_to_exclude:
            queryset = 'Event.objects.exclude(id__in=' + str(ids_to_exclude) + ')'
        else:
            queryset = 'Event.objects.all()'

        if return_ids:
            database_options = ['event.id']
        elif data_source == 'T':
            database_options = ['event.title']
        elif data_source == 'S':
            database_options = ['event.summary']
        elif data_source == 'TS':
            if join_title_and_summary:
                database_options = ['event.title', 'event.summary']
            else:
                database_options = ['" ".join((event.title, event.summary))']
        else:
            raise ValueError(data_source, 'is an unsupported value for data_source')

        generators = []
        for database_option in database_options:
            gen = '(' + database_option  + ' for event in ' + queryset  + ')'
            generators.append(eval(gen))

        return chain.from_iterable(generators)
