# -*- coding: utf-8 -*-

"""
This class contains SearchIndex subclasses to be used by the search engine

    Whoosh: https://bitbucket.org/mchaput/whoosh/wiki/Home

"""


from haystack import indexes

from apps.venyoo_events_app.models import Event


class EventIndex(indexes.SearchIndex, indexes.Indexable):

    """
    This class specifies the search index of the Event model

    """

    text = indexes.CharField(document=True, use_template=True)
    url = indexes.CharField(model_attr='url')

    def get_model(self):
        """"""
        return Event
