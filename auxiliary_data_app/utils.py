# -*- coding: utf-8 -*-

"""
This module contains custom utilities that can be applied to a lot of scenarios
but are not tied to any specific Django functionality.

"""

import gc
import htmlentitydefs
import re
from urlparse import urlparse


class CommonUtils(object):

    """
    This class provides common utility methods that can be useful for various
    use cases.

    """

    @staticmethod
    def chunk_iterable(iterable, n):
        """Chunk an iterable data structure into equally sized portions."""
        if isinstance(iterable, (list, tuple, set, dict)):
            for i in xrange(0, len(iterable), n):
                yield iterable[i:i+n]
        else:
            raise TypeError('Iterable must be of type list, tuple, set or dict')

    @staticmethod
    def unescape(text):
        """
        Remove HTML or XML character references and entities from a text string.

        These numeric character references will be replaced by standard
        unicode characters. This method was directly adopted from:

            http://effbot.org/zone/re-sub.htm#unescape-html

        For more information about this issue, see:

            http://en.wikipedia.org/wiki/Numeric_character_reference
        """
        def fixup(m):
            text = m.group(0)
            if text[:2] == "&#":
                # character reference
                try:
                    if text[:3] == "&#x":
                        return unichr(int(text[3:-1], 16))
                    else:
                        return unichr(int(text[2:-1]))
                except ValueError:
                    pass
            else:
                # named entity
                try:
                    text = unichr(htmlentitydefs.name2codepoint[text[1:-1]])
                except KeyError:
                    pass
            return text # leave as is
        return re.sub("&#?\w+;", fixup, text)

    @staticmethod
    def extract_netlocs(urls):
        """"""
        netlocs = [urlparse(url).netloc for url in urls]
        for i, netloc in enumerate(netlocs):
            netlocs[i] = '.'.join(netloc.split('.')[-2:])
            if ':' in netloc:
                netlocs[i] = netloc[:netloc.index(':')]
        return list(set(netlocs))


class DjangoUtils(object):

    """
    This class provides utilities specific to Django.

    """

    @staticmethod
    def queryset_iterator(queryset, chunksize=100):
        """
        Iterate over a Django Queryset ordered by the primary key.

        This method loads a maximum of chunksize (default: 100) rows in it's
        memory at the same time while Django normally would load all rows at
        once in it's memory. Note that the implementation of the iterator
        does not support ordered query sets.
        This method was directly adopted from:

            http://djangosnippets.org/snippets/1949/
        """
        pk = 0
        last_pk = queryset.order_by('-pk')[0].pk
        queryset = queryset.order_by('pk')
        while pk < last_pk:
            for row in queryset.filter(pk__gt=pk)[:chunksize]:
                pk = row.pk
                yield row
            gc.collect()
