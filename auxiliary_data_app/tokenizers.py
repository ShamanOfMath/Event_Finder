# -*- coding: utf-8 -*-

"""
This class provides tokenizer classes to be used in the data and classifier
selection form.

"""


import re


class ClassifierFeatureTokenizer(object):

    """
    This class provides tokenizer methods for
        - date expressions
        - numerics (digits and floating point numbers)
        - words
        - all three above combined

    """

    _numbers_prepended_by_zero = '|'.join('0' + str(n) for n in xrange(0, 10))

    _years = '20\d{2}'

    _months = '|'.join(
        (_numbers_prepended_by_zero[3:],
         '|'.join(str(n) for n in xrange(10, 13)),
         '|'.join(str(n) for n in xrange(1, 10))))

    _month_names = u'Januar|Februar|März|April|Mai|Juni|Juli|August|' \
                   u'September|Oktober|November|Dezember|Jan|Feb|Mär|Apr|' \
                   u'Mai|Jun|Jul|Aug|Sept?|Okt|Nov|Dez'

    _days = '|'.join(
        (_numbers_prepended_by_zero[3:],
         '|'.join(str(n) for n in xrange(10, 32)),
         '|'.join(str(n) for n in xrange(1, 10))))

    _hours = '10|11|12|13|14|15|16|17|18|19|20|21|22|23|0?\d(?!\d)'

    _minutes_and_seconds = '|'.join(
        (_numbers_prepended_by_zero,
         '|'.join(str(n) for n in xrange(10, 60))))


    _pattern_whitespace = re.compile(r'\s{2,}')


    _pattern_url = re.compile(
        ur"""
        (?:
            (?:http|ftp)s?://\S+
        )
        """, flags=re.UNICODE|re.VERBOSE|re.IGNORECASE)


    _pattern_word = re.compile(
        ur"""
        (?:
            [^ \W \d _ ]{2,}
        )
        """, flags=re.UNICODE|re.VERBOSE|re.IGNORECASE)


    _pattern_date = re.compile(
        ur"""
        # 0: days
        # 1: months
        # 2: month names
        # 3: years
        # 4: hours
        # 5: minutes and seconds
        (?:
            (?: {0} ) \.
            (?:
                (?: {1} ) \. (?: {3} )?
                |
                \s+ (?: {2} ) (?: \s+ {3} )?
            )
            |
            (?: {3} )
        )
        |
        (?:
            (?: {4} )
            (?:
                (?:
                    (?:
                        [.:] (?: {5} )
                    )
                    (?:
                        [.:] (?: {5} )
                    )?
                    (?: \s+ Uhr )?
                )
                |
                (?:
                    \s+ Uhr
                    (?:
                        \s+
                        (?: {5} )
                        (?!
                            \.
                            (?:
                                \d | \s+ (?: {2} )
                            )
                        )
                    )?
                )
            )
        )
        """.format(_days, _months, _month_names, _years, _hours,
                   _minutes_and_seconds),
        flags=re.UNICODE|re.VERBOSE|re.IGNORECASE)


    _pattern_numeric = re.compile(
        ur"""
        (?:
            \b \d{1,3} (?: \. \d{3} )+ (?: , \d+ )? \b
            |
            \d+ (?: , \d+ \b )?
        )
        """, flags=re.UNICODE|re.VERBOSE|re.IGNORECASE)


    _pattern_entire_text = re.compile(
        ur"""
        {0}
        |
        {1}
        |
        {2}
        |
        {3}
        """.format(_pattern_url.pattern, _pattern_word.pattern,
                   _pattern_date.pattern, _pattern_numeric.pattern),
        flags=re.UNICODE|re.VERBOSE|re.IGNORECASE)


    @classmethod
    def _normalize_whitespace(cls, s):
        """Normalize two or more whitespaces to exactly one whitespace."""
        return cls._pattern_whitespace.sub(' ', s)

    @classmethod
    def tokenize_dates(cls, s):
        """Tokenize a text and return date expressions."""
        return (
            cls._normalize_whitespace(token.group())
            for token in cls._pattern_date.finditer(s)
        )

    @classmethod
    def tokenize_numerics(cls, s):
        """Tokenize a text and return numerics."""
        return (token.group() for token in cls._pattern_numeric.finditer(s))

    @classmethod
    def tokenize_words(cls, s):
        """Tokenize a text and return words."""
        return (token.group() for token in cls._pattern_word.finditer(s))

    @classmethod
    def tokenize_entire_text(cls, s):
        """Use url, word, date and numeric patterns to tokenize an input text."""
        return (
            cls._normalize_whitespace(token.group())
            for token in cls._pattern_entire_text.finditer(s)
        )
