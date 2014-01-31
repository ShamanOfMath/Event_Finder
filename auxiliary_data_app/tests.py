# -*- coding: utf-8 -*-

"""
This module contains test cases for the auxiliary_data_app.

For more information on unit tests in general, see:
    http://docs.python.org/2.7/library/unittest.html

For more information on testing Django applications, see:
    https://docs.djangoproject.com/en/1.5/topics/testing/

"""


import inspect

from django.test import TestCase

from apps.auxiliary_data_app.tokenizers import ClassifierFeatureTokenizer


class TokenizerTest(TestCase):

    """
    This class provides test cases for the tokenizers module.
    Currently, it contains the ClassifierFeatureTokenizer only.

    """

    def setUp(self):
        """Specify variables to be used by more than one test method."""
        self.text = u"""
        Diese Veranstaltung findet am 08. Mai 2013 um 20:15 Uhr statt.
        Der Eintrittspreis beträgt 10,50 Euro.
        Für mehr Informationen besuchen Sie uns im Internet unter
        http://www.venyoo.de
        """

    def test_whitespace_normalization(self):
        """Check whether whitespace normalization in strings works correctly."""
        self.assertEqual(
            ClassifierFeatureTokenizer._normalize_whitespace(
                '   abc  def 12345   8!%&          g hij    klm   '
            ),
            ' abc def 12345 8!%& g hij klm '
        )

    def test_dates_tokenizer(self):
        """Check whether the tokenization of date expressions works correctly."""
        date1 = '8.5.2013'
        date2 = '8.5.2013 20'
        date3 = '8.5.2013 20:15'
        date4 = '8.5.2013 20:15:48'
        date5 = '8.5.2013 20 Uhr'
        date6 = '8.5.2013 20 Uhr 15'
        date7 = '8.5.2013 20:15 Uhr'
        date8 = '8.5.2013 20:15:48 Uhr'

        date9 = '20 8.5.2013'
        date10 = '20:15 8.5.2013'
        date11 = '20:15:48 8.5.2013'
        date12 = '20 Uhr 8.5.2013'
        date13 = '20 Uhr 15 8.5.2013'
        date14 = '20:15 Uhr 8.5.2013'
        date15 = '20:15:48 Uhr 8.5.2013'

        date16 = '8. Mai 2013'
        date17 = '8. Mai 2013 20'
        date18 = '8. Mai 2013 20:15'
        date19 = '8. Mai 2013 20:15:48'
        date20 = '8. Mai 2013 20 Uhr'
        date21 = '8. Mai 2013 20 Uhr 15'
        date22 = '8. Mai 2013 20:15 Uhr'
        date23 = '8. Mai 2013 20:15:48 Uhr'

        date24 = '20 8. Mai 2013'
        date25 = '20:15 8. Mai 2013'
        date26 = '20:15:48 8. Mai 2013'
        date27 = '20 Uhr 8. Mai 2013'
        date28 = '20 Uhr 15 8. Mai 2013'
        date29 = '20:15 Uhr 8. Mai 2013'
        date30 = '20:15:48 Uhr 8. Mai 2013'

        date31 = '20:15'
        date32 = '20:15:48'
        date33 = '20 Uhr'
        date34 = '20 Uhr 15'
        date35 = '20:15 Uhr'
        date36 = '20:15:48 Uhr'

        date1_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date1)
        date2_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date2)
        date3_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date3)
        date4_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date4)
        date5_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date5)
        date6_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date6)
        date7_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date7)
        date8_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date8)

        date9_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date9)
        date10_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date10)
        date11_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date11)
        date12_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date12)
        date13_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date13)
        date14_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date14)
        date15_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date15)

        date16_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date16)
        date17_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date17)
        date18_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date18)
        date19_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date19)
        date20_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date20)
        date21_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date21)
        date22_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date22)

        date23_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date23)
        date24_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date24)
        date25_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date25)
        date26_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date26)
        date27_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date27)
        date28_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date28)
        date29_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date29)
        date30_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date30)

        date31_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date31)
        date32_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date32)
        date33_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date33)
        date34_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date34)
        date35_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date35)
        date36_tokenized = ClassifierFeatureTokenizer.tokenize_dates(date36)

        # Assert that tokenization result is a generator
        self.assertEqual(inspect.isgenerator(date1_tokenized), True)

        # Assert that tokenization results are correct
        self.assertEqual(list(date1_tokenized), ['8.5.2013'])
        self.assertEqual(list(date2_tokenized), ['8.5.2013'])
        self.assertEqual(list(date3_tokenized), ['8.5.2013', '20:15'])
        self.assertEqual(list(date4_tokenized), ['8.5.2013', '20:15:48'])
        self.assertEqual(list(date5_tokenized), ['8.5.2013', '20 Uhr'])
        self.assertEqual(list(date6_tokenized), ['8.5.2013', '20 Uhr 15'])
        self.assertEqual(list(date7_tokenized), ['8.5.2013', '20:15 Uhr'])
        self.assertEqual(list(date8_tokenized), ['8.5.2013', '20:15:48 Uhr'])

        self.assertEqual(list(date9_tokenized), ['8.5.2013'])
        self.assertEqual(list(date10_tokenized), ['20:15', '8.5.2013'])
        self.assertEqual(list(date11_tokenized), ['20:15:48', '8.5.2013'])
        self.assertEqual(list(date12_tokenized), ['20 Uhr', '8.5.2013'])
        self.assertEqual(list(date13_tokenized), ['20 Uhr 15', '8.5.2013'])
        self.assertEqual(list(date14_tokenized), ['20:15 Uhr', '8.5.2013'])
        self.assertEqual(list(date15_tokenized), ['20:15:48 Uhr', '8.5.2013'])

        self.assertEqual(list(date16_tokenized), ['8. Mai 2013'])
        self.assertEqual(list(date17_tokenized), ['8. Mai 2013'])
        self.assertEqual(list(date18_tokenized), ['8. Mai 2013', '20:15'])
        self.assertEqual(list(date19_tokenized), ['8. Mai 2013', '20:15:48'])
        self.assertEqual(list(date20_tokenized), ['8. Mai 2013', '20 Uhr'])
        self.assertEqual(list(date21_tokenized), ['8. Mai 2013', '20 Uhr 15'])
        self.assertEqual(list(date22_tokenized), ['8. Mai 2013', '20:15 Uhr'])
        self.assertEqual(list(date23_tokenized), ['8. Mai 2013', '20:15:48 Uhr'])

        self.assertEqual(list(date24_tokenized), ['8. Mai 2013'])
        self.assertEqual(list(date25_tokenized), ['20:15', '8. Mai 2013'])
        self.assertEqual(list(date26_tokenized), ['20:15:48', '8. Mai 2013'])
        self.assertEqual(list(date27_tokenized), ['20 Uhr', '8. Mai 2013'])
        self.assertEqual(list(date28_tokenized), ['20 Uhr 15', '8. Mai 2013'])
        self.assertEqual(list(date29_tokenized), ['20:15 Uhr', '8. Mai 2013'])
        self.assertEqual(list(date30_tokenized), ['20:15:48 Uhr', '8. Mai 2013'])

        self.assertEqual(list(date31_tokenized), ['20:15'])
        self.assertEqual(list(date32_tokenized), ['20:15:48'])
        self.assertEqual(list(date33_tokenized), ['20 Uhr'])
        self.assertEqual(list(date34_tokenized), ['20 Uhr 15'])
        self.assertEqual(list(date35_tokenized), ['20:15 Uhr'])
        self.assertEqual(list(date36_tokenized), ['20:15:48 Uhr'])

    def test_numerics_tokenizer(self):
        """Check whether the tokenization of numerics works correctly."""
        numeric1 = '58473'
        numeric2 = '58473,28573'
        numeric3 = '58.473,28573'
        numeric4 = '58.473'
        numeric5 = '58.473285'

        numeric6 = '123456789321'
        numeric7 = '123456789321,4'
        numeric8 = '123.456.789.321,4'
        numeric9 = '123.456.789.321'
        numeric10 = '123456.789.321,4'

        numeric1_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric1)
        numeric2_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric2)
        numeric3_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric3)
        numeric4_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric4)
        numeric5_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric5)

        numeric6_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric6)
        numeric7_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric7)
        numeric8_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric8)
        numeric9_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric9)
        numeric10_tokenized = ClassifierFeatureTokenizer.tokenize_numerics(numeric10)

        # Assert that tokenization result is a generator
        self.assertEqual(inspect.isgenerator(numeric1_tokenized), True)

        # Assert that tokenization results are correct
        self.assertEqual(list(numeric1_tokenized), ['58473'])
        self.assertEqual(list(numeric2_tokenized), ['58473,28573'])
        self.assertEqual(list(numeric3_tokenized), ['58.473,28573'])
        self.assertEqual(list(numeric4_tokenized), ['58.473'])
        self.assertEqual(list(numeric5_tokenized), ['58', '473285'])

        self.assertEqual(list(numeric6_tokenized), ['123456789321'])
        self.assertEqual(list(numeric7_tokenized), ['123456789321,4'])
        self.assertEqual(list(numeric8_tokenized), ['123.456.789.321,4'])
        self.assertEqual(list(numeric9_tokenized), ['123.456.789.321'])
        self.assertEqual(list(numeric10_tokenized), ['123456', '789.321,4'])

    def test_words_tokenizer(self):
        """Check whether the tokenization of words works correctly."""
        text_tokenized = ClassifierFeatureTokenizer.tokenize_words(self.text)

        # Assert that tokenization result is a generator
        self.assertEqual(inspect.isgenerator(text_tokenized), True)

        # Assert that tokenization results are correct
        self.assertEqual(
            list(text_tokenized),
            [u'Diese', u'Veranstaltung', u'findet', u'am', u'Mai', u'um',
             u'Uhr', u'statt', u'Der', u'Eintrittspreis', u'beträgt', u'Euro',
             u'Für', u'mehr', u'Informationen', u'besuchen', u'Sie', u'uns',
             u'im', u'Internet', u'unter', u'http', u'www', u'venyoo', u'de']
        )

    def test_entire_text_tokenizer(self):
        """
        Check whether the tokenization of all feature types works correctly.
        """
        text_tokenized = ClassifierFeatureTokenizer.tokenize_entire_text(self.text)

        # Assert that tokenization result is a generator
        self.assertEqual(inspect.isgenerator(text_tokenized), True)

        # Assert that tokenization results are correct
        self.assertEqual(
            list(text_tokenized),
            [u'Diese', u'Veranstaltung', u'findet', u'am', u'08. Mai 2013',
             u'um', u'20:15 Uhr', u'statt', u'Der', u'Eintrittspreis',
             u'beträgt', u'10,50', u'Euro', u'Für', u'mehr', u'Informationen',
             u'besuchen', u'Sie', u'uns', u'im', u'Internet', u'unter',
             u'http://www.venyoo.de']
        )
