# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command implementing
the 1-DNF technique for labeling webpages as reliable negatives.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from __future__ import division
from itertools import chain
import numpy as np

from django.core.management.base import CommandError

from apps.auxiliary_data_app.feature_extraction import ExtendedCountVectorizer
from apps.auxiliary_data_app.tokenizers import ClassifierFeatureTokenizer
from apps.auxiliary_data_app.models import StopWord
from apps.crawled_webpages_app.models import CrawledWebpage
from apps.crawled_webpages_app.utils import CrawledWebpageUtility
from apps.venyoo_events_app.models import Event
from apps.venyoo_events_app.utils import VenyooDocumentUtility
from _custom_basecommand import CustomBaseCommand


class Command(CustomBaseCommand):

    """
    This Django custom management command labels webpages as
    reliable negatives according to the 1-DNF method.

    It first builds a positive feature set PF containing words that occur in
    the positive set P more frequently than in the unlabeled set U (lines 1–5).
    Line 1 collects all the words in U ∪ P to obtain a vocabulary V.
    Lines 6–9 try to identify reliable negative documents from U.
    A document in U that does not contain any feature in PF is regarded as
    a reliable negative document.

    Among others, this technique is used in

          Yu, H., J. Han, and K. Chang.
          PEBL: positive example based learning for Web page
          classification using SVM. In: Proceedings of ACM SIGKDD
          International Conference on Knowledge Discovery and
          Data Mining (KDD- 2002), 2002.

    Algorithm 1-DNF(P, U)
    1.    Assume the word feature set be V = { w_1, ..., w_n }, w_i ∈ U ∪ P;
    2.    Let positive feature set PF <- ∅;
    3.    for each w_i ∈ V do
    4.        if ( freq(w_i, P) / |P| > freq(w_i, U) / |U| ) then
    5.            PF <- PF ∪ {w_i};
    6.    RN <- U;
    7.    for each document d ∈ U do
    8.        if ∃ w_j freq(w_j, d) > 0 and w_j ∈ PF then
    9.            RN <- RN - {d};

    """

    def handle(self, *args, **options):
        """"""
        if args:
            raise CommandError('this command does not support any non-keyword arguments')

        tokenizer = options.get('tokenizer')
        is_stemming = options.get('stemming')
        positive_data_source = options.get('positive_data_source')
        unlabeled_data_source = options.get('unlabeled_data_source')
        rn_domains = options.get('rn_domains').split(',')
        positive_set_extension = options.get('positive_set_extension')
        is_hand_labeled_data = options.get('hand_labeled_data')

        ngram_size = [int(num) for num in options.get('ngram_size').split(',')]
        if len(ngram_size) == 1:
            ngram_size.append(ngram_size[0])

        # If positive and unlabeled data source are not provided,
        # the user has to choose them on the command line
        if not positive_data_source and not unlabeled_data_source:
            positive_data_source, unlabeled_data_source = CustomBaseCommand.get_user_input()

        positive_vectorizer = ExtendedCountVectorizer(
            lowercase=True,
            min_df=1,
            ngram_range=ngram_size,
            stemming=is_stemming,
            strip_accents='unicode',
            stop_words=[stop_word.stop_word for stop_word in StopWord.objects.all()],
            tokenizer=CustomBaseCommand.get_tokenizer(tokenizer))

        unlabeled_vectorizer = ExtendedCountVectorizer(
            lowercase=True,
            min_df=1,
            ngram_range=ngram_size,
            stemming=is_stemming,
            strip_accents='unicode',
            stop_words=[stop_word.stop_word for stop_word in StopWord.objects.all()],
            tokenizer=CustomBaseCommand.get_tokenizer(tokenizer))

        # Determine which data to use for the calculations
        positive_data = VenyooDocumentUtility.webpage_generator(data_source=positive_data_source)
        if positive_set_extension not in (None, 'None', '-'):
            positive_data = chain(
                positive_data,
                CrawledWebpageUtility.webpage_generator(
                    data_source=unlabeled_data_source,
                    filter_positives=positive_set_extension,
                    exclude_hand_labeled_pages=is_hand_labeled_data)[0])

        unlabeled_data, unlabeled_data_ids = CrawledWebpageUtility.webpage_generator(
            data_source=unlabeled_data_source,
            exclude_positives=positive_set_extension,
            filter_domains=rn_domains,
            exclude_hand_labeled_pages=is_hand_labeled_data)[:2]

        print 'Create matrix of token counts per document for positive data...'
        X_positive = positive_vectorizer.fit_transform(positive_data).tocsr()
        print 'X_positive:', repr(X_positive), '\n'

        print 'Create matrix of token counts per document for unlabeled data...'
        X_unlabeled = unlabeled_vectorizer.fit_transform(unlabeled_data).tocsr()
        print 'X_unlabeled:', repr(X_unlabeled), '\n'

        print 'Sum up token counts over all documents...'
        X_positive_feature_counts = X_positive.sum(axis=0)
        X_unlabeled_feature_counts = X_unlabeled.sum(axis=0)

        print 'Sum up token counts over entire data set...'
        positive_feature_total_count = X_positive_feature_counts.sum(axis=1).tolist()[0][0]
        unlabeled_feature_total_count = X_unlabeled_feature_counts.sum(axis=1).tolist()[0][0]

        # The creation of the total feature set represents line 1 of the algorithm.
        print 'Create union of positive and unlabeled feature set...'
        total_feature_set = set(
            positive_vectorizer.get_feature_names()).union(
            set(unlabeled_vectorizer.get_feature_names()))

        # The following code represents lines 3 to 5 of the algorithm.
        print 'Determine features for positive feature set...'
        positive_feature_set = set()
        for feature in total_feature_set:
            positive_column_index = positive_vectorizer.vocabulary_.get(feature)
            unlabeled_column_index = unlabeled_vectorizer.vocabulary_.get(feature)

            if positive_column_index and unlabeled_column_index:

                positive_feature_count = X_positive_feature_counts[0, positive_column_index]
                unlabeled_feature_count = X_unlabeled_feature_counts[0, unlabeled_column_index]

                if ((positive_feature_count / positive_feature_total_count) >
                    (unlabeled_feature_count / unlabeled_feature_total_count)):

                    positive_feature_set.add(feature)

            elif positive_column_index and not unlabeled_column_index:

                positive_feature_set.add(feature)

        print 'Check unlabeled data for features in positive feature set...'
        reliable_negative_ids = set()
        unlabeled_feature_names = np.array(unlabeled_vectorizer.get_feature_names())

        # Reset all negative webpages back to 'Unlabeled'
        CrawledWebpage.objects.filter(
            is_1dnf_reliable_negative='N').update(is_1dnf_reliable_negative='-')

        # The following code represents lines 6 to 9 of the algorithm.
        for webpage in X_unlabeled:
            current_id = unlabeled_data_ids.next()
            contains_positive_features = False

            for feature in unlabeled_feature_names[webpage.nonzero()[1]]:
                if feature in positive_feature_set:
                    contains_positive_features = True
                    break

            if not contains_positive_features:
                reliable_negative_ids.add(current_id)

        affected_pages = CrawledWebpage.objects.filter(
            id__in=reliable_negative_ids).update(is_1dnf_reliable_negative='N')

        print 'Done! Annotation of unlabeled data successful.'
        print affected_pages, 'pages have been annotated as reliable negatives.'
