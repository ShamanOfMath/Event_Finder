# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command implementing
the Spy technique for labeling webpages as reliable negatives.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from __future__ import division
from itertools import chain
import numpy as np
import random

from django.core.management.base import CommandError
from sklearn.naive_bayes import MultinomialNB

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
    reliable negatives according to the Spy technique.

    This technique works by sending some “spy” documents from the
    positive set P to the unlabeled set U.

    The algorithm has three sub-steps:

    Sub-step 1
    //////////
    It randomly samples a set S of positive documents from P and put them
    in U (lines 2 and 3). The documents in S act as “spy” documents from the
    positive set to the unlabeled set U. Since the spies behave similarly to
    the unknown positive documents in U, they allow the algorithm to infer
    the behavior of the unknown positive documents in U.

    Sub-step 2
    //////////
    It runs the naive Bayesian (NB) algorithm using the set P - S as positive
    and the set U ∪ S as negative (lines 3–7). The NB classifier is then
    applied to classify each document d in U ∪ S (or Us), i.e., to assign it
    a probabilistic class label Pr(1|d), where 1 represents the positive class.

    Sub-step 3
    //////////
    It uses the probabilistic labels of the spies to decide which documents are
    most likely to be negative. A threshold t is employed to make the decision.
    Those documents in U with lower probabilities Pr(1|d) than t are the most
    likely negative documents, denoted by RN (lines 10–12).

    Among others, this technique is used in

          Liu, B., W. Lee, P. Yu, and X. Li.
          Partially supervised classification of text documents.
          In: Proceedings of International Conference on Machine Learning
          (ICML-2002), 2002.

    Algorithm Spy(P, U)
    1.    RN <- ∅;
    2.    S <- Sample(P, s%);
    3.    Us <- U ∪ S;
    4.    Ps <- P - S;
    5.    Assign each document in Ps the class label 1;
    6.    Assign each document in Us the class label -1;
    7.    NB(Us, Ps);    # This produces a NB classifier.
    8.    Classify each document in Us using the NB classifier;
    9.    Determine a probability threshold t using S;
    10.   for each document d ∈ Us do
    11.       if its probability Pr(1|d) < t then
    12.           RN <- RN ∪ {d};

    """

    def handle(self, *args, **options):
        """"""
        if args:
            raise CommandError(
                'this command does not support any non-keyword arguments')

        tokenizer = options.get('tokenizer')
        sampling_ratio = options.get('sampling_ratio')
        is_stemming = options.get('stemming')
        noise_level = options.get('noise_level')
        iterations = options.get('iterations')
        positive_data_source = options.get('positive_data_source')
        unlabeled_data_source = options.get('unlabeled_data_source')
        rn_domains = options.get('rn_domains').split(',')
        positive_set_extension = options.get('positive_set_extension')
        is_hand_labeled_data = options.get('hand_labeled_data')

        print 'UNLABELED DATA SOURCE:', unlabeled_data_source

        ngram_size = [int(num) for num in options.get('ngram_size').split(',')]
        if len(ngram_size) == 1:
            ngram_size.append(ngram_size[0])

        if not 0.0 <= sampling_ratio <= 1.0:
            raise CommandError('sampling ratio must be in range (0.0, 1.0)')

        if not 0.0 <= noise_level <= 1.0:
            raise CommandError('noise level must be in range (0.0, 1.0)')

        if iterations < 1:
            raise CommandError('number of iterations must be greater than or equal to 1')

        # If positive and unlabeled data source are not provided,
        # the user has to choose them on the command line
        if not positive_data_source and not unlabeled_data_source:
            positive_data_source, unlabeled_data_source = CustomBaseCommand.get_user_input()

        vectorizer = ExtendedCountVectorizer(
            lowercase=True,
            min_df=1,
            ngram_range=ngram_size,
            stemming=is_stemming,
            strip_accents='unicode',
            stop_words=[stop_word.stop_word for stop_word in StopWord.objects.all()],
            tokenizer=CustomBaseCommand.get_tokenizer(tokenizer))

        number_positive_docs = Event.objects.count()
        if positive_set_extension == 'emnb':
            number_positive_docs += CrawledWebpage.objects.filter(em_nb_tagging='P').count()
        elif positive_set_extension == 'svm':
            number_positive_docs += CrawledWebpage.objects.filter(svm_tagging='P').count()
        elif positive_set_extension == 'by_hand':
            number_positive_docs += CrawledWebpage.objects.filter(is_eventpage='Y').count()

        number_unlabeled_docs = CrawledWebpageUtility.webpage_generator(
            data_source=unlabeled_data_source,
            exclude_positives=positive_set_extension,
            filter_domains=rn_domains,
            exclude_hand_labeled_pages=is_hand_labeled_data)[2]

        final_reliable_negative_ids = set()

        # Process the part below for the number of specified iterations
        current_iteration = 1
        total_iterations = str(iterations)
        while iterations:

            print 'Iteration', current_iteration, '(out of ' + total_iterations + ')'

            print 'Determine random samples from positive docs...'
            event_ids = random.sample(
                Event.objects.values_list('id', flat=True),
                int(sampling_ratio * number_positive_docs))

            print 'Number of samples:', len(event_ids)
            number_positive_without_spy_docs = number_positive_docs - len(event_ids)
            number_unlabeled_and_spy_docs = number_unlabeled_docs + len(event_ids)

            print 'Create X_train matrix of token counts for training...'
            unlabeled_data, unlabeled_data_ids = CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                exclude_positives=positive_set_extension,
                filter_domains=rn_domains,
                exclude_hand_labeled_pages=is_hand_labeled_data)[:2]

            unlabeled_and_spy_data = chain(
                unlabeled_data, VenyooDocumentUtility.webpage_generator(
                    data_source=positive_data_source,
                    ids_to_filter=event_ids,
                    return_ids=False))

            unlabeled_and_spy_data_ids = chain(
                unlabeled_data_ids, VenyooDocumentUtility.webpage_generator(
                    data_source=positive_data_source,
                    ids_to_filter=event_ids,
                    return_ids=True))

            X_train = vectorizer.fit_transform(
                chain(
                    VenyooDocumentUtility.webpage_generator(
                        data_source=positive_data_source,
                        ids_to_exclude=event_ids),
                    unlabeled_and_spy_data))
            print 'X_train: ', repr(X_train), '\n'

            print 'Create y_train vector of target values (=classes)...'
            y_train = np.append(
                np.array(number_positive_without_spy_docs * [1]),
                np.array(number_unlabeled_and_spy_docs * [-1]))
            print 'y_train:', y_train.shape, '\n'

            print 'Create X_test matrix of token counts for testing...'
            unlabeled_data = CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                exclude_positives=positive_set_extension,
                filter_domains=rn_domains,
                exclude_hand_labeled_pages=is_hand_labeled_data)[0]

            unlabeled_and_spy_data = chain(
                unlabeled_data, VenyooDocumentUtility.webpage_generator(
                    data_source=positive_data_source,
                    ids_to_filter=event_ids,
                    return_ids=False))

            X_test = vectorizer.transform(unlabeled_and_spy_data)
            print 'X_test:', repr(X_test), '\n'

            print 'Create X_spy matrix to determine threshold t...'
            X_spy = X_test.asformat('csr')[-len(event_ids):]

            print 'Train Multinomial NB classifier...'
            classifier = MultinomialNB(alpha=0.1)
            classifier.fit(X=X_train, y=y_train)

            print 'Create log_probabilities for X_test...'
            X_test_log_proba = classifier.predict_log_proba(X_test)

            print 'Create log_probabilities for X_spy...'
            X_spy_log_proba = classifier.predict_log_proba(X_spy)

            print 'Determine probability threshold t...'

            if 0 < noise_level < 1:

                # Determine number of spy documents to ignore
                num_docs_to_ignore = int(noise_level * X_spy.shape[0])
                print num_docs_to_ignore, \
                    'spy documents were labeled as noise and will be ignored...'

                # Determine number of spy documents to consider for further calculation
                num_docs_to_consider = X_spy.shape[0] - num_docs_to_ignore

                # Create the fraction of documents and determine the threshold from it
                threshold = np.sort(X_spy_log_proba.T[1])[::-1][:num_docs_to_consider].min()

            else:
                threshold = X_spy_log_proba.T[1].min()
            print 'Threshold t =', threshold, '\n'

            print 'Determine reliable negatives...'
            reliable_negative_ids = set()

            for doc in X_test_log_proba:
                current_id = unlabeled_and_spy_data_ids.next()
                if doc[1] < threshold:
                    reliable_negative_ids.add(current_id)

            if not final_reliable_negative_ids:
                final_reliable_negative_ids = final_reliable_negative_ids | reliable_negative_ids
            else:
                final_reliable_negative_ids = final_reliable_negative_ids & reliable_negative_ids

            current_iteration += 1
            iterations -= 1

        # Reset all negative webpages back to 'Unlabeled'
        CrawledWebpage.objects.filter(
            is_spy_reliable_negative='N').update(
            is_spy_reliable_negative='-')

        print 'Label reliable negatives in database...'
        affected_pages = CrawledWebpage.objects.filter(
            id__in=final_reliable_negative_ids).update(
            is_spy_reliable_negative='N')

        print 'Done! Annotation of unlabeled data successful.'
        print affected_pages, 'documents have been annotated as reliable negatives.'
