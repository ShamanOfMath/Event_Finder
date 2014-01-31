# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command implementing
the Naive Bayes technique for labeling webpages as reliable negatives.

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
    reliable negatives according to the Naive Bayes technique.

    It simply uses a multinomial naïve Bayesian classifier to
    identify a set of reliable negative documents RN from the unlabeled set U.
    This method may also be run multiple times. Each time we randomly remove
    a few documents from P to obtain a different set of reliable negative
    documents, denoted by RN_i. The final set of reliable negative documents
    RN is the intersection of all RN_i.

    Among others, this technique is used in

        Liu, B., Y. Dai, X. Li, W. Lee, and P. Yu.
        Building text classifiers using positive and unlabeled examples.
        In: Proceedings of IEEE International Conference on Data Mining
        (ICDM-2003), 2003.

    Algorithm Naive Bayes(P, U)
    1.    Assign each document in P the class label 1;
    2.    Assign each document in U the class label -1;
    3.    Build a NB classifier using P and U;
    4.    Use the classifier to classify U. Those documents in U that are
          classified as negative form the reliable negative set RN.

    """

    def handle(self, *args, **options):
        """"""
        if args:
            raise CommandError('this command does not support any non-keyword arguments')

        tokenizer = options.get('tokenizer')
        sampling_ratio = options.get('sampling_ratio')
        is_stemming = options.get('stemming')
        iterations = options.get('iterations')
        positive_data_source = options.get('positive_data_source')
        unlabeled_data_source = options.get('unlabeled_data_source')
        rn_domains = options.get('rn_domains').split(',')
        positive_set_extension = options.get('positive_set_extension')
        is_hand_labeled_data = options.get('hand_labeled_data')

        ngram_size = [int(num) for num in options.get('ngram_size').split(',')]
        if len(ngram_size) == 1:
            ngram_size.append(ngram_size[0])

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

        final_reliable_negative_ids = set()

        # Process the part below for the number of specified iterations
        current_iteration = 1
        total_iterations = str(iterations)
        while iterations:

            print '\nIteration', current_iteration, '(out of ' + total_iterations + ')'

            print 'Determine random samples from positive docs...'
            event_ids = random.sample(
                Event.objects.values_list('id', flat=True),
                int(sampling_ratio * number_positive_docs))

            print 'Number of positive samples to ignore:', len(event_ids)
            number_positive_without_sample_docs = number_positive_docs - len(event_ids)

            positive_data = VenyooDocumentUtility.webpage_generator(
                data_source=positive_data_source, ids_to_exclude=event_ids)

            if positive_set_extension not in (None, 'None', '-'):
                positive_data = chain(
                    positive_data,
                    CrawledWebpageUtility.webpage_generator(
                        data_source=unlabeled_data_source,
                        filter_positives=positive_set_extension,
                        exclude_hand_labeled_pages=is_hand_labeled_data)[0])

            unlabeled_data, unlabeled_data_ids, number_unlabeled_docs = CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                exclude_positives=positive_set_extension,
                filter_domains=rn_domains,
                exclude_hand_labeled_pages=is_hand_labeled_data)

            print 'Create X_train matrix of token counts for training...'
            X_train = vectorizer.fit_transform(
                chain(positive_data, unlabeled_data)).tocsr()
            print 'X_train: ', repr(X_train), '\n'

            print 'Create y_train vector of target values (=classes)...'
            y_train = np.concatenate([
                np.array(number_positive_without_sample_docs * [0]),
                np.array(number_unlabeled_docs * [-1])])
            print 'y_train:', y_train.shape, '\n'

            print 'Create X_test matrix of token counts for testing...'
            unlabeled_data = CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                exclude_positives=positive_set_extension,
                filter_domains=rn_domains,
                exclude_hand_labeled_pages=is_hand_labeled_data)[0]

            X_test = vectorizer.transform(unlabeled_data)
            print 'X_test:', repr(X_test), '\n'

            print 'Train Multinomial NB classifier...'
            classifier = MultinomialNB(alpha=0.1)
            classifier.fit(X=X_train, y=y_train)

            print 'Determine reliable negatives...'
            predictions = classifier.predict(X=X_test)
            reliable_negative_ids = set()
            for prediction in predictions:
                current_id = unlabeled_data_ids.next()
                if prediction == -1:
                    reliable_negative_ids.add(current_id)

            if not final_reliable_negative_ids:
                final_reliable_negative_ids = final_reliable_negative_ids | reliable_negative_ids
            else:
                final_reliable_negative_ids = final_reliable_negative_ids & reliable_negative_ids

            current_iteration += 1
            iterations -= 1

        # Reset all negative webpages back to 'Unlabeled'
        CrawledWebpage.objects.filter(
            is_nb_reliable_negative='N').update(
            is_nb_reliable_negative='-')

        print 'Label reliable negatives in database...'
        affected_pages = CrawledWebpage.objects.filter(
            id__in=final_reliable_negative_ids).update(
            is_nb_reliable_negative='N')

        print 'Done! Annotation of unlabeled data successful.'
        print affected_pages, 'documents have been annotated as reliable negatives.'
