# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command implementing
the Iterative SVM algorithm for a binary classification of webpages.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from __future__ import division
import csv
from datetime import datetime
from itertools import chain
from optparse import make_option
import os

import matplotlib.pyplot as plt
import numpy as np
from django.conf import settings
from django.core.management.base import CommandError
from sklearn.metrics import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from apps.auxiliary_data_app.feature_extraction import ExtendedCountVectorizer
from apps.auxiliary_data_app.tokenizers import ClassifierFeatureTokenizer
from apps.auxiliary_data_app.models import StopWord
from apps.crawled_webpages_app.models import CrawledWebpage
from apps.crawled_webpages_app.utils import CrawledWebpageUtility
from apps.venyoo_events_app.models import Event
from apps.venyoo_events_app.utils import VenyooDocumentUtility
from _custom_basecommand import ClassifiersCustomBaseCommand


class Command(ClassifiersCustomBaseCommand):

    """
    This Django custom management command labels webpages as
    either positive or negative using the Iterative SVM algorithm.

    In this method, SVM is run iteratively using P, RN and Q (U - RN).
    In each iteration, a new SVM classifier f is constructed from P and RN
    (line 4). Here RN is regarded as the set of negative examples (line 2).
    The classifier f is then applied to classify the documents in Q
    (line 5). The set W of documents in Q that are classified as negative
    (line 6) is removed from Q (line 8) and added to RN (line 9).
    The iteration stops when no document in Q is classified as negative,
    i.e., W = ∅ (line 7). The final classifier is the result.

    Among others, this algorithm is used in

          Li, X. and B. Liu.
          Learning to classify texts using positive and unlabeled data.
          In: Proceedings of International Joint Conference on Artificial
          Intelligence (IJCAI-2003), 2003.

    Algorithm I-SVM(P, RN, Q)
    1.    Every document in P is assigned class label 1;
    2.    Every document in RN is assigned class label -1;
    3.    loop
    4.        Use P and RN to train a SVM classifier f;
    5.        Classify Q using f;
    6.        Let W be the set of documents in Q which are classified as negative;
    7.        if W = ∅ then exit-loop;  // convergence
    8.        else
    9.            Q <- Q - W;
    10.           RN <- RN ∪ W;

    """

    option_list = ClassifiersCustomBaseCommand.option_list + (
        make_option('--iterative-mode',
            action='store_true',
            dest='iterative_mode',
            default=False,
            help='whether to apply the SVM classifier iteratively '
                 'until convergence [default: %default]'
        ),
    )

    def handle(self, *args, **options):
        """"""
        if args:
            raise CommandError(
                'this command does not support any non-keyword arguments')

        save_statistics = options.get('save_statistics')
        tokenizer = options.get('tokenizer')
        rn_method = options.get('rn_method')
        apply_iterative_mode = options.get('iterative_mode')
        is_stemming = options.get('stemming')
        positive_data_source = options.get('positive_data_source')
        unlabeled_data_source = options.get('unlabeled_data_source')
        rn_domains = options.get('rn_domains').split(',')
        clf_domains = options.get('clf_domains').split(',')
        positive_set_extension = options.get('positive_set_extension')
        is_hand_labeled_data = options.get('hand_labeled_data')

        ngram_size = [int(num) for num in options.get('ngram_size').split(',')]
        if len(ngram_size) == 1:
            ngram_size.append(ngram_size[0])

        # If positive and unlabeled data source are not provided,
        # the user has to choose them on the command line
        if not positive_data_source and not unlabeled_data_source:
            positive_data_source, unlabeled_data_source = ClassifiersCustomBaseCommand.get_user_input()

        vectorizer = ExtendedCountVectorizer(
            lowercase=True,
            min_df=1,
            ngram_range=ngram_size,
            stemming=is_stemming,
            strip_accents='unicode',
            stop_words=[stop_word.stop_word for stop_word in StopWord.objects.all()],
            tokenizer=ClassifiersCustomBaseCommand.get_tokenizer(tokenizer))

        scaler = StandardScaler(with_mean=False)

        number_positive_docs = Event.objects.count()
        if positive_set_extension == 'emnb':
            number_positive_docs += CrawledWebpage.objects.filter(em_nb_tagging='P').count()
        elif positive_set_extension == 'svm':
            number_positive_docs += CrawledWebpage.objects.filter(svm_tagging='P').count()
        elif positive_set_extension == 'by_hand':
            number_positive_docs += CrawledWebpage.objects.filter(is_eventpage='Y').count()

        reliable_negative_data,\
        reliable_negative_data_ids,\
        number_reliable_negative_docs = CrawledWebpageUtility.webpage_generator(
            data_source=unlabeled_data_source,
            filter_rn=rn_method)

        unlabeled_data,\
        unlabeled_data_ids,\
        number_unlabeled_docs = CrawledWebpageUtility.webpage_generator(
            data_source=unlabeled_data_source,
            #exclude_positives=positive_set_extension,
            exclude_rn=rn_method,
            filter_domains=clf_domains,
            filter_hand_labeled_pages=is_hand_labeled_data)

        current_iteration = 1

        while True:

            print 'Iteration', current_iteration, 'of classification process'
            print '=' * 70
            current_iteration += 1

            positive_data = VenyooDocumentUtility.webpage_generator(
                data_source=positive_data_source)

            if positive_set_extension not in (None, 'None', '-'):
                positive_data = chain(
                    positive_data,
                    CrawledWebpageUtility.webpage_generator(
                        data_source=unlabeled_data_source,
                        filter_positives=positive_set_extension,
                        #exclude_hand_labeled_pages=is_hand_labeled_data
                    )[0])

            print 'Create X_train matrix of token counts for training...'
            X_train = vectorizer.fit_transform(
            chain(positive_data, reliable_negative_data))
            print 'X_train: ', repr(X_train), '\n'

            print 'Create y_train vector of target values (=classes)...'
            y_train = np.concatenate([
                np.array(number_positive_docs * [0]),
                np.array(number_reliable_negative_docs * [-1])])
            print 'y_train:', y_train.shape, '\n'

            print 'Create X_test matrix of token counts for testing...'
            X_test = vectorizer.transform(unlabeled_data)
            print 'X_test:', repr(X_test), '\n'

            print 'Standardize features by removing the mean and scaling to unit variance'
            X_train_scaled = scaler.fit_transform(X=X_train.astype(np.float))
            X_test_scaled = scaler.transform(X=X_test.astype(np.float))
            print 'X_train_scaled', repr(X_train_scaled), '\n'
            print 'X_test_scaled', repr(X_test_scaled), '\n'

            print 'Create linear SVM classifier...'
            linear_svm_classifier = LinearSVC()

            print 'Apply classifier...'
            linear_svm_classifier.fit(X=X_train_scaled, y=y_train)

            print 'Predict labels for unlabeled data...'
            predictions = linear_svm_classifier.predict(X=X_test_scaled)
            probabilities = linear_svm_classifier.decision_function(X=X_test_scaled)
            print 'PROBABILITIES:', len(probabilities)

            # Determine how many documents were labeled as negative
            unlabeled_data_ids = np.array(sorted(list(unlabeled_data_ids)))
            negative_ids = set(unlabeled_data_ids[(predictions == -1).nonzero()])

            # If iterative mode is disabled, exit loop after one iteration
            if not apply_iterative_mode:
                print 'Iterative mode is disabled.'
                print 'Using the first trained classifier to label unlabeled data...'
                break

            # if the set of documents classified as negative is empty,
            # exit loop
            elif not negative_ids:
                print 'No documents anymore classified as negative'
                print 'Using the last trained classifier to label unlabeled data...'
                break

            # Q <- Q - W
            unlabeled_data_ids = set(unlabeled_data_ids) - negative_ids
            # RN <- RN ∪ W
            reliable_negative_data_ids = set(reliable_negative_data_ids) | negative_ids

            # Create data for next iteration of the loop
            reliable_negative_data  = CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                filter_ids=reliable_negative_data_ids)[0]

            number_reliable_negative_docs = CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                filter_ids=reliable_negative_data_ids)[2]

            unlabeled_data = CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                filter_ids=unlabeled_data_ids)[0]

        # Reset webpages back to 'Unlabeled'
        if positive_set_extension in ('emnb', 'svm', 'by_hand'):
            CrawledWebpage.objects.filter(
                svm_tagging='N').update(svm_tagging='-')
        else:
            CrawledWebpage.objects.filter(
                svm_tagging__in=['P', 'N']).update(svm_tagging='-')

        print 'Label unlabeled webpages in database...'

        if apply_iterative_mode:
            unlabeled_data,\
            unlabeled_data_ids,\
            number_unlabeled_docs = CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                exclude_positives=positive_set_extension,
                exclude_rn=rn_method,
                filter_domains=clf_domains,
                filter_hand_labeled_pages=is_hand_labeled_data)

            print 'Create X_test matrix of token counts for testing...'
            X_test = vectorizer.transform(unlabeled_data)
            print 'X_test:', repr(X_test), '\n'

            print 'Standardize features by removing the mean and scaling to unit variance'
            X_test_scaled = scaler.transform(X=X_test.astype(np.float))

            print 'Predict labels for unlabeled data...'
            predictions = linear_svm_classifier.predict(X=X_test_scaled)
            probabilities = linear_svm_classifier.decision_function(X=X_test_scaled)

            unlabeled_data_ids = np.array(sorted(list(unlabeled_data_ids)))

        # Copy the tagging of reliable negatives to EMNB column of the database
        if rn_method == '1dnf':
            reliable_negatives = CrawledWebpage.objects.filter(is_1dnf_reliable_negative='N')
        elif rn_method == 'cr':
            reliable_negatives = CrawledWebpage.objects.filter(is_cosine_rocchio_reliable_negative='N')
        elif rn_method == 'nb':
            reliable_negatives = CrawledWebpage.objects.filter(is_nb_reliable_negative='N')
        elif rn_method == 'rocchio':
            reliable_negatives = CrawledWebpage.objects.filter(is_rocchio_reliable_negative='N')
        elif rn_method == 'spy':
            reliable_negatives = CrawledWebpage.objects.filter(is_spy_reliable_negative='N')

        reliable_negatives.update(svm_tagging='N')

        # Label the rest of unlabeled pages
        positive_ids = set(unlabeled_data_ids[(predictions == 0).nonzero()])
        negative_ids = set(unlabeled_data_ids[(predictions == -1).nonzero()])

        CrawledWebpage.objects.filter(id__in=positive_ids).update(svm_tagging='P')
        CrawledWebpage.objects.filter(id__in=negative_ids).update(svm_tagging='N')

        if is_hand_labeled_data:
            queryset = CrawledWebpage.objects.filter(is_eventpage__in=['Y', 'N'])
        else:
            queryset = CrawledWebpage.objects.filter(domain__name__in=clf_domains)

        negative_pages = queryset.filter(svm_tagging='N').count()
        positive_pages = queryset.filter(svm_tagging='P').count()

        print 'Done! Annotation of unlabeled data successful.'
        print positive_pages, 'documents have been annotated as positive.'
        print negative_pages, 'documents have been annotated as negative.'


        # ======================================================================
        # --------------------- EVALUATION STATISTICS --------------------------
        # ======================================================================

        if is_hand_labeled_data and save_statistics:

            # Get RN-technique specific options for for creating file and
            # directory names.
            sampling_ratio = options.get('sampling_ratio')
            noise_level = options.get('noise_level')
            iterations = options.get('iterations')
            kmeans_clustering = options.get('kmeans_clustering')

            # Build subdirectory name that will hold the statistics
            subdirectory = 'PDS-' + positive_data_source + '_'
            subdirectory += 'UDS-' + unlabeled_data_source + '_'
            subdirectory += 'RN-' + rn_method + '_'

            if rn_method == 'cr':
                subdirectory += 'NL-' + str(noise_level) + '_'

            elif rn_method == 'nb':
                subdirectory += 'SR-' + str(sampling_ratio) + '_'
                subdirectory += 'IT-' + str(iterations) + '_'

            elif rn_method == 'rocchio':
                subdirectory += 'KMEANS-' + str(kmeans_clustering) + '_'

            elif rn_method == 'spy':
                subdirectory += 'SR-' + str(sampling_ratio) + '_'
                subdirectory += 'IT-' + str(iterations) + '_'
                subdirectory += 'NL-' + str(noise_level) + '_'

            subdirectory += 'CLF-SVM_'
            subdirectory += 'SVMIT-' + str(apply_iterative_mode) + '_'
            subdirectory += 'PSE-' + positive_set_extension + '_'
            subdirectory += 'FEA-' + tokenizer + '_'
            subdirectory += 'NGRAM-' + '-'.join([str(num) for num in ngram_size]) + '_'
            subdirectory += 'STEM-' + str(is_stemming)

            # Create the absolute path to this subdirectory
            full_path = os.path.join(settings.ROOT_PATH, 'evaluation_results', subdirectory)

            # Check if this subdirectory exists already. If not, create it.
            # (This created the directory evaluation_results as well if not existent.)
            if not os.path.exists(full_path):
                os.makedirs(full_path)

            # Get the values_list of hand-labeled data and machine-labeled data
            queryset = CrawledWebpage.objects.filter(is_eventpage__in=['Y', 'N']).order_by('id')
            labels_by_hand = queryset.values_list('is_eventpage', flat=True)
            labels_by_machine = queryset.values_list('svm_tagging', flat=True)

            # Map labels to their integer representations
            labels_by_hand = Command._map_clf_results(labels_by_hand)
            labels_by_machine = Command._map_clf_results(labels_by_machine)

            # Get statistics for precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(
                y_true=labels_by_hand,
                probas_pred=probabilities)
            pr_auc = auc(recall, precision)

            # Get statistics for ROC curve
            fpr, tpr, roc_thresholds = roc_curve(
                y_true=labels_by_hand,
                y_score=probabilities)
            roc_auc = auc(fpr, tpr)

            # Get report for precision, recall, f-measure as a string to print
            clf_report = classification_report(
                y_true=labels_by_hand,
                y_pred=labels_by_machine)

            # Get accuracy classification score as float number
            acc_score = accuracy_score(
                y_true=labels_by_hand,
                y_pred=labels_by_machine)

            # Get average precision score as float number
            aps_score = average_precision_score(
                y_true=labels_by_hand,
                y_score=probabilities)

            # Get precision score as float number
            pre_score = precision_score(
                y_true=labels_by_hand,
                y_pred=labels_by_machine,
                average=None)

            # Get recall score as float number
            rec_score = recall_score(
                y_true=labels_by_hand,
                y_pred=labels_by_machine,
                average=None)

            # Get Matthews correlation coefficient (MCC, also known as Phi coefficient)
            mcc = matthews_corrcoef(
                y_true=labels_by_hand,
                y_pred=labels_by_machine)

            # ------------------------ Text statistics -------------------------

            print 'Writing evaluation results to file ...'
            with open(os.path.join(full_path, subdirectory + '.txt'), 'w') as fobj:

                print >> fobj, '=' * 70

                print >> fobj, 'EVALUATION STATISTICS from ' + str(datetime.today()).partition('.')[0] + '\n'

                print >> fobj, Command._format_eval_results('Positive Data Source:', positive_data_source)
                print >> fobj, Command._format_eval_results('Unlabeled Data Source:', unlabeled_data_source)

                print >> fobj, ''

                print >> fobj, Command._format_eval_results('RN Technique:', rn_method)

                if rn_method == 'cr':
                    print >> fobj, Command._format_eval_results('Noise Level:', str(noise_level))

                elif rn_method == 'nb':
                    print >> fobj, Command._format_eval_results('Sampling Ratio:', str(sampling_ratio))
                    print >> fobj, Command._format_eval_results('Iterations:', str(iterations))

                elif rn_method == 'rocchio':
                    print >> fobj, Command._format_eval_results('K-Means Clustering Applied:', str(kmeans_clustering))

                elif rn_method == 'spy':
                    print >> fobj, Command._format_eval_results('Sampling Ratio', str(sampling_ratio))
                    print >> fobj, Command._format_eval_results('Iterations', str(iterations))
                    print >> fobj, Command._format_eval_results('Noise Level', str(noise_level))

                print >> fobj, ''

                print >> fobj, Command._format_eval_results('Classifier for Final Labeling:', 'SVM')
                print >> fobj, Command._format_eval_results('Iterative Mode Applied:', str(apply_iterative_mode))
                print >> fobj, Command._format_eval_results('Positive Set Extension:', positive_set_extension)
                print >> fobj, Command._format_eval_results('Features to Consider:', tokenizer)
                print >> fobj, Command._format_eval_results('N-Gram Range:', ', '.join([str(num) for num in ngram_size]))
                print >> fobj, Command._format_eval_results('Stemming Applied:', str(is_stemming))

                print >> fobj, '=' * 70 + '\n' * 2

                rn_input_total = CrawledWebpage.objects.filter(domain__name__in=rn_domains).exclude(is_eventpage__in=['Y', 'N']).count()
                rn_count = reliable_negatives.count()
                rn_percentage = '{} ({:.2f}%)'.format(rn_count, rn_count / rn_input_total * 100)
                print >> fobj, Command._format_eval_results('Documents as Input to Determine Reliable Negatives:', str(rn_input_total))
                print >> fobj, Command._format_eval_results('Documents Annotated as Reliable Negative:', rn_percentage)

                print >> fobj, '\n'

                clf_input_total = positive_pages + negative_pages
                pos_percentage = '{} ({:.2f}%)'.format(positive_pages, positive_pages / clf_input_total * 100)
                neg_percentage = '{} ({:.2f}%)'.format(negative_pages, negative_pages / clf_input_total * 100)
                print >> fobj, Command._format_eval_results('Documents as Input for Final Labeling:', str(clf_input_total))
                print >> fobj, Command._format_eval_results('Documents Annotated as Positive:', pos_percentage)
                print >> fobj, Command._format_eval_results('Documents Annotated as Negative:', neg_percentage)

                print >> fobj, '\n'

                print >> fobj, Command._format_eval_results('Accuracy Score:', '{:.3f}'.format(acc_score))
                print >> fobj, Command._format_eval_results('Precision Score:', '{:.3f}'.format(pre_score[1]))
                print >> fobj, Command._format_eval_results('Average Precision Score:', '{:.3f}'.format(aps_score))
                print >> fobj, Command._format_eval_results('Recall Score:', '{:.3f}'.format(rec_score[1]))
                print >> fobj, Command._format_eval_results('Matthews Correlation (=Phi) Coefficient:', '{:.3f}'.format(mcc))

                print >> fobj, '\n'

                print >> fobj, Command._format_eval_results('Area under Precision-Recall Curve:', '{:.3f}'.format(pr_auc))
                print >> fobj, Command._format_eval_results('Area under ROC Curve:', '{:.3f}'.format(roc_auc))

                print >> fobj, '\n'

                print >> fobj, clf_report

            print 'Evaluation results have been written to:', subdirectory + '.txt'

            print 'Save precision and recall values to CSV file...'
            f = 'precision_recall_values.csv'
            with open(os.path.join(full_path, f), 'wb') as fobj:
                new_file = csv.writer(fobj, delimiter='\t')
                new_file.writerow(['# Precision', 'Recall'])
                for pr, rec in zip(precision, recall):
                    new_file.writerow([pr, rec])
            print 'Values saved to', f

            print 'Save false and true positive rate values to CSV file...'
            f = 'roc_values.csv'
            with open(os.path.join(full_path, f), 'wb') as fobj:
                new_file = csv.writer(fobj, delimiter='\t')
                new_file.writerow(['# True Positive Rate', 'False Positive Rate'])
                for tr, fa in zip(tpr, fpr):
                    new_file.writerow([tr, fa])
            print 'Values saved to', f

            # --------------------- Plotting Statistics ------------------------

            print 'Plotting Precision-Recall curve'

            # Plot precision against recall
            plt.plot(recall, precision, label='Precision-Recall curve', linewidth=2)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.grid(True)

            # Save the plot as a pdf
            plt.savefig(os.path.join(full_path, 'plot-pr_' + subdirectory + '.pdf'), dpi=300)

            print 'Precision-Recall curve plot saved!'
            print 'Plotting ROC curve'

            # Clear current figure
            plt.clf()

            # Plot ROC curve
            plt.plot(fpr, tpr, linewidth=2)
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.grid(True)
            plt.savefig(os.path.join(full_path, 'plot-roc_' + subdirectory + '.pdf'), dpi=300)

            print 'ROC curve plot saved!'
