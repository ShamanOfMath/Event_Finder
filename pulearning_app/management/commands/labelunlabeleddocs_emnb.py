# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command implementing
the EM algorithm with Naive Bayesian Classification for a binary classification
of webpages.

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
from sklearn.naive_bayes import MultinomialNB
from sklearn.semisupervised_naive_bayes import SemiMultinomialNB, SemisupervisedNB

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
    either positive or negative using the EM algorithm with Bayesian
    Classification.

    The EM algorithm produces and revises the probabilistic labels of the
    documents in U - RN. The parameters are estimated in the Maximization step
    after the missing data are filled. This leads to the next iteration of the
    algorithm. EM converges when its parameters stabilize. Using NB in each
    iteration, EM employs the same equations as those used in building a NB
    classifier. The class probability given to each document in U - RN takes
    the value in [0, 1] instead of {0, 1}.

    Among others, this algorithm is used in

          Liu, B., W. Lee, P. Yu, and X. Li.
          Partially supervised classification of text documents.
          In: Proceedings of International Conference on Machine Learning
          (ICML-2002), 2002.

    Algorithm EM(P, U, RN)
    1.    Each document in P is assigned the class label 1;
    2.    Each document in RN is assigned the class label -1;
    3.    Learn an initial NB classifier f from P and RN;
    4.    Repeat
              // E-step
    5.        for each example d_i ∈ U - RN do
    6.            Use the current classifier f to compute P(c_j|d_i);
              // M-step
    7.        Learn a new NB classifier f from P, RN and U - RN by computing
              P(c_j) and P(w_t|c_j);
    8.    until the classifier parameters stabilize;
    9.    Return the classifier f from the last iteration;

    """

    option_list = ClassifiersCustomBaseCommand.option_list + (
        make_option('--apply-em',
            action='store_true',
            dest='apply_em',
            default=False,
            help='run the naive bayes algorithm iteratively using '
                 'expectation maximization [default: %default]'
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
        apply_em = options.get('apply_em')
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

        number_positive_docs = Event.objects.count()
        if positive_set_extension == 'emnb':
            number_positive_docs += CrawledWebpage.objects.filter(em_nb_tagging='P').count()
        elif positive_set_extension == 'svm':
            number_positive_docs += CrawledWebpage.objects.filter(svm_tagging='P').count()
        elif positive_set_extension == 'by_hand':
            number_positive_docs += CrawledWebpage.objects.filter(is_eventpage='Y').count()

        positive_data = VenyooDocumentUtility.webpage_generator(data_source=positive_data_source)

        if positive_set_extension not in (None, 'None', '-'):
            positive_data = chain(
                positive_data,
                CrawledWebpageUtility.webpage_generator(
                    data_source=unlabeled_data_source,
                    filter_positives=positive_set_extension,
                    exclude_hand_labeled_pages=is_hand_labeled_data)[0])

        reliable_negative_data, \
        reliable_negative_data_ids, \
        number_reliable_negative_docs = CrawledWebpageUtility.webpage_generator(
            data_source=unlabeled_data_source,
            filter_rn=rn_method)

        unlabeled_data, \
        unlabeled_data_ids, \
        number_unlabeled_docs = CrawledWebpageUtility.webpage_generator(
            data_source=unlabeled_data_source,
            exclude_positives=positive_set_extension,
            exclude_rn=rn_method,
            filter_domains=clf_domains,
            filter_hand_labeled_pages=is_hand_labeled_data)

        if apply_em:

            print 'Create X_train matrix of token counts for training...'
            X_train = vectorizer.fit_transform(
                chain(positive_data, reliable_negative_data, unlabeled_data))
            print 'X_train: ', repr(X_train), '\n'

            print 'Create y_train vector of target values (=classes)...'
            y_train = np.concatenate([
                np.array(number_positive_docs * [1]),
                np.array(number_reliable_negative_docs * [0]),
                np.array(number_unlabeled_docs * [-1])])
            print 'y_train:', y_train.shape, '\n'

        else:
            print 'Create X_train matrix of token counts for training...'
            X_train = vectorizer.fit_transform(
                chain(positive_data, reliable_negative_data))
            print 'X_train: ', repr(X_train), '\n'

            print 'Create y_train vector of target values (=classes)...'
            y_train = np.concatenate([
                np.array(number_positive_docs * [1]),
                np.array(number_reliable_negative_docs * [0])])
            print 'y_train:', y_train.shape, '\n'

        print 'Create X_test matrix of token counts for testing...'
        X_test = vectorizer.transform(
            CrawledWebpageUtility.webpage_generator(
                data_source=unlabeled_data_source,
                exclude_positives=positive_set_extension,
                exclude_rn=rn_method,
                filter_domains=clf_domains,
                filter_hand_labeled_pages=is_hand_labeled_data)[0])
        print 'X_test:', repr(X_test), '\n'

        print 'Create classifiers...'
        if apply_em:
            multinomial_classifier = SemiMultinomialNB(alpha=0.1)
            semisupervised_classifier = SemisupervisedNB(
                estimator=multinomial_classifier,
                verbose=True)

            print 'Apply EM algorithm...'
            semisupervised_classifier.fit(X=X_train, y=y_train)

            print 'Predict labels for unlabeled data...'
            predictions = semisupervised_classifier.predict(X=X_test)
            probabilities = semisupervised_classifier.predict_proba(X=X_test)

        else:
            multinomial_classifier = MultinomialNB(alpha=0.1)
            print 'Apply classifier...'
            multinomial_classifier.fit(X=X_train, y=y_train)

            print 'Predict labels for unlabeled data...'
            predictions = multinomial_classifier.predict(X=X_test)
            probabilities = multinomial_classifier.predict_proba(X=X_test)

        # Reset webpages back to 'Unlabeled'
        if positive_set_extension not in (None, 'None', '-'):
            CrawledWebpage.objects.filter(
                em_nb_tagging='N').update(em_nb_tagging='-')
        else:
            CrawledWebpage.objects.filter(
                em_nb_tagging__in=['P', 'N']).update(em_nb_tagging='-')

        print 'Label unlabeled webpages in database...'
        # The IDs of those pages which are still unlabeled
        unlabeled_data_ids = CrawledWebpageUtility.webpage_generator(
            data_source=unlabeled_data_source,
            exclude_positives=positive_set_extension,
            exclude_rn=rn_method,
            filter_domains=clf_domains,
            filter_hand_labeled_pages=is_hand_labeled_data)[1]

        # Convert the unlabeled_data_ids generator to a numpy.ndarray
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

        # Corresponds to
        # CrawledWebpage.objects.filter(is_1dnf_reliable_negative='N').count()
        reliable_negatives.update(em_nb_tagging='N')

        # Label the rest of unlabeled pages
        positive_ids = set(unlabeled_data_ids[(predictions == 1).nonzero()])
        negative_ids = set(unlabeled_data_ids[(predictions == 0).nonzero()])

        CrawledWebpage.objects.filter(id__in=positive_ids).update(em_nb_tagging='P')

        CrawledWebpage.objects.filter(id__in=negative_ids).update(em_nb_tagging='N')

        if is_hand_labeled_data:
            queryset = CrawledWebpage.objects.filter(is_eventpage__in=['Y', 'N'])
        else:
            queryset = CrawledWebpage.objects.filter(domain__name__in=clf_domains)

        positive_pages = queryset.filter(em_nb_tagging='P').count()
        negative_pages = queryset.filter(em_nb_tagging='N').count()

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

            subdirectory += 'CLF-EMNB_'
            subdirectory += 'EM-' + str(apply_em) + '_'
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
            labels_by_machine = queryset.values_list('em_nb_tagging', flat=True)

            # Map labels to their integer representations
            labels_by_hand = Command._map_clf_results(labels_by_hand)
            labels_by_machine = Command._map_clf_results(labels_by_machine)

            # Get probabilities for positive class
            # (second entry in each sublist, i.e. index 1)
            p_probs = probabilities[:, 1]

            # Get statistics for precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(
                y_true=labels_by_hand,
                probas_pred=p_probs)
            pr_auc = auc(recall, precision)

            # Get statistics for ROC curve
            fpr, tpr, roc_thresholds = roc_curve(
                y_true=labels_by_hand,
                y_score=p_probs)
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
                y_score=p_probs)

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

                print >> fobj, Command._format_eval_results('Classifier for Final Labeling:', 'EM-NB')
                print >> fobj, Command._format_eval_results('EM Applied:', str(apply_em))
                print >> fobj, Command._format_eval_results('Positive Set Extension:', positive_set_extension)
                print >> fobj, Command._format_eval_results('Features to Consider:', tokenizer)
                print >> fobj, Command._format_eval_results('N-Gram Range:', ', '.join([str(num) for num in ngram_size]))
                print >> fobj, Command._format_eval_results('Stemming Applied:', str(is_stemming))

                print >> fobj, '=' * 70 + '\n' * 2

                rn_input_total = CrawledWebpage.objects.filter(domain__name__in=rn_domains).exclude(is_eventpage__in=['Y', 'N']).count()
                rn_count = reliable_negatives.count()
                rn_percentage = '{} ({:.2f}%)'.format(rn_count, rn_count / rn_input_total * 100)
                print >> fobj, Command._format_eval_results('Documents as Input for Reliable Negatives Detection:', str(rn_input_total))
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
