# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command which acts as
the base command for the classification management commands. It encapsulates
the parameters that can be fed to the commands. The parameters are then
inherited by the classification management commands. This way, they only
need to be defined once in this class for all classifiers and RN techniques.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from optparse import make_option

from django.core.management.base import BaseCommand

from apps.auxiliary_data_app.tokenizers import ClassifierFeatureTokenizer
from apps.pulearning_app.utils import PULearningUtils
from apps.crawled_webpages_app.utils import CrawledWebpageUtility


class CustomBaseCommand(BaseCommand):

    """
    This command contains all parameter options that will be shared by
    the classification management commands.

    """

    option_list = BaseCommand.option_list + (
        make_option('--tokenizer',
            action='store',
            type='choice',
            dest='tokenizer',
            choices=('dates', 'numerics', 'words', 'entire_text'),
            metavar='TOKENIZER',
            default='entire_text',
            help='choose the tokenizer for creating training features, '
                 '[default: %default]'
        ),
        make_option('--ngram-size',
            action='store',
            type='choice',
            dest='ngram_size',
            choices=('1', '2', '3', '1,2', '1,3', '2,3'),
            metavar='NGRAM_SIZE',
            default='1',
            help='determine the size of ngrams which should be used as features, '
                 '[default: %default]'
        ),
        make_option('--apply-stemming',
            action='store_true',
            dest='stemming',
            default=False,
            help='decide whether stemming should be applied to extracted features '
                 '[default: %default]'
        ),
        make_option('--positive-data-source',
            action='store',
            dest='positive_data_source',
            default='',
            help='the data source to use for positive data; '
                 'if this is not provided, let the user choose the data source '
                 'on the command line [default: %default]'
        ),
        make_option('--unlabeled-data-source',
            action='store',
            dest='unlabeled_data_source',
            default='Te',
            help='the data source to use for unlabeled data; '
                 'if this is not provided, let the user choose the data source '
                 'on the command line [default: %default]'
        ),
        make_option('--positive-set-extension',
            action='store',
            dest='positive_set_extension',
            default='None',
            help='extend the positive set by webpages classified as '
                 'positive during the last run of the classifier'
        ),
        make_option('--noise-level',
            action='store',
            type='float',
            dest='noise_level',
            metavar='NOISE_LEVEL',
            default=0.0,
            help='set the noise level NOISE_LEVEL in range (0.0, 1.0) to '
                 'ignore a fraction of documents which are most dissimilar to '
                 'the positive representative vector / the fraction '
                 'SAMPLING_RATIO of positive documents to be '
                 'treated as spy documents [default: %default]'
        ),
        make_option('--iterations',
            action='store',
            type='int',
            dest='iterations',
            metavar='ITERATIONS',
            default=1,
            help='the number of times the algorithm should be run in sequence. '
                 'the final set of reliable negative documents will be created '
                 'by intersecting the sets of each iteration [default: %default]'
        ),
        make_option('--sampling-ratio',
            action='store',
            type='float',
            dest='sampling_ratio',
            metavar='SAMPLING_RATIO',
            default=0.0,
            help='the fraction SAMPLING_RATIO of positive documents to be '
                 'ignored or treated as spy documents [default: %default]'
        ),
        make_option('--enable-kmeans-clustering',
            action='store_true',
            dest='kmeans_clustering',
            default=False,
            help='enable kmeans clustering to obtain a purer partition of '
                 'reliable negative documents [default: %default]'
        ),
        make_option('--hand-labeled-data',
            action='store_true',
            dest='hand_labeled_data',
            default=False,
            help='select whether hand-labeled data should be used for '
                 'final labeling. This is mainly for evaluation and saving '
                 'statistics. [default: %default]'
        ),
        make_option('--rn-domains',
            action='store',
            dest='rn_domains',
            default='None',
            help='limit the amount of webpages to be labeled as '
                 'reliable negatives by certain domains [default: %default]'
        ),
        make_option('--clf-domains',
            action='store',
            dest='clf_domains',
            default='None',
            help='limit the amount of webpages to be labeled in the final step '
                 'by certain domains [default: %default]'
        ),
    )

    @classmethod
    def get_tokenizer(cls, tokenizer_option):
        """
        Map the chosen tokenizer parameter value to
        the respective tokenizer method.
        """
        opt_tok_mapping = {
            'dates': ClassifierFeatureTokenizer.tokenize_dates,
            'numerics': ClassifierFeatureTokenizer.tokenize_numerics,
            'words': ClassifierFeatureTokenizer.tokenize_words,
            'entire_text': ClassifierFeatureTokenizer.tokenize_entire_text
        }

        return opt_tok_mapping[tokenizer_option]


    @classmethod
    def get_user_input(cls):
        """
        Ask the user for positive and unlabeled data source if
        the classification commands are executed from the command line
        and not from the web interface.
        """
        print 'Please choose the text location(s) to learn features from:\n'
        print 'Locations for positive features:'
        positive_data_source = PULearningUtils.get_user_input(data_source='positive')

        print '\nLocations for unlabeled features:'
        unlabeled_data_source = PULearningUtils.get_user_input(data_source='unlabeled')

        return positive_data_source, unlabeled_data_source


class ClassifiersCustomBaseCommand(CustomBaseCommand):

    """
    This command encapsulates those optparse options that are exlusive for
    the step-2 classifiers.

    """

    option_list = CustomBaseCommand.option_list + (
        make_option('--reliable-negatives',
            action='store',
            type='choice',
            choices=('1dnf', 'cr', 'nb', 'rocchio', 'spy'),
            dest='rn_method',
            default='rocchio',
            help='the source of reliable negatives'
        ),
        make_option('--save-statistics',
            action='store_true',
            dest='save_statistics',
            default=False,
            help='select whether certain evaluation metrics should be '
                 'calculated and saved [default: %default]'
        )
    )

    @classmethod
    def _map_clf_results(cls, result_list):
        """
        The boolean parameter values Y, P and N denote whether
        a webpage is really an event page (Y - Yes), whether it has been labeled
        as positive (P) or negative (N). Map the value to its integer
        representation which will be used by the classifier commands to
        calculate the evaluation results.
        """
        mapping = {'Y': 1, 'P': 1, 'N': -1}
        return [mapping[result] for result in result_list]

    @classmethod
    def _format_eval_results(cls, left, right):
        """
        Format the evaluation results by
        adding equal amounts of spaces to each line before saving them
        to the text file.
        """
        return left + ('{:>' + str(60 + len(right) - len(left)) + '}').format(right)
