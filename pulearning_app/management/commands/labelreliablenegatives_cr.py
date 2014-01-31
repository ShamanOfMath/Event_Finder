# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command implementing
the Cosine-Rocchio technique for labeling webpages as reliable negatives.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from __future__ import division
from itertools import chain

import numpy as np
from django.core.management.base import CommandError
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer

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
    reliable negatives according to the Cosine-Rocchio technique.

    It consists of two sub-steps:

    Sub-step 1 (lines 1–9)
    //////////////////////
    This sub-step extracts a set of potential negatives PN from U by
    computing similarities of the unlabeled documents in U with the positive
    documents in P. Those documents in U that are very dissimilar to the
    documents in P are likely to be negative (lines 7–9).

    To make the decisions, a similarity measure and a similarity threshold are
    needed. The similarity measure is the well-known cosine similarity.
    To compute the similarity, each document in P and U is first converted to
    a vector d using the TF-IDF scheme. The positive documents in P are used to
    compute the threshold value. First, a positive representative vector (v_P)
    is constructed by summing up the documents in P (line 3). The similarity of
    each document d in P with v_P is calculated using the cosine measure,
    cos(v_P, d), in line 4.

    Line 5 sorts the documents in P according to their cos(v_P, d) values,
    which helps to determine the similarity threshold. The threshold is used to
    filter out as many as possible hidden positive documents from U so that a
    very pure negative set PN can be obtained. Since the hidden positives in U
    should have the same behaviors as the positives in P in terms of their
    similarities to v_P, ideally we should set the minimum similarity value of
    all documents d ∈ P and v_P as the threshold value ω.

    However, we need to consider possible noise in P. It would therefore be
    prudent to ignore a small percentage l of documents in P that are most
    dissimilar to v_P and assume them to be noise or outliers. The default
    noise level of l = 5% is used. In line 6, l is used to decide the
    similarity threshold ω. Then, for each document d in U, if its cosine
    similarity cos(v_P, d) is less than ω, it is regarded as a potential
    negative and stored in PN (lines 8–9). PN, however, is still not sufficient
    for accurate learning. Using PN, sub-step 2 produces the final RN.

    Sub-step 2 (lines 10-14)
    ////////////////////////
    To extract the final reliable negatives, the algorithm employs the Rocchio
    classification method to build a classifier f using P and PN. Those
    documents in U that are classified as negatives by f are regarded as the
    final reliable negatives and stored in set RN.

    Following the Rocchio formula, the classifier f actually consists of a
    positive and a negative prototype vectors c_P and c_PN (lines 11 and 12).
    α and β are parameters for adjusting the relative impact of the examples
    in P and PN. α = 16 and β = 4 are used as default values. The classification
    is done in lines 12–14.

    Among others, this technique is used in

          Li, X., B. Liu, and S. Ng.
          Negative Training Data can be Harmful to Text Classification.
          In: Proceedings of Conference on Empirical Methods in
          Natural Language Processing (EMNLP-2010), 2010.

    Algorithm Cosine-Rocchio(P, U)
    1.    PN = ∅; RN = ∅;
    2.    Represent each document d ∈ P and U as a vector using the TF-IDF scheme;
    3.    v_P = ( 1 / |P| ) * ∑_d∈P ( d / ||d|| );
    4.    Compute cos(v_P, d) for each d ∈ P;
    5.    Sort all documents d ∈ P according to cos(v_P, d) in decreasing order;
    6.    ω = cos(v_P, d) where d is ranked in position of (1-l) * |P|;
    7.    for each d ∈ U do
    8.        if cos(v_P, d) < ω then
    9.            PN = PN ∪ {d};
    10.   c_P  = ( α / |P| ) * ∑_d∈P ( d / ||d|| ) - ( β / |PN| ) * ∑_d∈PN ( d / ||d|| );
    11.   c_PN = ( α / |PN| ) * ∑_d∈PN ( d / ||d|| ) - ( β / |P| ) * ∑_d∈P ( d / ||d|| );
    12.   for each d ∈ U do
    13.       if cos(c_PN, d) > cos(c_P, d) then
    14.           RN = RN ∪ {d};

    """

    def handle(self, *args, **options):
        """"""
        if args:
            raise CommandError(
                'this command does not support any non-keyword arguments')

        tokenizer = options.get('tokenizer')
        noise_level = options.get('noise_level')
        is_stemming = options.get('stemming')
        positive_data_source = options.get('positive_data_source')
        unlabeled_data_source = options.get('unlabeled_data_source')
        rn_domains = options.get('rn_domains').split(',')
        positive_set_extension = options.get('positive_set_extension')
        is_hand_labeled_data = options.get('hand_labeled_data')

        ngram_size = [int(num) for num in options.get('ngram_size').split(',')]
        if len(ngram_size) == 1:
            ngram_size.append(ngram_size[0])

        if not 0.0 <= noise_level <= 1.0:
            raise CommandError('noise level must be in range (0.0, 1.0)')

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

        tfidf_transformer = TfidfTransformer()

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

        # Create sparse matrices of feature counts
        print 'Calculate feature counts for both positive and unlabeled data...'
        X_positive_plus_unlabeled = vectorizer.fit_transform(chain(positive_data, unlabeled_data))
        print 'X_positive_plus_unlabeled', repr(X_positive_plus_unlabeled), '\n'

        # Line 2 of the algorithm
        print 'Transform counts into TF-IDF values...'
        X_positive_plus_unlabeled_tfidf = tfidf_transformer.fit_transform(X_positive_plus_unlabeled)
        print repr(X_positive_plus_unlabeled_tfidf), '\n'

        # Determine the total number of positive documents |P|
        total_positive_docs = Event.objects.count()
        if positive_set_extension == 'emnb':
            total_positive_docs += CrawledWebpage.objects.filter(em_nb_tagging='P').count()
        elif positive_set_extension == 'svm':
            total_positive_docs += CrawledWebpage.objects.filter(svm_tagging='P').count()
        elif positive_set_extension == 'by_hand':
            total_positive_docs += CrawledWebpage.objects.filter(is_eventpage='Y').count()

        # Split the matrix into positive and unlabeled data
        X_positive_tfidf = X_positive_plus_unlabeled_tfidf[:total_positive_docs]
        print repr(X_positive_tfidf), '\n'

        X_unlabeled_tfidf = X_positive_plus_unlabeled_tfidf[total_positive_docs:]
        print repr(X_unlabeled_tfidf), '\n'

        # Initialize helper vectors for holding the sums d / ||d|| for each
        # document in P and PN:
        positive_helper_vector = csr_matrix((1, X_positive_tfidf.shape[1]))
        potential_negative_helper_vector = csr_matrix((1, X_positive_tfidf.shape[1]))

        # Line 3 of the algorithm
        print 'Calculate sum of positive vectors...'
        for vector in X_positive_tfidf:
            # v_p += (vector / np.linalg.norm(vector.todense()))
            # --> the above raises NotImplementedError !!!

            # Converting every vector to dense array is not very efficient !
            # v_p = v_p + (vector / np.linalg.norm(vector.todense()))

            # Better do it this way:
            #v_p = v_p + (vector / np.sqrt(np.dot(vector, vector.T).toarray()[0])[0])
            #positive_helper_vector = positive_helper_vector + (vector / np.linalg.norm(vector.todense()))

            # Avoid division by zero by checking whether the sum of
            # a vector's values is equal to zero. A vector may contain zeros
            # in cases where the respective field in the database is empty,
            # e.g. if features are only taken from a website's title and a
            # website doesn't have a title. Then the corresponding vector
            # contains only zeros.
            if vector.sum() > 0:
                positive_helper_vector = positive_helper_vector + (vector / np.sqrt(np.dot(vector, vector.T).toarray()[0])[0])

        # Normalize the vector by the number of positive documents
        # and create the positive representative vector
        v_P = positive_helper_vector * (1 / total_positive_docs)

        # Line 4 of the algorithm
        # Compute the cosine between the positive representative vector
        # and each positive document vector.
        # Save the results in a dictionary with the matrix row index as key
        # and the cosine value as value (saved as an np.ndarray()).
        print 'Calculate cosine values for positive documents...'
        positive_cosine_values = {}
        for i, vector in enumerate(X_positive_tfidf):
            if vector.sum() > 0:
                positive_cosine_values[i] = (np.dot(v_P, vector.T).toarray()[0] / np.dot(
                    np.sqrt(np.dot(vector, vector.T).toarray()[0]),
                    np.sqrt(np.dot(v_P, v_P.T).toarray()[0])))[0]
            else:
                positive_cosine_values[i] = None

        # Lines 5 and 6 of the algorithm
        # If noise level is less than 1.0, treat a fraction of
        # noise_level * |P| as outliers and recalculate the positive
        # representative vector
        if 0 < noise_level < 1:

            # Sort entries whose values are not None in descending order
            positive_cosine_values_sorted = sorted(
                {i: positive_cosine_values[i]
                 for i in positive_cosine_values
                 if positive_cosine_values[i] is not None}.iteritems(),
                key=lambda (k, v): (v, k),
                reverse=True)

            # Determine number of documents to ignore
            num_docs_to_ignore = int(noise_level * total_positive_docs)

            # Determine number of documents to consider for further calculations
            num_docs_to_consider = total_positive_docs - num_docs_to_ignore

            # Create a fraction of documents
            positive_cosine_values_fraction = dict(positive_cosine_values_sorted[:num_docs_to_consider])

            # Determine the threshold from this fraction
            threshold = min(positive_cosine_values_fraction.values())

            # Get the list of indices of this fraction
            fraction_indices = positive_cosine_values_fraction.keys()
            print num_docs_to_ignore, \
                'documents from the positive set were labeled as noise and will be ignored...'

            # Recreate the variables `total_positive_docs`, `positive_helper_vector`, `v_P`
            total_positive_docs = len(fraction_indices)
            positive_helper_vector = csr_matrix((1, X_positive_tfidf.shape[1]))
            for vector in X_positive_tfidf[fraction_indices]:
                positive_helper_vector = positive_helper_vector + (vector / np.sqrt(np.dot(vector, vector.T).toarray()[0])[0])
            v_P = positive_helper_vector * (1 / total_positive_docs)
        else:
            threshold = min([value for value in positive_cosine_values.values() if value is not None])
        print 'The determined cosine threshold is:', threshold

        # Lines 7 to 9 of the algorithm
        # Now, iterate over the unlabeled documents and compute their
        # cosine values with the positive representative vector v_P.
        # If a cosine value is lower than the threshold, put it into the set
        # of potential negative documents.
        print 'Calculate cosine values for unlabeled documents...'
        unlabeled_cosine_values = {}
        for i, vector in enumerate(X_unlabeled_tfidf):
            if vector.sum() > 0:
                unlabeled_cosine_values[i] = (np.dot(v_P, vector.T).toarray()[0] / np.dot(
                    np.sqrt(np.dot(vector, vector.T).toarray()[0]),
                    np.sqrt(np.dot(v_P, v_P.T).toarray()[0])))[0]
            else:
                unlabeled_cosine_values[i] = None

        potential_negatives_indices = []
        for i in unlabeled_cosine_values:
            if unlabeled_cosine_values[i] < threshold:
                # Database row index = matrix row index + 1
                potential_negatives_indices.append(i)

        # Determine the total number of potential negative docs
        total_potential_negative_docs = len(potential_negatives_indices)

        # Calculate the sum of potential negative vectors
        print 'Calculate sum of potential negative vectors...'
        for vector in X_unlabeled_tfidf[potential_negatives_indices]:
            if vector.sum() > 0:
                potential_negative_helper_vector = potential_negative_helper_vector + \
                (vector / np.sqrt(np.dot(vector, vector.T).toarray()[0])[0])

        # Setting parameters for adjusting the relative impact of the examples
        # in P and RN
        alpha = 16
        beta = 4

        # Line 10 of the algorithm
        # Calculate positive prototype vector c_P
        print 'Calculate prototype vectors...'
        c_P = (alpha / total_positive_docs) * positive_helper_vector - \
        (beta / total_potential_negative_docs) * potential_negative_helper_vector

        # Line 11 of the algorithm
        # Calculate potential negative prototype vector c_PN
        c_PN = (alpha / total_potential_negative_docs) * potential_negative_helper_vector - \
        (beta / total_positive_docs) * positive_helper_vector

        # Lines 12 to 14 of the algorithm
        # Iterate over all unlabeled documents and calculate their cosine
        # values with the prototype vectors. If the cosine for the potential
        # negative prototype vector is greater than the cosine for the
        # positive prototype vector, treat the respective unlabeled document
        # as a reliable negative document.
        print 'Determine reliable negative documents...'
        reliable_negative_ids = []
        for vector in X_unlabeled_tfidf:
            current_id = unlabeled_data_ids.next()

            if vector.sum() > 0:
                positive_cosine = (np.dot(c_P, vector.T).toarray()[0] / np.dot(
                    np.sqrt(np.dot(vector, vector.T).toarray()[0]),
                    np.sqrt(np.dot(c_P, c_P.T).toarray()[0])))[0]

                potential_negative_cosine = (np.dot(c_PN, vector.T).toarray()[0] / np.dot(
                    np.sqrt(np.dot(vector, vector.T).toarray()[0]),
                    np.sqrt(np.dot(c_PN, c_PN.T).toarray()[0])))[0]

                if potential_negative_cosine > positive_cosine:
                    reliable_negative_ids.append(current_id)

        # Reset all negative webpages back to 'Unlabeled'
        CrawledWebpage.objects.filter(
            is_cosine_rocchio_reliable_negative='N').update(
            is_cosine_rocchio_reliable_negative='-')

        # Label the reliable negative documents in the database
        print 'Label reliable negatives in database...'
        affected_pages = CrawledWebpage.objects.filter(
            id__in=reliable_negative_ids).update(
            is_cosine_rocchio_reliable_negative='N')

        print 'Done! Annotation of unlabeled data successful.'
        print affected_pages, ' pages have been annotated as reliable negatives.'

