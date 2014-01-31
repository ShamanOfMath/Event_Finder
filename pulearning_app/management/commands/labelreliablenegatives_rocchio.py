# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command implementing
the Rocchio technique for labeling webpages as reliable negatives.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from __future__ import division
from itertools import chain

import numpy as np
from django.core.management.base import CommandError
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
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
    reliable negatives according to the Rocchio technique.

    Two alternative methods can be employed:
    (1) Rocchio
    (2) Rocchio with Clustering

    Rocchio
    ///////
    This method treats the entire unlabeled set U as negative documents and
    then uses the positive set P and U as the training data to build a
    Rocchio classifier. The classifier is then used to classify U.
    Those documents that are classified as negative are considered (reliable)
    negative data, denoted by RN.

    Rocchio with Clustering
    ///////////////////////
    Rocchio is a linear classifier based on cosine similarity. When the
    decision boundary is non-linear or does not conform to the separating
    plane resulted from cosine similarity, Rocchio may still extract some
    positive documents and put them in RN. We propose an enhancement to
    the Rocchio approach in order to purify RN further, i.e., to discard
    some likely positive documents from RN.

    This approach uses clustering to partition RN into many similarity groups
    (or clusters). It then uses Rocchio again to build a classifier using each
    cluster and the positive set P. The classifier is then applied to identify
    likely positive documents in the cluster and delete them. The idea is to
    identify and remove some positive documents in RN in a localized manner,
    which gives better accuracy. Since both the cluster and the positive set
    are homogeneous, they allow Rocchio to build better prototype vectors.

    The clustering technique that we use is k-means, which is an efficient
    technique. The k-means method needs the input cluster number k. This new
    method can further purify RN without removing too many negative documents
    from RN.

    Among others, this technique is used in

          Li, X. and B. Liu.
          Learning to classify texts using positive and unlabeled data.
          In: Proceedings of International Joint Conference on Artificial
          Intelligence (IJCAI-2003), 2003.

    Algorithm Rocchio(P, U)
    1.    RN = ∅;
    2.    c_P  = ( α / |P| ) * ∑_d∈P ( d / ||d|| ) - ( β / |U| ) * ∑_d∈U ( d / ||d|| );
    3.    c_N = ( α / |U| ) * ∑_d∈U ( d / ||d|| ) - ( β / |P| ) * ∑_d∈P ( d / ||d|| );
    4.    for each d ∈ U do
    5.        if cos(c_N, d) > cos(c_P, d) then
    6.            RN = RN ∪ {d};

    Algorithm Rocchio with k-means clustering(P, RN)
    1.    Perform the Rocchio algorithm as shown above and generate the initial
          negative set RN;
    2.    Choose k initial clusters { O_1, O_2, ..., O_k } randomly from RN;
    3.    Perform k-means clustering to produce k clusters, i.e. N_1, N_2, ..., N_k;
    4.    for j = 1 to k do
    5.        n_j = ( α / |N_j| ) * ∑_d∈N_j ( d / ||d|| ) - ( β / |P| ) * ∑_d∈P ( d / ||d|| );
    6.        p_j = ( α / |P| ) * ∑_d∈P ( d / ||d|| ) - ( β / |N_j| ) * ∑_d∈N_j ( d / ||d|| );
    7.    RN' = ∅;
    8.    for each document d_i ∈ RN do
    9.        Find the nearest positive prototype vector p_v to d_i, where
              v = arg max(j): cos(p_j, d_i);
    10.       if there exists a n_j (j = 1, 2, ..., k), so that
              cos(n_j, d_i) > cos(p_v, d_i) then
    11.           RN' = RN' ∪ {d_i};

    """

    def handle(self, *args, **options):
        """"""
        if args:
            raise CommandError('this command does not support any non-keyword arguments')

        tokenizer = options.get('tokenizer')
        is_stemming = options.get('stemming')
        apply_kmeans_clustering = options.get('kmeans_clustering')
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

        unlabeled_data, unlabeled_data_ids, number_unlabeled_docs = CrawledWebpageUtility.webpage_generator(
            data_source=unlabeled_data_source,
            exclude_positives=positive_set_extension,
            filter_domains=rn_domains,
            exclude_hand_labeled_pages=is_hand_labeled_data)

        # Create sparse matrices of feature counts
        print 'Calculate feature counts for both positive und unlabeled data...'
        X_positive_plus_unlabeled = vectorizer.fit_transform(
            chain(positive_data, unlabeled_data))
        print 'X_positive_plus_unlabeled', repr(X_positive_plus_unlabeled), '\n'

        # Transform counts into TF-IDF values
        print 'Transform counts into TF-IDF values...'
        X_positive_plus_unlabeled_tfidf = tfidf_transformer.fit_transform(X_positive_plus_unlabeled)
        print repr(X_positive_plus_unlabeled_tfidf), '\n'

        number_positive_docs = Event.objects.count()
        if positive_set_extension == 'emnb':
            number_positive_docs += CrawledWebpage.objects.filter(em_nb_tagging='P').count()
        elif positive_set_extension == 'svm':
            number_positive_docs += CrawledWebpage.objects.filter(svm_tagging='P').count()
        elif positive_set_extension == 'by_hand':
            number_positive_docs += CrawledWebpage.objects.filter(is_eventpage='Y').count()

        # Split the matrix into positive and unlabeled data
        X_positive_tfidf = X_positive_plus_unlabeled_tfidf[:number_positive_docs]
        print repr(X_positive_tfidf), '\n'

        X_unlabeled_tfidf = X_positive_plus_unlabeled_tfidf[number_positive_docs:]
        print repr(X_unlabeled_tfidf), '\n'

        # Initialize helper vectors for holding the sums d / ||d|| for each
        # document in P and U:
        positive_helper_vector = csr_matrix((1, X_positive_tfidf.shape[1]))
        unlabeled_helper_vector = csr_matrix((1, X_positive_tfidf.shape[1]))

        # Calculate the sum of positive vectors
        print 'Calculate sum of positive vectors...'
        for vector in X_positive_tfidf:
            if vector.sum() > 0:
                positive_helper_vector = positive_helper_vector + (vector / np.sqrt(np.dot(vector, vector.T).toarray()[0])[0])

        # Calculate the sum of unlabeled vectors
        print 'Calculate sum of unlabeled vectors...'
        for vector in X_unlabeled_tfidf:
            if vector.sum() > 0:
                unlabeled_helper_vector = unlabeled_helper_vector + (vector / np.sqrt(np.dot(vector, vector.T).toarray()[0])[0])

        # Set variables as suggested in [4]
        alpha = 16
        beta = 4

        # Calculate prototype vectors c_P and c_U
        print 'Calculate prototype vectors...'
        c_P = (alpha / number_positive_docs) * positive_helper_vector -\
              (beta / number_unlabeled_docs) * unlabeled_helper_vector

        c_U = (alpha / number_unlabeled_docs) * unlabeled_helper_vector -\
              (beta / number_positive_docs) * positive_helper_vector

        # Iterate over all unlabeled documents and calculate their cosine
        # values with the prototype vectors. If the cosine for the potential
        # negative prototype vector is greater than the cosine for the
        # positive prototype vector, treat the respective unlabeled document
        # as a reliable negative document.
        print 'Determine reliable negative documents...'
        reliable_negative_indices = set()
        reliable_negative_ids = []

        # This set will hold the indices of those reliable negative documents
        # which have been identified as pure reliable negative
        pure_reliable_negative_ids = set()

        for i, vector in enumerate(X_unlabeled_tfidf):
            current_id = unlabeled_data_ids.next()

            if vector.sum() > 0:
                positive_cosine = (np.dot(c_P, vector.T).toarray()[0] / np.dot(
                    np.sqrt(np.dot(vector, vector.T).toarray()[0]),
                    np.sqrt(np.dot(c_P, c_P.T).toarray()[0])))[0]

                unlabeled_cosine = (np.dot(c_U, vector.T).toarray()[0] / np.dot(
                    np.sqrt(np.dot(vector, vector.T).toarray()[0]),
                    np.sqrt(np.dot(c_U, c_U.T).toarray()[0])))[0]

                if unlabeled_cosine > positive_cosine:
                    reliable_negative_indices.add(i)
                    reliable_negative_ids.append(current_id)

        # Get the matrix of reliable negative documents
        X_reliable_negatives_tfidf = X_unlabeled_tfidf[sorted(reliable_negative_indices)]

        if apply_kmeans_clustering:

            kmeans_estimator = KMeans(
                n_clusters=5,
                n_jobs=-1
            )

            print 'X_reliable_negatives_tfidf', repr(X_reliable_negatives_tfidf)

            print '\nApply k-means clustering to reliable negatives...'
            cluster_indices = kmeans_estimator.fit_predict(X_reliable_negatives_tfidf)

            # This dictionary will hold the prototype vectors for each cluster
            prototype_vectors = {}

            print 'Calculate prototype vector for each cluster...'
            for cluster_index in xrange(kmeans_estimator.n_clusters):

                # Identify the document indices belonging to the same cluster
                document_indices = np.where(cluster_indices == cluster_index)[0]

                # Get the matrix that contains only these documents
                X_cluster_docs_tfidf = X_reliable_negatives_tfidf[document_indices]

                # Calculate the sums for negative documents
                negative_helper_vector = csr_matrix((1, X_cluster_docs_tfidf.shape[1]))
                for vector in X_cluster_docs_tfidf:
                    if vector.sum() > 0:
                        negative_helper_vector = negative_helper_vector + (vector / np.sqrt(np.dot(vector, vector.T).toarray()[0])[0])

                # Calculate positive and negative prototype vectors for each cluster
                c_P = (alpha / number_positive_docs) * positive_helper_vector -\
                      (beta / X_cluster_docs_tfidf.shape[0]) * negative_helper_vector

                c_N = (alpha / X_cluster_docs_tfidf.shape[0]) * negative_helper_vector -\
                      (beta / number_positive_docs) * positive_helper_vector

                # Add the vectors to the dictionary
                prototype_vectors[cluster_index] = (c_P, c_N)

            print 'Determine pure reliable negative documents...'
            for i, vector in enumerate(X_reliable_negatives_tfidf):
                current_id = reliable_negative_ids[i]

                # This dictionary holds the cosine similarity values
                # for each positive prototype vector
                similarity_values = {}

                # Find the nearest positive document vector
                # to this document vector
                if vector.sum() > 0:

                    for cluster_index in prototype_vectors:

                        c_P = prototype_vectors[cluster_index][0]

                        cosine_similarity = (np.dot(c_P, vector.T).toarray()[0] / np.dot(
                            np.sqrt(np.dot(vector, vector.T).toarray()[0]),
                            np.sqrt(np.dot(c_P, c_P.T).toarray()[0])))[0]

                        similarity_values[cluster_index] = cosine_similarity

                    # Sort the cosine values in decreasing order
                    similarity_values_sorted = sorted(
                        similarity_values.iteritems(),
                        key=lambda (k, v): (v, k),
                        reverse=True)

                    # Take the prototype vector with the highest similarity
                    highest_similarity_value = similarity_values_sorted[0][1]

                    for cluster_index in prototype_vectors:

                        # Search for a negative prototype vector whose
                        # cosine similarity with the document vector is higher
                        # than the nearest positive prototype vector
                        # if this is the case, add it to the set of the
                        # pure reliable negative documents
                        c_N = prototype_vectors[cluster_index][1]

                        cosine_similarity = (np.dot(c_N, vector.T).toarray()[0] / np.dot(
                            np.sqrt(np.dot(vector, vector.T).toarray()[0]),
                            np.sqrt(np.dot(c_N, c_N.T).toarray()[0])))[0]

                        if cosine_similarity > highest_similarity_value:
                            pure_reliable_negative_ids.add(current_id)
                            break

        # Reset all negative webpages back to 'Unlabeled'
        CrawledWebpage.objects.filter(
            is_rocchio_reliable_negative='N').update(
            is_rocchio_reliable_negative='-')

        # Label the reliable negative documents in the database
        # Hint: database index = matrix row index + 1
        print 'Label reliable negatives in database...'
        if apply_kmeans_clustering:
            affected_pages = CrawledWebpage.objects.filter(
                id__in=pure_reliable_negative_ids).update(
                is_rocchio_reliable_negative='N')
        else:
            affected_pages = CrawledWebpage.objects.filter(
                id__in=reliable_negative_ids).update(
                is_rocchio_reliable_negative='N')

        print 'Done! Annotation of unlabeled data successful.'
        print affected_pages, 'pages have been annotated as reliable negatives.'
