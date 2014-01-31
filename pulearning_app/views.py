# -*- coding: utf-8 -*-

"""
This module contains the views for the pulearning_app.

Generally, a view is a Python function that takes a
Web request and returns a Web response. This response can
be the HTML contents of a Web page, a redirect, a 404 error,
an XML document, an image... There is also a range of
generic class-based views that take certain common idioms and
patterns found in view development and abstract them so that one can
quickly write common views of data without having to write too much code.

This file contains all the server-side logic of the entire web application.

For an overview about Django's function-based views, see:
    https://docs.djangoproject.com/en/1.5/topics/http/views/

For an overview about Django's class-based views, see:
    https://docs.djangoproject.com/en/1.5/topics/class-based-views/

For a detailed explanation of Django's class-based views, see:
    https://docs.djangoproject.com/en/1.5/ref/class-based-views/

"""


from __future__ import division
import json
import locale
import math
import os
import signal
import subprocess
from urlparse import urlparse

from django.conf import settings
from django.db.models import Count
from django.http import HttpResponse
from django.shortcuts import render
from haystack.inputs import AutoQuery
from haystack.models import SearchResult
from haystack.query import SearchQuerySet
from haystack.utils import Highlighter
import pytz

from apps.auxiliary_data_app.utils import CommonUtils
from apps.crawled_webpages_app.models import Domain, CrawledWebpage
from apps.pulearning_app.forms import WebPageCrawlerForm
from apps.pulearning_app.forms import DataAndClassifierSelectionForm
from apps.pulearning_app.forms import HaystackSearchForm
from apps.venyoo_events_app.models import Event

locale.setlocale(locale.LC_ALL, 'de_DE')


def provide_main_page_forms(request):
    """
    Provide all forms for this application and add them to the template context.
    Render the index html file that serves as the single extry point for this
    application.
    """
    template_context = {
        'webpage_crawler_form': WebPageCrawlerForm(),
        'd_and_c_selection_form': list(DataAndClassifierSelectionForm()),
        'haystack_search_form': list(HaystackSearchForm())}

    return render(request, 'index.html', template_context)


def return_search_results_ajax(request):
    """
    Process queries issued from the haystack search form and
    validate the form. If the form is valid, highlight the queried terms and
    return the top hundred search results as highlighted snippets.

    NOTE: Due to performance issues, only the top one hundred search results
    will be returned at maximum, no matter how many search results have been
    found. This shortcoming can be improved by providing more efficient
    database queries and adding a sophisticated caching functionality.
    """
    haystack_search_form = HaystackSearchForm(request.GET)
    response = {}

    if haystack_search_form.is_valid():
        search_query = haystack_search_form.cleaned_data['search_query']
        search_source = haystack_search_form.cleaned_data['search_source']
        max_results = haystack_search_form.cleaned_data['max_results']

        search_source_to_model_name = {
            'venyoo_events': 'event',
            'crawled_webpages': 'crawledwebpage'}

        highlighter = Highlighter(
            search_query,
            html_tag='strong',
            css_class='highlighted',
            max_length=250)

        search_results = SearchQuerySet().filter(content=AutoQuery(search_query))
        end = int(math.ceil(search_results.count() / 1000))

        results = []
        webpage_urls = []
        highlighted_snippets = []

        a, b = 0, 1000

        for i in xrange(end):
            if search_source in ('venyoo_events', 'crawled_webpages'):
                results = results + \
                    [result for result in search_results[a:b]
                     if isinstance(result, SearchResult)
                     and result.model_name ==
                     search_source_to_model_name[search_source]]
            else:
                results = results +\
                    [result for result in search_results[a:b]
                     if isinstance(result, SearchResult)]

            webpage_urls = webpage_urls +\
                           [result.get_stored_fields()['url'] for result
                            in results[a:b]]

            highlighted_snippets = highlighted_snippets +\
                                   [highlighter.highlight(result.text) for
                                    result in results[a:b]]
            a += 1000
            b += 1000

        results_total = len(results)

        response['results_total'] = results_total
        response['results_shown'] = max_results if max_results <= results_total else results_total
        response['webpage_urls'] = webpage_urls[:max_results]
        response['highlighted_snippets'] = highlighted_snippets[:max_results]

    return HttpResponse(json.dumps(response), mimetype='application/json')


def load_domain_names_ajax(request):
    """
    Upon loading this web application in the browser for the first time,
    get a list of all domains for which there are crawled webpages in the
    database, together with the number of crawled webpages for each domain.
    This list will be displayed in the classification results section.
    """
    domain_queryset = Domain.objects.order_by('name').annotate(Count('crawledwebpage'))
    domain_names = list(domain_queryset.values_list('name', flat=True))
    crawled_pages = [domain.crawledwebpage__count for domain in domain_queryset]

    response = {
        'domain_names': domain_names,
        'crawled_pages': crawled_pages
    }

    return HttpResponse(json.dumps(response), mimetype='application/json')


def load_classification_results_ajax(request):
    """
    Return classification results for a specific combination of domain name and
    classification method. If a method for step 1 has been chosen, return a set
    of reliable negative webpages and a set of unlabeled webpages. If a method
    for step 2 has been chosen, return a set of positive webpages and a set
    of negative webpages.
    """
    classification_method = request.GET['classification_method']
    domain_name = request.GET['domain_name']

    response = {}

    crawledwebpage_set = Domain.objects.get(name=domain_name).crawledwebpage_set

    step1_methods = ['1dnf', 'cr', 'nb', 'rocchio', 'spy']
    step2_methods = ['emnb', 'svm', 'by-hand']

    filter_exp_start = 'list(crawledwebpage_set.filter('
    filter_exp_middle = ''
    tag_positive = '="P"'
    tag_negative = '="N"'
    tag_unlabeled = '="-"'
    filter_exp_end = ').order_by("url").values_list("url", flat=True))'

    if classification_method in step1_methods:

        if classification_method == '1dnf':
            filter_exp_middle = 'is_1dnf_reliable_negative'

        elif classification_method == 'cr':
            filter_exp_middle = 'is_cosine_rocchio_reliable_negative'

        elif classification_method == 'nb':
            filter_exp_middle = 'is_nb_reliable_negative'

        elif classification_method == 'rocchio':
            filter_exp_middle = 'is_rocchio_reliable_negative'

        elif classification_method == 'spy':
            filter_exp_middle = 'is_spy_reliable_negative'

        reliable_negatives = eval(filter_exp_start + filter_exp_middle + tag_negative + filter_exp_end)
        unlabeled = eval(filter_exp_start + filter_exp_middle + tag_unlabeled + filter_exp_end)

        # Fill up the short list to match the length of the longer list
        amount_of_unlabeled = len(unlabeled)
        amount_of_reliable_negatives = len(reliable_negatives)
        difference = abs(amount_of_unlabeled - amount_of_reliable_negatives)

        if amount_of_reliable_negatives < amount_of_unlabeled:
            reliable_negatives += difference * ['']
        elif amount_of_unlabeled < amount_of_reliable_negatives:
            unlabeled += difference * ['']

        response['urls'] = zip(reliable_negatives, unlabeled)
        response['reliable_negatives'] = amount_of_reliable_negatives
        response['unlabeled'] = amount_of_unlabeled

    elif classification_method in step2_methods:

        if classification_method == 'emnb':
            filter_exp_middle = 'em_nb_tagging'

        elif classification_method == 'svm':
            filter_exp_middle = 'svm_tagging'

        elif classification_method == 'by-hand':
            filter_exp_middle = 'is_eventpage'
            tag_positive = '="Y"'

        positives = eval(filter_exp_start + filter_exp_middle + tag_positive + filter_exp_end)
        negatives = eval(filter_exp_start + filter_exp_middle + tag_negative + filter_exp_end)

        # Fill up the short list to macht the length of the longer list
        amount_of_positives = len(positives)
        amount_of_negatives = len(negatives)
        difference = abs(amount_of_negatives - amount_of_positives)

        if amount_of_positives < amount_of_negatives:
            positives += difference * ['']
        elif amount_of_negatives < amount_of_positives:
            negatives += difference * ['']

        response['urls'] = zip(positives, negatives)
        response['positives'] = amount_of_positives
        response['negatives'] = amount_of_negatives

    return HttpResponse(json.dumps(response), mimetype='application/json')


def get_webpage_details_ajax(request):
    """Return the content of a requested crawled webpage."""
    page_url = request.GET['page_url']
    if 'venyoo.de' in urlparse(page_url).netloc:

        event_queryset = Event.objects.filter(url=page_url)
        event_values = event_queryset.values(
            'title', 'summary', 'start_date', 'end_date')[0]
        event = event_queryset[0]

        loc = event.location
        full_address = '<br />'.join((
            loc.name,
            loc.street_address,
            ' '.join((loc.postal_code, loc.town.name)),
            loc.town.federate_state.name,
            '<br />'.join(
                sorted(
                    (''.join(
                        ('<a href="',
                         website.url,
                         '" title="',
                         website.url,
                         '" target="_blank">',
                         website.url,
                         '</a>')) for website in loc.websites.all())
                )
            )
        ))

        categories = '<br />'.join(sorted(
            (category.title for category in event.categories.all())
        ))

        event_values['full_address'] = full_address
        event_values['categories'] = categories
        event_values['user'] = event.user.name

        timezone_new = pytz.timezone('Europe/Berlin')
        date_format_str = '%A, %d. %B %Y, %H:%M Uhr %Z'

        try:
            start_date_converted = event_values['start_date'].astimezone(timezone_new)
            start_date_str = start_date_converted.strftime(date_format_str)
            event_values['start_date'] = start_date_str
        except AttributeError:
            event_values['start_date'] = ''

        try:
            end_date_converted = event_values['end_date'].astimezone(timezone_new)
            end_date_str = end_date_converted.strftime(date_format_str)
            event_values['end_date'] = end_date_str
        except AttributeError:
            event_values['end_date'] = ''

        response = {
            'venyoo_event': event_values,
            'page_url': page_url
        }

    else:
        webpage = CrawledWebpage.objects.filter(url=page_url).values(
            'html_text', 'html_meta', 'html_title', 'html_headings',
            'html_lists', 'html_tables', 'html_emphs')[0]
        webpage['html_meta'] = ''.join((
            '<pre class="page-meta">',
            webpage['html_meta'].replace('<', '&lt;').replace('>', '&gt;'),
            '</pre>'))
        response = {
            'webpage': webpage,
            'page_url': page_url
        }

    return HttpResponse(json.dumps(response), mimetype='application/json')


def process_data_and_classifier_selection_form_ajax(request):
    """
    Process the input from the data and classifier selection form and
    check whether it is valid. If so, add the selected options to option
    strings. These option strings will serve as optparse options for the
    custom management commands in the package
    apps.pulearning_app.management.commands.

    Additionally, this view starts the subprocess for classifying reliable
    negative webpages.
    """
    d_and_c_selection_form = DataAndClassifierSelectionForm(request.POST)
    response = {}

    if d_and_c_selection_form.is_valid():

        # obligatory fields
        positive_data_source = d_and_c_selection_form.cleaned_data['positive_data_source']
        unlabeled_data_source = d_and_c_selection_form.cleaned_data['unlabeled_data_source']
        rn_method = d_and_c_selection_form.cleaned_data['rn_method']
        rn_method_domains = d_and_c_selection_form.cleaned_data['rn_method_domains']
        classifier = d_and_c_selection_form.cleaned_data['classifier']
        classifier_domains = d_and_c_selection_form.cleaned_data['classifier_domains']
        hand_labeled_data = d_and_c_selection_form.cleaned_data['hand_labeled_data']
        positive_set_extension = d_and_c_selection_form.cleaned_data['positive_set_extension']
        tokenizer = d_and_c_selection_form.cleaned_data['tokenizer']
        ngram_size = d_and_c_selection_form.cleaned_data['ngram_size']
        apply_stemming = d_and_c_selection_form.cleaned_data['apply_stemming']
        save_statistics = d_and_c_selection_form.cleaned_data['save_statistics']

        # optional fields
        sampling_ratio = d_and_c_selection_form.cleaned_data['sampling_ratio']
        iterations = d_and_c_selection_form.cleaned_data['iterations']
        noise_level = d_and_c_selection_form.cleaned_data['noise_level']
        kmeans_clustering = d_and_c_selection_form.cleaned_data['kmeans_clustering']
        apply_em = d_and_c_selection_form.cleaned_data['apply_em']
        svm_iterative_mode = d_and_c_selection_form.cleaned_data['svm_iterative_mode']

        rn_program = 'python manage.py labelreliablenegatives_' + rn_method
        classifier_program = 'python manage.py labelunlabeleddocs_' + classifier

        # add positive data source setting
        rn_program += ' --positive-data-source=' + positive_data_source
        classifier_program += ' --positive-data-source=' + positive_data_source

        # add unlabeled data source setting
        rn_program += ' --unlabeled-data-source=' + unlabeled_data_source
        classifier_program += ' --unlabeled-data-source=' + unlabeled_data_source

        # add domain data to use as input for reliable negatives detection
        rn_program += ' --rn-domains=' + ','.join(rn_method_domains)
        classifier_program += ' --rn-domains=' + ','.join(rn_method_domains)

        # add domain data to use as input for final labeling
        classifier_program += ' --clf-domains=' + ','.join(classifier_domains)

        # add positive set extension setting
        rn_program += ' --positive-set-extension=' + positive_set_extension
        classifier_program += ' --positive-set-extension=' + positive_set_extension

        # add tokenizer setting
        rn_program += ' --tokenizer=' + tokenizer
        classifier_program += ' --tokenizer=' + tokenizer

        # add n-gram setting
        rn_program += ' --ngram-size=' + ngram_size
        classifier_program += ' --ngram-size=' + ngram_size

        # add reliable negatives setting for classifier
        classifier_program += ' --reliable-negatives=' + str(rn_method)

        # add stemming setting
        if apply_stemming:
            rn_program += ' --apply-stemming'
            classifier_program += ' --apply-stemming'

        # add sampling ratio setting
        if sampling_ratio:
            rn_program += ' --sampling-ratio=' + str(sampling_ratio)
            classifier_program += ' --sampling-ratio=' + str(sampling_ratio)

        # add noise level setting
        if noise_level:
            rn_program += ' --noise-level=' + str(noise_level)
            classifier_program += ' --noise-level=' + str(noise_level)

        # add iterations setting
        if iterations:
            rn_program += ' --iterations=' + str(iterations)
            classifier_program += ' --iterations=' + str(iterations)

        # add k-means clustering setting
        if kmeans_clustering:
            rn_program += ' --enable-kmeans-clustering'
            classifier_program += ' --enable-kmeans-clustering'

        # add expectation-maximization
        if apply_em:
            classifier_program += ' --apply-em'

        # add svm iterative mode setting
        if svm_iterative_mode:
            classifier_program += ' --iterative-mode'

        # add hand labeled data setting
        if hand_labeled_data:
            rn_program += ' --hand-labeled-data'
            classifier_program += ' --hand-labeled-data'

        # add save statistics setting
        if save_statistics:
            classifier_program += ' --save-statistics'

        # This is for filling the spaces so that the
        # wrapclassifiercall management command does not recognize
        # the options of rn_program as its own options
        # (which would result in an error since wrapclassifiercall does not
        # have any of these options)
        rn_program = rn_program.replace(' ', '#')
        pid_filename = os.path.join(settings.PIDFILE_ROOT, 'rn_program_process.pid')
        wrapped_process_call = 'python manage.py wrapclassifiercall' + \
                               ' --program-to-execute=' + rn_program + \
                               ' --pid-filename=' + pid_filename

        p = subprocess.Popen(wrapped_process_call.split())

        response['validation_status'] = 'success'
        response['rn_method'] = rn_method
        response['rn_method_domains'] = ','.join(rn_method_domains)
        response['classifier'] = classifier
        response['classifier_domains'] = ','.join(classifier_domains)
        response['classifier_program'] = classifier_program

    else:
        response['validation_status'] = 'failure'

    return HttpResponse(json.dumps(response), mimetype='application/json')


def monitor_rn_program_ajax(request):
    """
    Check whether the step 1 subprocess that identifies reliable negative
    webpages has already terminated. If it is still running, do nothing.
    If it has terminated, determine how many webpages in total have been
    identified as reliable negatives. Return the total and percentage values.
    """
    rn_method = request.GET['rn_method']
    rn_method_domains = request.GET['rn_method_domains'].split(',')
    response = {
        'classifier_program': request.GET['classifier_program']}
    rn_program_process_id = None
    rn_program_pidfile = os.path.join(settings.PIDFILE_ROOT, 'rn_program_process.pid')

    try:
        with open(rn_program_pidfile, 'rb') as pidfile:
            rn_program_process_id = int(pidfile.read().strip())
    except IOError:
        response['rn_program_status'] = 'finished'

    if rn_program_process_id:
        try:
            os.kill(rn_program_process_id, 0)
            response['rn_program_status'] = 'running'
        except OSError:
            os.remove(rn_program_pidfile)
            response['rn_program_status'] = 'finished'

            reliable_negatives = 0

            if rn_method == '1dnf':
                reliable_negatives = CrawledWebpage.objects.filter(is_1dnf_reliable_negative='N').count()
            elif rn_method == 'cr':
                reliable_negatives = CrawledWebpage.objects.filter(is_cosine_rocchio_reliable_negative='N').count()
            elif rn_method == 'nb':
                reliable_negatives = CrawledWebpage.objects.filter(is_nb_reliable_negative='N').count()
            elif rn_method == 'rocchio':
                reliable_negatives = CrawledWebpage.objects.filter(is_rocchio_reliable_negative='N').count()
            elif rn_method == 'spy':
                reliable_negatives = CrawledWebpage.objects.filter(is_spy_reliable_negative='N').count()

            total = CrawledWebpage.objects.filter(domain__name__in=rn_method_domains).exclude(is_eventpage__in=['Y', 'N']).count()

            response['reliable_negatives'] = '{} ({:.2f}%)'.format(reliable_negatives, reliable_negatives / total * 100)
            response['total'] = total

    return HttpResponse(json.dumps(response), mimetype='application/json')


def start_classifier_program_ajax(request):
    """
    Start the second subprocess that represents step 2 for the partially
    supervised classification model.
    """
    classifier_program = request.GET['classifier_program']
    classifier_program = classifier_program.replace(' ', '#')
    pid_filename = os.path.join(settings.PIDFILE_ROOT, 'classifier_program_process.pid')

    wrapped_process_call = 'python manage.py wrapclassifiercall' +\
                           ' --program-to-execute=' + classifier_program +\
                           ' --pid-filename=' + pid_filename

    p = subprocess.Popen(wrapped_process_call.split())

    return HttpResponse(json.dumps(''), mimetype='application/json')


def monitor_classifier_program_ajax(request):
    """
    Check whether the step 2 subprocess that classifies unlabeled webpages as
    either positive or negative has already termined. If it is still running,
    do nothing. If it has terminated, determine how many webpages have been
    classified as positive or negative, respectively.
    Return the total and percentage values.
    """
    classifier = request.GET['classifier']
    classifier_domains = request.GET['classifier_domains'].split(',')
    response = {}
    classifier_program_process_id = None
    classifier_program_pidfile = os.path.join(settings.PIDFILE_ROOT, 'classifier_program_process.pid')

    try:
        with open(classifier_program_pidfile, 'rb') as pidfile:
            classifier_program_process_id = int(pidfile.read().strip())
    except IOError:
        response['classifier_program_status'] = 'finished'

    if classifier_program_process_id:
        try:
            os.kill(classifier_program_process_id, 0)
            response['classifier_program_status'] = 'running'
        except OSError:
            os.remove(classifier_program_pidfile)
            response['classifier_program_status'] = 'finished'

            negatives = 0
            positives = 0

            # If classifier_domains is empty, it is equal to [u''].
            # Choose hand-labeled data instead
            if not classifier_domains[0]:
                queryset = CrawledWebpage.objects.filter(is_eventpage__in=['Y', 'N'])
            else:
                queryset = CrawledWebpage.objects.filter(domain__name__in=classifier_domains)

            if classifier == 'emnb':
                positives = queryset.filter(em_nb_tagging='P').count()
                negatives = queryset.filter(em_nb_tagging='N').count()
            elif classifier == 'svm':
                positives = queryset.filter(svm_tagging='P').count()
                negatives = queryset.filter(svm_tagging='N').count()

            total = positives + negatives

            response['positives'] = '{} ({:.2f}%)'.format(positives, positives / total * 100)
            response['negatives'] = '{} ({:.2f}%)'.format(negatives, negatives / total * 100)
            response['total'] = total

    return HttpResponse(json.dumps(response), mimetype='application/json')


def stop_rn_and_classifier_program_ajax(request):
    """Stop either of the two classifier subprocesses immediately upon request."""
    response = {}
    process_id = None
    rn_program_pidfile = os.path.join(settings.PIDFILE_ROOT, 'rn_program_process.pid')
    classifier_program_pidfile = os.path.join(settings.PIDFILE_ROOT, 'classifier_program_process.pid')

    try:
        with open(rn_program_pidfile, 'rb') as pidfile:
            process_id = int(pidfile.read().strip())
        os.remove(rn_program_pidfile)
    except IOError:
        try:
            with open(classifier_program_pidfile, 'rb') as pidfile:
                process_id = int(pidfile.read().strip())
            os.remove(classifier_program_pidfile)
        except IOError:
            response['process_status'] = 'finished'

    if process_id:
        print 'Terminating classifiers ...'
        try:
            os.kill(process_id, signal.SIGKILL)
            response['process_status'] = 'terminating'
        except OSError:
            response['process_status'] = 'finished'

    return HttpResponse(json.dumps(response), mimetype='application/json')


def process_webpage_crawler_form_ajax(request):
    """
    Process the input from the webpage crawler form and check whether
    it is valid. If so, start the web crawler subprocess which crawls webpages
    from the specified domain.
    """
    webpage_crawler_form = WebPageCrawlerForm(request.POST)
    response = {}

    if webpage_crawler_form.is_valid():
        url_to_crawl = webpage_crawler_form.cleaned_data['url_to_crawl']
        maximum_number_of_pages = webpage_crawler_form.cleaned_data['maximum_number_of_pages']
        delete_previous_pages = webpage_crawler_form.cleaned_data['delete_previous_pages']
        netloc = CommonUtils.extract_netlocs([url_to_crawl])[0]

        response['delete_previous_pages'] = delete_previous_pages
        if delete_previous_pages:
            try:
                domain = Domain.objects.get(name=netloc)
                number_previous_pages = domain.crawledwebpage_set.count()
                domain.delete()
                response['number_previous_pages'] = number_previous_pages
            except Domain.DoesNotExist:
                response['number_previous_pages'] = 0

        program = 'python manage.py crawlwebpages' + ' -n ' + str(maximum_number_of_pages) + ' ' + url_to_crawl
        p = subprocess.Popen(program.split())

        response['validation_status'] = 'success'
        response['domain'] = netloc
        response['url_to_crawl'] = url_to_crawl
        response['maximum_number_of_pages'] = maximum_number_of_pages
    else:
        response['validation_status'] = 'failure'

    return HttpResponse(json.dumps(response), mimetype='application/json')


def monitor_crawler_process_ajax(request):
    """
    Check whether the webpage crawler subprocess has already terminated.
    Return the status and the current number of crawled webpages for the given
    domain.
    """
    response = {}
    scrapy_process_id = None
    scrapy_pidfile = os.path.join(settings.PIDFILE_ROOT, 'scrapy_crawler_process.pid')

    try:
        with open(scrapy_pidfile, 'rb') as pidfile:
            scrapy_process_id = int(pidfile.read().strip())
    except IOError:
        response['crawler_status'] = 'finished'

    try:
        response['crawled_pages'] = Domain.objects.get(
            name=request.GET['domain']).crawledwebpage_set.count()
    except Domain.DoesNotExist:
        response['crawled_pages'] = 0

    if scrapy_process_id:
        try:
            os.kill(scrapy_process_id, 0)
            response['crawler_status'] = 'running'
        except OSError:
            os.remove(scrapy_pidfile)
            response['crawler_status'] = 'finished'

    return HttpResponse(json.dumps(response), mimetype='application/json')


def stop_crawler_process_ajax(request):
    """Stop the web crawler subprocess immediately upon request."""
    response = {}
    scrapy_process_id = None
    scrapy_pidfile = os.path.join(settings.PIDFILE_ROOT, 'scrapy_crawler_process.pid')

    try:
        with open(scrapy_pidfile, 'rb') as pidfile:
            scrapy_process_id = int(pidfile.read().strip())
    except IOError:
        response['crawler_status'] = 'finished'

    if scrapy_process_id:
        try:
            os.kill(scrapy_process_id, signal.SIGKILL)
            response['crawler_status'] = 'terminating'
        except OSError:
            os.remove(scrapy_pidfile)
            response['crawler_status'] = 'finished'

    return HttpResponse(json.dumps(response), mimetype='application/json')

