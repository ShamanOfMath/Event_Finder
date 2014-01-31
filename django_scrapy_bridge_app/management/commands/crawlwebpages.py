# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command that enables the
user to start the Scrapy crawler from the command line using a Python subprocess
using specific settings.

It uses the GenericWebpageSpiderPipeline to crawl pages from arbitrary domains.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""

from optparse import make_option
import os.path
import subprocess
import time

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError

from apps.auxiliary_data_app.tokenizers import ClassifierFeatureTokenizer
from apps.auxiliary_data_app.utils import CommonUtils
from apps.crawled_webpages_app.models import Domain


class Command(BaseCommand):

    """


    """

    args = '<http://... http://... ...>'
    help = 'Crawls the content of one or more webpages in breadth-first order'

    option_list = BaseCommand.option_list + (
        make_option(
            '-f', '--filename',
            metavar='FILE',
            action='store',
            dest='filename',
            help='absolute path to FILE containing urls to crawl' +
                 ' (one url per line)'
        ),
        make_option(
            '-n', '--number-of-pages',
            action='store',
            type='int',
            dest='number_of_pages',
            default=100,
            help='maximum number of pages to crawl [default: %default]'
        )
    )

    def _all(self, iterable):
        """"""
        for element in iterable:
            if not element:
                return False
        if not iterable:
            return False
        return True

    def handle(self, *args, **options):
        """"""
        filename = options.get('filename')
        number_of_pages = options.get('number_of_pages')

        if number_of_pages < 1:
            raise CommandError('number of pages must be greater than or equal to 1')

        program = 'python manage.py scrapy crawl generic_spider' +\
                  ' -s LOG_FILE=logs/generic_spider.log' +\
                  ' -s DEPTH_PRIORITY=1' +\
                  ' -s SCHEDULER_DISK_QUEUE=scrapy.squeue.PickleFifoDiskQueue' +\
                  ' -s SCHEDULER_MEMORY_QUEUE=scrapy.squeue.FifoMemoryQueue' +\
                  ' -s DOWNLOAD_DELAY=5.0' +\
                  ' -s ITEM_PIPELINES=event_crawler.pipelines.GenericWebpageSpiderPipeline'

        if args and filename:
            raise CommandError(
                'urls on command line and option -f are mutually exclusive')

        if filename:
            fobj = open(filename, 'rb')
            content_list = [line.strip() for line in fobj.readlines()]
            fobj.close()

            program += ' -a args_num=' + str(len(content_list))
            url_counter = len(content_list)

            for i, line in enumerate(content_list):
                if not ClassifierFeatureTokenizer._pattern_url.match(line):
                    raise CommandError('\'{0}\' is not a valid url'.format(line))

                program += ' -a start_url{0}='.format(i) + line

            netlocs = CommonUtils.extract_netlocs(args)
            netlocs_str = '_'.join(netlocs)

            #program += ' -s JOBDIR=event_crawler/crawled_data/crawls/' + netlocs_str

        elif args:
            program += ' -a args_num=' + str(len(args))
            url_counter = len(args)

            for i, arg in enumerate(args):
                if not ClassifierFeatureTokenizer._pattern_url.match(arg):
                    raise CommandError('\'{0}\' is not a valid url'.format(arg))

                program += ' -a start_url{0}='.format(i) + arg

            netlocs = CommonUtils.extract_netlocs(args)
            netlocs_str = '_'.join(netlocs)

            #program += ' -s JOBDIR=event_crawler/crawled_data/crawls/' + netlocs_str

        else:
            raise CommandError('please provide one or more urls on the ' +
                               'command line or use option -f to open ' +
                               'a file containing urls to crawl')

        # Start the subprocess
        print '\n{0} urls detected'.format(url_counter)
        print 'Starting crawling process... (press CTRL+C to terminate)'
        p = subprocess.Popen(program.split())

        print 'Writing process id to file...'
        scrapy_process_pidfile = os.path.join(settings.PIDFILE_ROOT, 'scrapy_crawler_process.pid')
        with open(scrapy_process_pidfile, 'wb') as pidfile:
            pidfile.write(str(p.pid))
        print 'File has been successfully written!'

        # Stay in a loop until KeyboardInterrupt, then terminate the process
        # Alternatively, if the maximum number of pages specified has been
        # crawled for each domain in the input, terminate as well
        try:
            while p.poll() is None:
                time.sleep(10)
                crawled_webpages_per_domain = []
                for netloc in netlocs:
                    try:
                        crawled_page_count = Domain.objects.get(
                            name=netloc).crawledwebpage_set.count()
                        crawled_webpages_per_domain.append(crawled_page_count)
                    except Domain.DoesNotExist:
                        break

                if self._all([number >= number_of_pages for number in crawled_webpages_per_domain]):
                    print 'Stopping crawler after', number_of_pages, 'crawled webpages...'
                    break

            try:
                #p.terminate()
                p.kill()
                while p.poll() is None:
                    time.sleep(10)
            except OSError:
                pass

        except KeyboardInterrupt:
            #print '\nReceived SIGTERM, shutting down gracefully...'
            print '\nReceived SIGKILL, shutting down ...'
            #p.terminate()
            p.kill()
            while p.poll() is None:
                time.sleep(10)

        print '\nCrawling process terminated.\n'
