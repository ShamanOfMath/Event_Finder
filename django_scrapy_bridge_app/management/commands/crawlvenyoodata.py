# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command that enables the
user to start the Scrapy crawler from the command line using a Python subprocess
using specific settings.

It uses the VenyooSpiderPipeline to crawl events from Venyoo.de.

NOTE: Venyoo.de by default now blocks the crawler.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


import subprocess
import time

from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):

    """


    """

    help = 'Crawls the content of event webpages from http://venyoo.de in depth-first order'

    def handle(self, *args, **options):
        """"""

        if args:
            raise CommandError('this command does not support any non-keyword arguments')

        program = 'python manage.py scrapy crawl venyoo.de' +\
                  ' -s LOG_FILE=logs/venyoo_spider.log' +\
                  ' -s JOBDIR=event_crawler/crawled_data/crawls/venyoo.de' +\
                  ' -s DOWNLOAD_DELAY=10.0' +\
                  ' -s ITEM_PIPELINES=event_crawler.pipelines.VenyooSpiderPipeline'

        # Start the subprocess
        print '\nStarting crawling process... (press CTRL+C to terminate)'
        p = subprocess.Popen(program.split())

        # Stay in a loop until KeyboardInterrupt, then terminate the process
        try:
            while p.poll() is None:
                time.sleep(10)

            print '\nCrawling process terminated automatically due to unknown reason.\n'

        except KeyboardInterrupt:
            print '\nReceived SIGTERM, shutting down gracefully...'
            p.terminate()

            while p.poll() is None:
                time.sleep(10)

            print '\nCrawling process terminated successfully.\n'

