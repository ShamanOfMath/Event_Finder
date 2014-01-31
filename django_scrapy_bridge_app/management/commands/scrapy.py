# -*- coding: utf-8 -*-

"""
This is a custom command for calling Scrapy from within this Django project.

Scrapy is a high-level screen scraping and Web crawling framework, used to
crawl websites and extract structured data from their pages.
For more information about Scrapy, see:
    http://scrapy.org/

With the help of this command, Scrapy can be invoked by:
    python manage.py scrapy

This simplifies the interaction between Scrapy and Django, namely to extract
data using Scrapy and saving it into a database using Django.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from __future__ import absolute_import

from django.core.management.base import BaseCommand
from scrapy.cmdline import execute


class Command(BaseCommand):

    """
    This is a custom command running Scrapy from within a Django project.

    """

    def run_from_argv(self, argv):
        """Save cmd arguments in a variable to forward them to Scrapy."""
        self._argv = argv
        self.execute()

    def handle(self, *args, **options):
        """Execute Scrapy with the forwarded cmd arguments."""
        execute(self._argv[1:])
