# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command that
allows to download the database content from a DropBox server.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


import urllib2

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):

    """
    This Django custom management command downloads the database content
    of the following apps from a DropBox server in the form of JSON fixtures:

        - auxiliary_data_app
        - crawled_webpages_app
        - venyoo_events_app

    The JSON files are put into the fixtures directories of the respective apps.

    """

    def handle(self, *args, **options):
        """"""
        print 'Opening auxiliary data fixture url ...'
        auxiliary_data_fixture = urllib2.urlopen(
            'http://dl.dropbox.com/u/44971657/auxiliary_data_app.json.zip')

        print 'Opening crawled webpages fixture url ...'
        crawled_webpages_fixture = urllib2.urlopen(
            'http://dl.dropbox.com/u/44971657/crawled_webpages_app.json.zip')

        print 'Opening venyoo events fixture url ...'
        venyoo_events_fixture = urllib2.urlopen(
            'http://dl.dropbox.com/u/44971657/venyoo_events_app.json.zip')

        print 'Downloading auxiliary data fixture into correct location ...'
        auxiliary_data_output = open('{0}/apps/auxiliary_data_app/fixtures/initial_data.json.zip'.format(settings.ROOT_PATH), 'wb')
        auxiliary_data_output.write(auxiliary_data_fixture.read())
        auxiliary_data_output.close()
        print 'Auxiliary data fixture downloaded successfully!\n'

        print 'Downloading crawled webpages fixture into correct location ...'
        crawled_webpages_output = open('{0}/apps/crawled_webpages_app/fixtures/initial_data.json.zip'.format(settings.ROOT_PATH), 'wb')
        crawled_webpages_output.write(crawled_webpages_fixture.read())
        crawled_webpages_output.close()
        print 'Crawled webpages fixture downloaded successfully!\n'

        print 'Downloading venyoo events fixture into correct location ...'
        venyoo_events_output = open('{0}/apps/venyoo_events_app/fixtures/initial_data.json.zip'.format(settings.ROOT_PATH), 'wb')
        venyoo_events_output.write(venyoo_events_fixture.read())
        venyoo_events_output.close()
        print 'Venyoo events fixture downloaded successfully!\n'

        print 'DONE! All fixtures were saved successfully!'
