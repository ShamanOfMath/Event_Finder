# -*- coding: utf-8 -*-

"""
This module contains a Django custom management command being able to
stop an arbitrary Python subprocess by deleting it process id file (pid file)
from the pid_files directory.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


import os
import signal
import time

from django.core.management.base import BaseCommand


class Command(BaseCommand):

    """


    """

    args = '<pid pid ...>'

    def handle(self, *args, **options):
        """"""

        for arg in args:
            os.kill(int(arg), signal.SIGKILL)

            while True:
                try:
                    time.sleep(5)
                    os.kill(int(arg), 0)
                except OSError:
                    break
