# -*- coding: utf-8 -*-

"""
This module contains a custom management command that starts other
commands by writing their pid files onto the hard drive so that they can be
stopped by deleting their pid files again.

For more information about custom management commands, see:
    https://docs.djangoproject.com/en/1.5/howto/custom-management-commands/

"""


from optparse import make_option
import os.path
import subprocess

from django.conf import settings
from django.core.management.base import BaseCommand


class Command(BaseCommand):

    """
    This management command is used to start other management commands by
    writing their pid files onto the hard drive so that they can be stopped
    by deleting their pid files again.

    """

    option_list = BaseCommand.option_list + (
        make_option('--program-to-execute',
            action='store',
            dest='program_to_execute',
        ),
        make_option('--pid-filename',
            action='store',
            dest='pid_filename'
        )
    )

    def handle(self, *args, **options):
        """"""
        program_to_execute = options.get('program_to_execute')
        pid_filename = options.get('pid_filename')

        program_to_execute = program_to_execute.replace('#', ' ')
        p = subprocess.Popen(program_to_execute.split())

        print 'Writing process id to file...'
        with open(os.path.join(settings.PIDFILE_ROOT, pid_filename), 'wb') as pidfile:
            pidfile.write(str(p.pid))
        print 'File has been successfully written!'
