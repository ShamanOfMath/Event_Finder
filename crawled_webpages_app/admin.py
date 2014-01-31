# -*- coding: utf-8 -*-

"""
This module contains the configuration of this application's admin interface.

This configuration states how the model data is displayed,
sorted, and searched in the admin interface.

For a detailed explanation of the admin site configuration, see:
    https://docs.djangoproject.com/en/1.5/ref/contrib/admin/

"""


from django.contrib import admin

from apps.crawled_webpages_app.models import CrawledWebpage, Domain


class CrawledWebpageAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for CrawledWebpage model data.

    """

    readonly_fields = ('url', 'html_title', 'html_text',
                       'html_meta', 'html_headings', 'html_lists',
                       'html_tables', 'html_emphs', 'domain',
                       'is_eventpage')
    list_display = ('url', 'domain', 'is_eventpage', 'is_1dnf_reliable_negative', #'classified_as_eventpage',
                    'is_spy_reliable_negative', 'is_cosine_rocchio_reliable_negative',
                    'is_nb_reliable_negative', 'is_rocchio_reliable_negative',
                    'em_nb_tagging', 'svm_tagging',
        )
    list_editable = ('is_eventpage', 'url')
    list_display_links = ('domain',)
    list_filter = ('domain',)
    search_fields = ('url',)


class DomainAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for Domain model data.

    """

    fields = ('name',)
    list_display = ('name', 'timestamp_first_created')
    search_fields = ('name',)
    ordering = ('-timestamp_first_created',)



admin.site.register(CrawledWebpage, CrawledWebpageAdmin)
admin.site.register(Domain, DomainAdmin)
