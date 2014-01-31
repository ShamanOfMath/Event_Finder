# -*- coding: utf-8 -*-

"""
This module contains the configuration of this application's admin interface.

This configuration states how the model data is displayed,
sorted, and searched in the admin interface.

For a detailed explanation of the admin site configuration, see:
    https://docs.djangoproject.com/en/1.5/ref/contrib/admin/

"""


from django.contrib import admin

from apps.auxiliary_data_app.models import StopWord, LanguageIdentificationTrainingData


class StopWordAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for StopWord model data.

    """

    fields = ('stop_word', 'language')
    list_display = ('stop_word', 'timestamp_first_created',
                    'timestamp_last_modified')
    list_filter = ('language',)
    search_fields = ('stop_word',)
    ordering = ('stop_word',)


class LanguageIdentificationTrainingDataAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for LanguageIdentificationTrainingData.

    """

    readonly_fields = ('language', 'training_data')
    list_display = ('language',)
    ordering = ('language',)


admin.site.register(StopWord, StopWordAdmin)
admin.site.register(LanguageIdentificationTrainingData,
    LanguageIdentificationTrainingDataAdmin)
