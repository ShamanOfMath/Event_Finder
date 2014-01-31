# -*- coding: utf-8 -*-

"""
This module contains the configuration of this application's admin interface.

This configuration states how the model data is displayed,
sorted, and searched in the admin interface.

For a detailed explanation of the admin site configuration, see:
    https://docs.djangoproject.com/en/1.5/ref/contrib/admin/

"""


from django.contrib import admin

from apps.venyoo_events_app.models import \
    Event, EventCategory, EventLocation, EventLocationWebsite,\
    FederateState, Town, VenyooUser


class EventAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for Event model data.

    """

    fields = (
        'title', ('start_date', 'end_date'), 'url',
        'location', 'categories', 'user', 'summary')

    list_display = (
        'title', 'timestamp_first_created', 'location', 'url', 'user')

    search_fields = ('title', )
    filter_horizontal = ('categories',)
    ordering = ('-timestamp_first_created',)


class EventCategoryAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for EventCategory model data.

    """

    fields = ('title',)
    list_display = ('title', 'timestamp_first_created')
    search_fields = ('title',)
    ordering = ('-timestamp_first_created',)


class EventLocationAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for EventLocation model data.

    """

    fields = ('name', 'town',
              ('street_address', 'postal_code'),
              'url', 'summary', 'websites')
    list_display = ('name', 'timestamp_first_created', 'street_address',
                    'postal_code', 'url')
    search_fields = ('name', 'street_address', 'postal_code')
    filter_horizontal = ('websites',)
    ordering = ('-timestamp_first_created',)


class EventLocationWebsiteAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for EventLocationWebsite model data.

    """

    fields = ('url',)
    list_display = ('url', 'timestamp_first_created')
    search_fields = ('url',)
    ordering = ('-timestamp_first_created',)


class FederateStateAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for FederateState model data.

    """

    fields = ('name',)
    list_display = ('name', 'timestamp_first_created')
    search_fields = ('name',)
    ordering = ('-timestamp_first_created',)


class TownAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for Town model data.

    """

    fields = ('name', 'federate_state')
    list_display = ('name', 'timestamp_first_created')
    search_fields = ('name',)
    ordering = ('-timestamp_first_created',)


class VenyooUserAdmin(admin.ModelAdmin):

    """
    This class provides admin settings for VenyooUser model data.

    """

    fields = ('name',)
    list_display = ('name', 'timestamp_first_created')
    search_fields = ('name',)
    ordering = ('-timestamp_first_created',)


admin.site.register(Event, EventAdmin)
admin.site.register(EventCategory, EventCategoryAdmin)
admin.site.register(EventLocation, EventLocationAdmin)
admin.site.register(EventLocationWebsite, EventLocationWebsiteAdmin)
admin.site.register(FederateState, FederateStateAdmin)
admin.site.register(Town, TownAdmin)
admin.site.register(VenyooUser, VenyooUserAdmin)
