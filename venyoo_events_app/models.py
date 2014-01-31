# -*- coding: utf-8 -*-

"""
This module contains the database models for the venyoo_events_app.

Each model defines the essential fields, relations and behaviors of
the data that should be stored. Generally, each model maps to a
single database table.

For an overview about Django's models, see:
    https://docs.djangoproject.com/en/1.5/topics/db/models/

For a detailed explanation of field options and field types, see:
    https://docs.djangoproject.com/en/1.5/ref/models/fields/

"""


from django.db import models

from apps.auxiliary_data_app.models import CommonInformation


class Event(CommonInformation):

    """
    The Event model describes a single Venyoo.de event.

    Its fields capture information that is extracted from
    the corresponding Venyoo.de webpage.

    """

    categories = models.ManyToManyField(
        'EventCategory',
        help_text='One or more categories that this event belongs to.')

    end_date = models.DateTimeField(
        blank=True,
        null=True,
        help_text='The date at which this event ends (optional)')

    location = models.ForeignKey(
        'EventLocation',
        help_text='The location that this event takes place in')

    start_date = models.DateTimeField(
        blank=True,
        null=True,
        help_text='The date at which this event starts')

    summary = models.TextField(
        blank=True,
        help_text='An informative summary describing ' +
                  'what this event is about (optional)')

    title = models.CharField(
        max_length=100,
        help_text='The title of this event')

    url = models.URLField(
        unique=True,
        help_text='The URL on venyoo.de where the event is mentioned')

    user = models.ForeignKey(
        'VenyooUser',
        help_text='The name of the user or bot ' +
                  'who entered this event into Venyoo')

    @property
    def number_of_event_categories(self):
        """Return the number of categories for a particular event."""
        return self.categories.count()

    def __unicode__(self):
        """Return a unicode representation for an Event instance."""
        return u'{0}'.format(self.title)


class EventCategory(CommonInformation):

    """
    The EventCategory model describes a single Venyoo.de category.

    Each category is associated with one or more instances of
    the Event model. Vice versa, each Event instance is associated with
    one or more instances of the EventCategory model.

    """

    title = models.CharField(
        max_length=50,
        unique=True,
        help_text='The name of this event category')

    class Meta:

        """
        This class provides metadata options for the EventCategory model.

        """

        verbose_name_plural = 'event categories'

    @property
    def number_of_events(self):
        """Return the number of events for a particular category."""
        return self.event_set.count()

    @property
    def number_of_event_locations(self):
        """Return the number of locations for events of a particular category."""
        location_set = set()
        for event in self.event_set.prefetch_related('location'):
            location_set.add(event.location)
        return len(location_set)

    @property
    def number_of_towns(self):
        """Return the number of towns for events of a particular category."""
        town_set = set()
        for event in self.event_set.prefetch_related('location__town'):
            town_set.add(event.location.town)
        return len(town_set)

    @property
    def number_of_users(self):
        """Return the number of users who have entered events for a particular category."""
        user_set = set()
        for event in self.event_set.prefetch_related('user'):
            user_set.add(event.user)
        return len(user_set)

    def __unicode__(self):
        """Return a unicode representation for an EventCategory instance."""
        return u'{0}'.format(self.title)


class EventLocation(CommonInformation):

    """
    The EventLocation model describes a single event location.

    A valid location can be any kind of buildings or spots outside
    of buildings. Each location is associated with one or more instances
    of the Event model. Additionally, one or more location instances
    are associated with a single instance of the Town model.

    """

    name = models.CharField(
        max_length=50,
        help_text='The name of this location')

    postal_code = models.CharField(
        blank=True,
        max_length=5,
        help_text='The postal code of this location (optional)')

    street_address = models.CharField(
        blank=True,
        max_length=50,
        help_text='The street address of this location (optional)')

    summary = models.TextField(
        blank=True,
        help_text='An informative summary describing ' +
                  'what this location is about (optional)')

    town = models.ForeignKey(
        'Town',
        help_text='The town that this location is situated in')

    url = models.URLField(
        unique=True,
        help_text='The URL on venyoo.de where the location is mentioned')

    websites = models.ManyToManyField(
        'EventLocationWebsite',
        blank=True,
        null=True,
        help_text='The websites related to this event location')

    @property
    def number_of_events(self):
        """Return the number of events for a particular location."""
        return self.event_set.count()

    @property
    def number_of_event_categories(self):
        """Return the number of categories for events from a particular location."""
        category_set = set()
        for event in self.event_set.prefetch_related('categories'):
            for category in event.categories.all():
                category_set.add(category)
        return len(category_set)

    @property
    def number_of_users(self):
        """Return the number of users who have entered events for a particular location."""
        user_set = set()
        for event in self.event_set.prefetch_related('user'):
            user_set.add(event.user)
        return len(user_set)

    @property
    def number_of_websites(self):
        """Return the number of websites for a particular location."""
        return self.websites.count()

    def __unicode__(self):
        """Return a unicode representation for an EventLocation instance."""
        return u'{0}'.format(self.name)


class EventLocationWebsite(CommonInformation):

    """
    The EventLocationWebsite model describes a single URL to an official event location website.

    One or more websites are associated with one instance of the
    EventLocation model.

    """

    url = models.URLField(
        unique=True,
        help_text='The URL to the official website of a location')

    @property
    def number_of_event_locations(self):
        """Return the number of event locations referring to a particular website."""
        return self.eventlocation_set.count()

    def __unicode__(self):
        """Return a unicode representation for an EventLocationWebsite instance."""
        return u'{0}'.format(self.url)


class FederateState(CommonInformation):

    """
    The FederateState model describes a single federate state of Germany.

    Germany has sixteen federate states in total. Each state is associated
    with one or more instances of the Town model.

    """

    name = models.CharField(
        max_length=25,
        help_text="The name of this federate state of Germany")

    @property
    def number_of_events(self):
        """Return the number of events taking place in a particular state."""
        return sum(
            (location.event_set.count()
             for town in self.town_set.prefetch_related('eventlocation_set__event_set')
             for location in town.eventlocation_set.all()))

    @property
    def number_of_event_categories(self):
        """Return the number of categories for events from a particular state."""
        category_set = set()
        for town in self.town_set.prefetch_related('eventlocation_set__event_set__categories'):
            for location in town.eventlocation_set.all():
                for event in location.event_set.all():
                    for category in event.categories.all():
                        category_set.add(category)
        return len(category_set)

    @property
    def number_of_event_locations(self):
        """Return the number of locations for events from a particular state."""
        return sum(
            (town.eventlocation_set.count()
             for town in self.town_set.prefetch_related('eventlocation_set')))

    @property
    def number_of_towns(self):
        """Return the number of towns for a particular state."""
        return self.town_set.count()

    @property
    def number_of_users(self):
        """Return the number of users who have entered events for a particular state."""
        user_set = set()
        for town in self.town_set.prefetch_related('eventlocation_set__event_set__user'):
            for location in town.eventlocation_set.all():
                for event in location.event_set.all():
                    user_set.add(event.user)
        return len(user_set)

    def __unicode__(self):
        """Return a unicode representation for a FederateState instance."""
        return u'{0}'.format(self.name)


class Town(CommonInformation):

    """
    The Town model describes a single German town.

    Each town is associated with one instance of the FederateState model.

    """

    federate_state = models.ForeignKey(
        'FederateState',
        help_text='The federate state that this town belongs to')

    name = models.CharField(
        max_length=50,
        help_text='The name of this German town')

    @property
    def number_of_events(self):
        """Return the number of events for a particular town."""
        return sum(
            (location.event_set.count()
             for location in self.eventlocation_set.prefetch_related('event_set')))

    @property
    def number_of_event_categories(self):
        """Return the number of categories for events from a particular town."""
        category_set = set()
        for location in self.eventlocation_set.prefetch_related('event_set__categories'):
            for event in location.event_set.all():
                for category in event.categories.all():
                    category_set.add(category)
        return len(category_set)

    @property
    def number_of_event_locations(self):
        """Return the number of locations for events from a particular town."""
        return self.eventlocation_set.count()

    @property
    def number_of_users(self):
        """Return the number of users who have entered events for a particular town."""
        user_set = set()
        for location in self.eventlocation_set.prefetch_related('event_set__user'):
            for event in location.event_set.all():
                user_set.add(event.user)
        return len(user_set)

    def __unicode__(self):
        """Return a unicode representation for a Town instance."""
        return u'{0}'.format(self.name)


class VenyooUser(CommonInformation):

    """
    The VenyooUser model describes a single Venyoo.de user.

    Each event on Venyoo.de is entered by a user. This user can either be
    a human person who registers on the website to enter their events or
    a virtual bot which automatically scrapes the Web for event
    information. Therefore, each event is associated with one instance of
    the VenyooUser model.

    """

    name = models.CharField(
        max_length=50,
        unique=True,
        help_text='The name of this user')

    @property
    def number_of_events(self):
        """Return the number of events for a particular user."""
        return self.event_set.count()

    @property
    def number_of_event_categories(self):
        """Return the number of categories for events from a particular user."""
        category_set = set()
        for event in self.event_set.prefetch_related('categories'):
            for category in event.categories.all():
                category_set.add(category)
        return len(category_set)

    @property
    def number_of_event_locations(self):
        """Return the number of locations for events from a particular user."""
        location_set = set()
        for event in self.event_set.prefetch_related('location'):
            location_set.add(event.location)
        return len(location_set)

    @property
    def number_of_towns(self):
        """Return the number of towns for events from a paticular user."""
        town_set = set()
        for event in self.event_set.prefetch_related('location__town'):
            town_set.add(event.location.town)
        return len(town_set)

    def __unicode__(self):
        """Return a unicode representation for a VenyooUser instance."""
        return u'{0}'.format(self.name)
