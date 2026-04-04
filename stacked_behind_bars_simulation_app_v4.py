# -*- coding: utf-8 -*-
"""
Stacked Behind Bars Simulation — Streamlit App
================================================
Interactive front-end for the Brazil prison-system discrete-event simulation.
The simulation engine is imported unchanged;
all parameter construction and UI logic lives exclusively in this file.

Author notes
------------
- PEP 8 compliant throughout.
- All simulation-specific logic is delegated to the imported module.
- App-side helpers are grouped at the top for readability.
"""

# ---------------------------------------------------------------------------
# STANDARD-LIBRARY IMPORTS
# ---------------------------------------------------------------------------
import copy
import io
import sys
import os

# ---------------------------------------------------------------------------
# THIRD-PARTY IMPORTS
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend required for Streamlit
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import expon, truncnorm

# ---------------------------------------------------------------------------
# PAGE CONFIG — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Stacked Behind Bars Simulation",
    layout="wide",
    page_icon="⚖️",
)

# ---------------------------------------------------------------------------
# CSS INJECTION — typography, theme overrides, section headings
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Oswald:wght@400;600&display=swap');

    /* Section title style */
    .section-title {
        font-family: 'Oswald', sans-serif;
        font-size: 1.25rem;
        font-weight: 600;
        color: #f1dfbc;
        margin-top: 1.2rem;
        margin-bottom: 0.3rem;
        letter-spacing: 0.04em;
    }

    /* Main app title */
    h1 {
        font-family: 'Oswald', sans-serif !important;
        font-weight: 600 !important;
        color: #f1dfbc !important;
    }

    /* Tab header font nudge */
    .stTabs [data-baseweb="tab"] {
        font-family: 'Oswald', sans-serif;
        font-size: 1rem;
        font-weight: 400;
    }

    /* Muted interpretive text */
    .note-text {
        font-size: 0.88rem;
        color: #b0b0b0;
        margin-bottom: 0.6rem;
    }

    /* Warning banner */
    .warn-box {
        background-color: #3a2222;
        border-left: 4px solid #fa0405;
        padding: 0.6rem 1rem;
        border-radius: 4px;
        font-size: 0.9rem;
        color: #f1dfbc;
        margin-bottom: 0.8rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# EMBEDDED SIMULATION ENGINE
# ---------------------------------------------------------------------------


# -----------------------------------------------------------------------
# IMPORTS
# -----------------------------------------------------------------------

# Core Python packages
import heapq
import random
import pandas as pd

# Numerical and statistical packages
import numpy as np
from scipy.stats import expon, truncnorm
import scipy.stats as sts  # You can keep this if you want flexibility with other distributions

# Visualization
import matplotlib.pyplot as plt

# Progress bar
from tqdm import tqdm

# -----------------------------------------------------------------------
# SIMULATION CLASSES AND FUNCTIONS
# -----------------------------------------------------------------------

### `Event()` Class


class Event:
    """
    Represents an event in the simulation, storing its execution time,
    associated function, and any arguments needed for execution.

    Attributes
    ----------
    timestamp : float
        The scheduled time at which the event should occur.
    function : callable
        The function that will be executed when the event runs.
    *args : tuple
        Positional arguments to pass to the function.
    **kwargs : dict
        Keyword arguments to pass to the function.

    Methods
    -------
    __init__(timestamp, function, *args, **kwargs)
        Initializes an event with a timestamp, function, and optional arguments.
    __lt__(other)
        Defines comparison for priority queue ordering based on timestamp.
    run(schedule)
        Executes the event by calling the assigned function with its arguments.
    """
    def __init__(self, timestamp, function, *args, **kwargs):
        """
        Initializes an event instance with the given execution time, function,
        and any additional arguments.

        Parameters
        ----------
        timestamp : float
            The scheduled time for the event.
        function : callable
            The function to execute when the event occurs.
        *args : tuple
            Additional positional arguments for the function.
        **kwargs : dict
            Additional keyword arguments for the function.
        """
        self.timestamp = timestamp
        self.function = function
        self.args = args
        self.kwargs = kwargs

    def __lt__(self, other):
        """
        Compares events based on their timestamp, allowing them to be sorted
        in a priority queue. Events with earlier timestamps are given higher priority.

        Parameters
        ----------
        other : Event
            Another event to compare against.

        Returns
        -------
        bool
            True if this event occurs earlier than the other event, False otherwise.
        """
        return self.timestamp < other.timestamp

    def run(self, schedule):
        """
        Executes the event by calling its associated function with the provided
        arguments. The schedule object is always passed as the first argument,
        allowing functions to interact with the simulation's event schedule.

        Parameters
        ----------
        schedule : Schedule
            The schedule managing all events in the simulation.
        """
        self.function(schedule, *self.args, **self.kwargs)


### `Person()` Class


class Person:
    """
    Represents an individual in the pretrial detention system with attributes
    informed by the selected crime profile. Tracks status across all system stages.
    """

    def __init__(self, crime_type, crime_profile, arrival_time, representation_type):
        """
        Initializes a person object with crime characteristics and representation info.

        Parameters
        ----------
        crime_type : str
            Selected crime type.
        crime_profile : dict
            Profile parameters for the selected crime type.
        arrival_time : float
            Time of arrest.
        representation_type : str
            'public' or 'private' defender.
        """
        self.person_id = id(self)  # Unique identifier (can be replaced with a counter if preferred)
        self.crime_type = crime_type
        self.conviction_probability = crime_profile["conviction_probability"]
        self.detention_distribution = crime_profile["detention_distribution"]
        self.service_time_dist = crime_profile["service_time_dist"]
        self.arrival_time = arrival_time
        self.representation_type = representation_type

        self.service_start_time = None
        self.service_end_time = None
        self.convicted = None

        # New attributes for the adapted flow
        self.status = 'queueing'  # Other values: 'on_trial', 'waiting_decision', 'serving_sentence', 'released'
        self.decision_wait_time = None
        self.decision_time = None
        self.sentence_time = None
        self.release_time = None

    def determine_conviction(self):
        """
        Determines if the person is convicted using their conviction probability.
        """
        self.convicted = random.random() < self.conviction_probability

    def mark_trial_start(self, current_time):
        """
        Marks the beginning of the trial stage.
        """
        self.status = 'on_trial'
        self.service_start_time = current_time

    def move_to_waiting_decision(self, decision_wait_time, current_time):
        """
        Moves the person to the post-trial decision waiting stage.

        Parameters
        ----------
        decision_wait_time : float
            Sampled waiting time until decision.
        current_time : float
            Current simulation time.
        """
        self.status = 'waiting_decision'
        self.decision_wait_time = decision_wait_time
        self.decision_time = current_time + decision_wait_time

    def set_conviction_outcome(self, convicted, sentence_time=None):
        """
        Sets the conviction outcome and updates status accordingly.

        Parameters
        ----------
        convicted : bool
            Whether the person was convicted.
        sentence_time : float or None
            Sentence length if convicted.
        """
        self.convicted = convicted
        if convicted:
            self.sentence_time = sentence_time
            self.status = 'serving_sentence'
        else:
            self.status = 'released'

    def mark_release(self, current_time):
        """
        Marks the person as released and records the release time.
        """
        self.status = 'released'
        self.release_time = current_time

### `Schedule()` Class


class Schedule:
    """
    Manages an event schedule using a priority queue. Time is represented in days.
    Supports adding events at specific times or after a delay and executing them in order.

    Attributes
    ----------
    now : float
        The time at which the last event was executed.
    priority_queue : list
        A priority queue storing scheduled events in chronological order.

    Methods
    -------
    add_event_at(timestamp, function, *args, **kwargs)
        Schedules an event to occur at a specific time.
    add_event_after(interval, function, *args, **kwargs)
        Schedules an event to occur after a specified time interval.
    next_event_time()
        Returns the timestamp of the next scheduled event.
    run_next_event()
        Executes the next event in the schedule.
    __repr__()
        Returns a string representation of the schedule.
    print_events()
        Prints details of all scheduled events for debugging.
    """
    def __init__(self):
        """
        Initializes the schedule with an empty priority queue and sets the
        current time (`now`) to zero.
        """
        self.now = 0
        self.priority_queue = []

    def add_event_at(self, timestamp, function, *args, **kwargs):
        """
        Schedules an event at a specified timestamp by adding it to the priority queue.

        Parameters
        ----------
        timestamp : float
            The time at which the event should occur.
        function : callable
            The function to execute when the event occurs.
        *args : tuple
            Additional positional arguments for the function.
        **kwargs : dict
            Additional keyword arguments for the function.
        """
        heapq.heappush(
            self.priority_queue,
            Event(timestamp, function, *args, **kwargs))

    def add_event_after(self, interval, function, *args, **kwargs):
        """
        Schedules an event to occur after a given time interval from the current time.

        Parameters
        ----------
        interval : float
            The delay after which the event should be scheduled.
        function : callable
            The function to execute when the event occurs.
        *args : tuple
            Additional positional arguments for the function.
        **kwargs : dict
            Additional keyword arguments for the function.
        """
        self.add_event_at(self.now + interval, function, *args, **kwargs)

    def next_event_time(self):
        """
        Retrieves the timestamp of the next scheduled event.

        Returns
        -------
        float
            The time at which the next event is scheduled.
        """
        return self.priority_queue[0].timestamp # if self.priority_queue else float('inf')

    def run_next_event(self):
        """
        Retrieves and executes the next event in the priority queue, updating
        the current time (`now`) to the event's timestamp.
        """
        # if self.priority_queue:
        event = heapq.heappop(self.priority_queue)
        self.now = event.timestamp
        event.run(self)

    def __repr__(self):
        """
        Returns a string representation of the schedule, displaying the current
        time and the number of pending events.

        Returns
        -------
        str
            A formatted string representing the schedule state.
        """
        return (
            f'Schedule() at time {self.now} ' +
            f'with {len(self.priority_queue)} events in the queue')

    def print_events(self):
        """
        Prints diagnostic information about all scheduled events, listing
        them in order of execution.
        """
        print(repr(self))
        for event in sorted(self.priority_queue):
            print(f'   {event.timestamp}: {event.function.__name__}')


### `Arrests()` Class


class Arrests:
    """
    Manages crime profiles for the arrest process, including arrival probabilities,
    conviction probabilities, detention lengths, and trial service time distributions.

    Allows sampling of crime type based on proportional incarceration data,
    and provides crime-specific parameters for simulation.

    Attributes
    ----------
    crime_profiles : dict
        Dictionary mapping each crime type to its arrival probability,
        conviction probability, detention distribution, and service time distribution.
    crime_types : list
        List of available crime type names.
    arrival_probs : numpy.ndarray
        Normalized array of arrival probabilities for crime types.

    Methods
    -------
    sample_crime()
        Randomly selects a crime type based on arrival probabilities and returns its parameters.
    sample_detention_length(crime_profile)
        Samples a detention length (in years) from the provided crime profile's detention distribution.
    """

    def __init__(self, crime_profiles):
        """
        Initializes the Arrests object with a given crime profile dictionary.

        Parameters
        ----------
        crime_profiles : dict
            Dictionary where each crime type maps to:
                - 'arrival_probability' : float
                - 'conviction_probability' : float
                - 'detention_distribution' : tuple of (support, probabilities) or distribution object
                - 'service_time_dist' : scipy.stats distribution object
        """
        self.crime_profiles = crime_profiles
        self.crime_types = list(crime_profiles.keys())
        self.arrival_probs = np.array([
            profile["arrival_probability"]
            for profile in crime_profiles.values()
        ])
        self.arrival_probs = self.arrival_probs / self.arrival_probs.sum()  # normalize

    def sample_crime(self):
        """
        Samples a crime type according to the arrival probabilities.

        Returns
        -------
        crime_type : str
            Selected crime type.
        crime_profile : dict
            Profile parameters for the selected crime type.
        """
        crime_type = np.random.choice(self.crime_types, p=self.arrival_probs)
        return crime_type, self.crime_profiles[crime_type]

    def sample_detention_length(self, detention_distribution):
        """
        Samples the detention length (in years) for a convicted person
        based on the detention distribution defined in the crime profile.

        Parameters
        ----------
        crime_profile : dict
            The crime profile containing the detention distribution.

        Returns
        -------
        detention_length : int
            Sampled detention duration (in years).
        """
        detention_dist = detention_distribution

        # If empirical distribution is provided as (support, probabilities)
        if isinstance(detention_dist, tuple):
            support, probabilities = detention_dist
            return np.random.choice(support, p=probabilities)
        # If it's a scipy distribution object (like truncnorm)
        else:
            return int(detention_dist.rvs())


### `Person()` Class


class Person:
    """
    Represents an individual in the pretrial detention system with attributes
    informed by the selected crime profile. Tracks status across all system stages.
    """

    def __init__(self, crime_type, crime_profile, arrival_time, representation_type):
        """
        Initializes a person object with crime characteristics and representation info.

        Parameters
        ----------
        crime_type : str
            Selected crime type.
        crime_profile : dict
            Profile parameters for the selected crime type.
        arrival_time : float
            Time of arrest.
        representation_type : str
            'public' or 'private' defender.
        """
        self.person_id = id(self)  # Unique identifier (can be replaced with a counter if preferred)
        self.crime_type = crime_type
        self.conviction_probability = crime_profile["conviction_probability"]
        self.detention_distribution = crime_profile["detention_distribution"]
        self.service_time_dist = crime_profile["service_time_dist"]
        self.arrival_time = arrival_time
        self.representation_type = representation_type

        self.service_start_time = None
        self.service_end_time = None
        self.convicted = None

        # New attributes for the adapted flow
        self.status = 'queueing'  # Other values: 'on_trial', 'waiting_decision', 'serving_sentence', 'released'
        self.decision_wait_time = None
        self.decision_time = None
        self.sentence_time = None
        self.release_time = None

    def determine_conviction(self):
        """
        Determines if the person is convicted using their conviction probability.
        """
        self.convicted = random.random() < self.conviction_probability

    def mark_trial_start(self, current_time):
        """
        Marks the beginning of the trial stage.
        """
        self.status = 'on_trial'
        self.service_start_time = current_time

    def move_to_waiting_decision(self, decision_wait_time, current_time):
        """
        Moves the person to the post-trial decision waiting stage.

        Parameters
        ----------
        decision_wait_time : float
            Sampled waiting time until decision.
        current_time : float
            Current simulation time.
        """
        self.status = 'waiting_decision'
        self.decision_wait_time = decision_wait_time
        self.decision_time = current_time + decision_wait_time

    def set_conviction_outcome(self, convicted, sentence_time=None):
        """
        Sets the conviction outcome and updates status accordingly.

        Parameters
        ----------
        convicted : bool
            Whether the person was convicted.
        sentence_time : float or None
            Sentence length if convicted.
        """
        self.convicted = convicted
        if convicted:
            self.sentence_time = sentence_time
            self.status = 'serving_sentence'
        else:
            self.status = 'released'

    def mark_release(self, current_time):
        """
        Marks the person as released and records the release time.
        """
        self.status = 'released'
        self.release_time = current_time


### `LitigationQueue()` Class 


class LitigationQueue:
    """
    Manages a FIFO queue of individuals awaiting trial (pre-trial incarceration).
    Supports multiple litigation_station (servers) and integrates post-trial
    decision waiting.

    Attributes
    ----------
    service_dist : scipy.stats.rv_continuous
        Distribution used to sample trial durations.
    queue_id : int
        Identifier for the queue.
    num_litigation_station : int
        Number of available litigation_station (parallel servers).
    is_print : bool
        Whether to print event logs.
    people_in_queue : list
        List of people currently waiting in the queue.
    busy_litigation_station : int
        Number of litigation_station currently occupied.
    times : list
        Timestamps of queue length changes.
    queue_lengths : list
        Recorded queue lengths over time.
    waiting_times : list
        Recorded waiting times.
    """

    def __init__(self, service_dist, queue_id, num_litigation_station, is_print=True):
        """
        Initializes the trial queue.

        Parameters
        ----------
        service_dist : scipy.stats.rv_continuous
            Distribution for sampling trial durations (short trial time).
        queue_id : int
            Identifier for the queue.
        num_litigation_station : int
            Number of judge servers available.
        is_print : bool, optional
            Whether to print event logs (default is True).
        """
        self.service_dist = service_dist
        self.queue_id = queue_id
        self.num_litigation_station = num_litigation_station
        self.is_print = is_print

        self.people_in_queue = []
        self.busy_litigation_station = 0

        self.times = []
        self.queue_lengths = []
        self.waiting_times = []

    def enter_queue(self, schedule, person):
        """
        Adds a person to the queue and starts trial if a judge is available.

        Parameters
        ----------
        schedule : Schedule
            The simulation schedule managing events.
        person : Person
            The person entering the queue (pre-trial).
        """
        self.people_in_queue.append(person)
        self.times.append(schedule.now)
        self.queue_lengths.append(len(self.people_in_queue))

        if self.is_print:
            print(f'📅 Month {schedule.now:.2f} 🚶‍➡️ Arrested for {person.crime_type}  '
                  f'→ Entered queue at Court {self.queue_id} (current queue length: {len(self.people_in_queue)})')

        # If litigation_station are free, start trial immediately
        self.try_start_trial(schedule)

    def try_start_trial(self, schedule):
        """Attempts to start trials for as many available litigation_station as possible."""
        while self.people_in_queue and self.busy_litigation_station < self.num_litigation_station:
            person = self.people_in_queue.pop(0)
            person.mark_trial_start(schedule.now)
            waiting_time = schedule.now - person.arrival_time
            self.waiting_times.append({
                'waiting_time': waiting_time,
                'crime_type': person.crime_type,
                'representation_type': person.representation_type
            })
            self.times.append(schedule.now)
            self.queue_lengths.append(len(self.people_in_queue))

            trial_duration = person.service_time_dist.rvs()
            self.busy_litigation_station += 1

            if self.is_print:
                print(f'📅 Month {schedule.now:.2f} ⚖️ Trial started for {person.crime_type} at Court {self.queue_id}  '
                      f'→ Judge at Court {self.queue_id} is now busy')

            # Schedule trial completion event
            schedule.add_event_after(trial_duration, self.leave_service, person)

    def leave_service(self, schedule, person):
        """
        Completes trial for a person and moves them to post-trial waiting stage.

        Parameters
        ----------
        schedule : Schedule
            The simulation schedule managing events.
        person : Person
            The person finishing trial.
        """
        self.busy_litigation_station -= 1
        trial_end_time = schedule.now
        person.service_end_time = trial_end_time

        # Sample post-trial decision wait time based on crime type + defense
        decision_wait_sampler = schedule.court_system.get_decision_wait_sampler(
            person.crime_type, person.representation_type
        )
        decision_wait_time = decision_wait_sampler()
        person.move_to_waiting_decision(decision_wait_time, trial_end_time)

        if self.is_print:
            print(f'📅 Month {trial_end_time:.2f} ⚖️ Trial completed for {person.crime_type} at Court {self.queue_id}  '
                  f'→ Moving to decision waiting (duration: {decision_wait_time:.2f} months)')
            print(f'→ Judge at Court {self.queue_id} is now available')

        # Schedule decision event
        schedule.add_event_after(decision_wait_time, schedule.court_system.handle_decision, person)

        # Try to start next person in queue (if any)
        self.try_start_trial(schedule)


### `JudicialSystem()` Class


class JudicialSystem:
    """
    Manages the pretrial detention system, including shared queues, incarceration tracking,
    post-trial decision waiting, and conviction sentence handling.

    The system models:
    - Pretrial incarceration (people in queue waiting for trial).
    - On-trial cases (people currently at trial with litigation_station).
    - Post-trial, pre-decision incarceration (awaiting conviction decision).
    - Post-decision incarceration (convicted and serving sentence).

    Attributes
    ----------
    defense_queues : list of LitigationQueue
        List of queues shared by all individuals (public and private defense).
    arrest_rate_dist : scipy.stats.rv_continuous
        Distribution for sampling inter-arrival times between arrests (in months).
    num_service_stations : int
        Number of judge servers (parallel trials).
    capacity_threshold : int
        Maximum total incarceration capacity for the prison system (for overcrowding flags).
    queue_capacity_threshold : int
        Maximum pretrial queue length before raising queue overcrowding flags.
    prob_private_defense : float
        Probability that a person has private defense representation (e.g., 0.2 = 20% private).
    is_print : bool
        Whether to print event logs during the simulation.
    schedule : Schedule or None
        The event scheduler object. Set when simulation starts.
    prison_population : list of dict
        Tracks convicted individuals serving sentences. Each entry contains:
            - 'person': the Person object.
            - 'sentence_time_remaining': months left to serve.
    awaiting_decision_population : list of Person
        Tracks individuals who are post-trial but still awaiting a conviction decision.
    prison_population_history : list of dict
        Time-series record of incarceration statistics over the simulation.
    arrests : Arrests
        The Arrests object, used for sampling crime types and accessing conviction and decision parameters.
    """

    def __init__(self, arrest_rate_dist,
                 num_queues,
                 num_service_stations,
                 capacity_threshold,
                 pre_trial_capacity_threshold,   # <- renamed here
                 prob_private_defense=0.2,
                 is_print=True):
        """
        Initializes the court system configuration and attributes.

        Parameters
        ----------
        arrest_rate_dist : scipy.stats.rv_continuous
            Distribution for inter-arrival times of arrests (in months).
        num_queues : int
            Number of shared queues (FIFO) for pretrial detention.
        num_service_stations : int
            Number of available judge servers (parallel trial capacity).
        capacity_threshold : int
            Maximum total prison population before overcrowding warning.
        pre_trial_capacity_threshold : int
            Maximum allowed pre_trial capacity (pretrial queue + on-trial + awaiting decision).
        prob_private_defense : float, optional
            Probability that an individual has private defense (default is 0.2).
        is_print : bool, optional
            Whether to print simulation logs (default is True).
        """
        self.defense_queues = [
            LitigationQueue(service_dist=None, queue_id=i, num_litigation_station=num_service_stations, is_print=is_print)
            for i in range(num_queues)
        ]
        self.num_service_stations = num_service_stations
        self.arrest_rate_dist = arrest_rate_dist
        self.capacity_threshold = capacity_threshold                                 # Total capacity (includes sentenced)
        self.pre_trial_capacity_threshold = pre_trial_capacity_threshold         # Renamed here
        self.prob_private_defense = prob_private_defense
        self.is_print = is_print
        self.schedule = None  # Will be set at simulation start

        # System dynamics tracking
        self.prison_population = []                # Serving sentence
        self.awaiting_decision_population = []     # Post-trial, awaiting decision
        self.prison_population_history = []        # Time-series incarceration tracking

        self.arrests = None  # Set externally when run_simulation initializes the system

        self.people = []  # Add this in JudicialSystem.__init__




    def choose_queue(self, queue_list):
        """
        Chooses the shortest queue from a given list of queues.

        Parameters
        ----------
        queue_list : list of PublicDefenseQueue or PrivateDefenseQueue
            The list of queues to choose from.

        Returns
        -------
        queue : PublicDefenseQueue or PrivateDefenseQueue
            The selected queue with the shortest length.
        """
        return min(queue_list, key=lambda q: len(q.people_in_queue))


    def schedule_arrests(self, schedule):
        """
        Schedules the arrival of a new person into the system, using the shared queue setup.

        Parameters
        ----------
        schedule : Schedule
            The simulation schedule managing events.
        """
        crime_type, crime_profile = self.arrests.sample_crime()

        # Determine representation type
        representation_type = "private" if random.random() < self.prob_private_defense else "public"

        person = Person(
            crime_type=crime_type,
            crime_profile=crime_profile,
            arrival_time=schedule.now,
            representation_type=representation_type
        )

        # Whenever a Person is created:
        self.people.append(person)

        # Choose the shortest queue (shared across all)
        chosen_queue = self.choose_queue(self.defense_queues)
        chosen_queue.enter_queue(schedule, person)

        # Schedule next arrival
        next_arrest_interval = self.arrest_rate_dist.rvs()
        schedule.add_event_after(next_arrest_interval, self.schedule_arrests)

    def get_decision_wait_sampler(self, crime_type, defense_type):
        """
        Returns a sampler function for the post-trial decision wait time.

        Parameters
        ----------
        crime_type : str
            The crime type of the person.
        defense_type : str
            'public' or 'private'.

        Returns
        -------
        function
            A function that returns a sampled decision wait time (in months).
        """
        # You could define these distributions in your crime_profiles or separately
        profile = self.arrests.crime_profiles[crime_type]

        if defense_type == 'public':
            return profile['public_decision_wait_dist'].rvs
        else:
            return profile['private_decision_wait_dist'].rvs

    def handle_decision(self, schedule, person):
        """
        Processes the decision event after the post-trial waiting period.

        Parameters
        ----------
        schedule : Schedule
            The simulation schedule managing events.
        person : Person
            The person receiving the decision.
        """
        # Determine conviction outcome
        person.determine_conviction()

        if person.convicted:
            # Sample sentence length (in months)
            sentence_length = self.arrests.sample_detention_length(person.detention_distribution)
            person.set_conviction_outcome(convicted=True, sentence_time=sentence_length)
            person.release_time = schedule.now + sentence_length  # For bookkeeping
            self.add_to_prison_population(person, sentence_length)

            if self.is_print:
                print(f'📅 Month {schedule.now:.2f} 🚔❌ Conviction: {person.crime_type} ({person.representation_type} defense) '
                      f'→ Sentence: {sentence_length} months.')

        else:
            person.set_conviction_outcome(convicted=False)
            person.mark_release(schedule.now)
            if self.is_print:
                print(f'📅 Month {schedule.now:.2f} ✅⚖️ Acquittal: {person.crime_type} ({person.representation_type} defense) '
                      f'→ Released from system.')


    def add_to_prison_population(self, person, sentence_length):
        """
        Adds a convicted person to the prison population tracking.

        Parameters
        ----------
        person : Person
            The convicted individual.
        sentence_length : float
            Sentence length in months.
        """
        # Add to a global list for the prison population
        if not hasattr(self, 'prison_population'):
            self.prison_population = []

        self.prison_population.append({
            'person': person,
            'sentence_time_remaining': sentence_length
        })

    def track_incarceration(self, schedule):
        """
        Updates incarceration tracking, including:
        - Pretrial (in queue)
        - On-trial (currently at trial)
        - Awaiting decision (post-trial)
        - Serving sentence (convicted)

        Raises capacity warnings:
        - pre_trial capacity: includes pretrial queue, on-trial, and awaiting decision.
        - Total capacity: pre_trial capacity + convicted serving sentence.

        Parameters
        ----------
        schedule : Schedule
            The simulation schedule managing events.
        """
        total_in_queue = 0
        total_on_trial = 0
        total_awaiting_decision = 0
        total_serving_sentence = 0
        crime_type_counts = {}

        # 1. Count pretrial queue and on-trial people
        for queue in self.defense_queues:
            # Pretrial (waiting in queue)
            total_in_queue += len(queue.people_in_queue)

            # On-trial (busy litigation_station)
            total_on_trial += queue.busy_litigation_station

        # 2. Count awaiting decision (post-trial but not yet convicted)
        awaiting_decision_population = [
            p for p in getattr(self, 'awaiting_decision_population', [])
            if p.status == 'waiting_decision'
        ]
        total_awaiting_decision = len(awaiting_decision_population)
        for person in awaiting_decision_population:
            crime = person.crime_type
            crime_type_counts[crime] = crime_type_counts.get(crime, 0) + 1

        # 3. Update and count convicted population (serving sentence)
        updated_prison_population = []
        for inmate in getattr(self, 'prison_population', []):
            inmate['sentence_time_remaining'] -= 1
            if inmate['sentence_time_remaining'] > 0:
                updated_prison_population.append(inmate)
                crime = inmate['person'].crime_type
                crime_type_counts[crime] = crime_type_counts.get(crime, 0) + 1
        self.prison_population = updated_prison_population
        total_serving_sentence = len(self.prison_population)

        # 4. Calculate totals
        pre_trial_population = total_in_queue + total_on_trial + total_awaiting_decision
        total_prison_population = pre_trial_population + total_serving_sentence

        # PRINT LOG
        if self.is_print:
            print(f'\n📊 Prison Population Report for Month {schedule.now:.2f}:')
            print(f'- Pretrial queue: {total_in_queue}')
            print(f'- On trial: {total_on_trial}')
            print(f'- Awaiting decision: {total_awaiting_decision}')
            print(f'- Serving sentence: {total_serving_sentence}')
            print(f'- pre_trial (queue + trial + decision): {pre_trial_population}')
            print(f'- Total incarcerated (incl. sentenced): {total_prison_population}\n')


        # pre_trial capacity flag
        if pre_trial_population > self.pre_trial_capacity_threshold:
            if self.is_print:
                print(f'🚩 WARNING: pre_trial Capacity OVERFLOW! Current: {pre_trial_population}, Threshold: {self.pre_trial_capacity_threshold}\n')

        # Total capacity flag
        if total_prison_population > self.capacity_threshold:
            if self.is_print:
                print(f'🚩 WARNING: Total Capacity OVERFLOW! Current: {total_prison_population}, Threshold: {self.capacity_threshold}\n')

        # RECORD POPULATION HISTORY
        self.prison_population_history.append({
            'time': schedule.now,
            'pretrial_queue': total_in_queue,
            'on_trial': total_on_trial,
            'awaiting_decision': total_awaiting_decision,
            'serving_sentence': total_serving_sentence,
            'pre_trial_total': pre_trial_population,
            'total': total_prison_population,
            'by_crime_type': crime_type_counts
        })

        # Schedule the next tracking event (next month)
        schedule.add_event_after(1, self.track_incarceration)

    def run(self, schedule):
        """
        Begins the simulation by scheduling the first arrest and setting up prison population tracking.

        Parameters
        ----------
        schedule : Schedule
            The simulation schedule managing events.
        """
        self.schedule = schedule
        # Schedule the first arrest
        schedule.add_event_after(self.arrest_rate_dist.rvs(), self.schedule_arrests)
        # Schedule daily prison population tracking
        schedule.add_event_after(1, self.track_incarceration)


### `run_simulation()` Function


def run_simulation(arrest_rate_dist, arrests,
                   num_queues,
                   num_service_stations,
                   capacity_threshold,
                   pre_trial_capacity_threshold,  # <- updated parameter name here
                   prob_private_defense=0.2,
                   run_until=12,                  # Time in months
                   is_print=True,
                   progress_bar=True):
    """
    Runs the pretrial detention simulation with shared queues, judge servers,
    decision waiting logic, and capacity tracking.

    Parameters
    ----------
    arrest_rate_dist : scipy.stats.rv_continuous
        Distribution for inter-arrival times of arrests (in months).
    arrests : Arrests
        The Arrests object managing crime type selection and profiles.
    num_queues : int
        Number of shared queues for trial (no separation between public/private).
    num_service_stations : int
        Number of trial service stations (judge servers).
    capacity_threshold : int
        Maximum total incarceration capacity (includes sentenced individuals).
    pre_trial_capacity_threshold : int
        Maximum pre_trial capacity (pretrial queue + on-trial + awaiting decision).
    prob_private_defense : float, optional
        Probability that an individual has private defense (default is 0.2).
    run_until : float, optional
        Total simulation time in months (default 12 months).
    is_print : bool, optional
        Whether to print event logs.
    progress_bar : bool, optional
        Whether to show progress bar during simulation.

    Returns
    -------
    court_system : JudicialSystem
        The court system object containing queues, incarceration tracking, and history.
    """
    # Initialize the schedule and court system
    schedule = Schedule()
    court_system = JudicialSystem(
        arrest_rate_dist=arrest_rate_dist,
        num_queues=num_queues,
        num_service_stations=num_service_stations,
        capacity_threshold=capacity_threshold,
        pre_trial_capacity_threshold=pre_trial_capacity_threshold,  # <- updated here
        prob_private_defense=prob_private_defense,
        is_print=is_print
    )

    # Set arrests object and schedule into the court system
    court_system.arrests = arrests
    schedule.court_system = court_system

    # Connect the correct service distribution to each queue (person-based, not global)
    for queue in court_system.defense_queues:
        queue.service_dist = None  # Trial service time is handled per person, not globally

    # Start the simulation
    court_system.run(schedule)

    # Event loop
    events_counter = 0
    with tqdm(desc="Simulation Progress", disable=not progress_bar) as pbar:
        while schedule.next_event_time() < run_until:
            schedule.run_next_event()
            events_counter += 1
            pbar.total = events_counter
            pbar.update(1)

    return court_system


### Test Case: Event Ordering in Schedule

def test_event_ordering():
    """
    Test Case 1:
    Verifies that events are processed in chronological order
    within the Schedule class.
    """
    print("Running test_event_ordering...")

    # Initialize schedule
    schedule = Schedule()

    # Define dummy functions
    def dummy_event(schedule, label):
        print(f"Event {label} executed at time {schedule.now}")

    # Add events in random order
    schedule.add_event_at(5, dummy_event, "B")
    schedule.add_event_at(2, dummy_event, "A")
    schedule.add_event_at(7, dummy_event, "C")

    # Capture execution order
    executed_labels = []
    original_run = Event.run

    # Monkey patch Event.run to capture execution order instead of printing
    def capture_run(self, schedule):
        executed_labels.append(self.args[0])  # label is the first arg

    Event.run = capture_run  # Replace run with capture

    # Run all events
    while schedule.priority_queue:
        schedule.run_next_event()

    # Restore original run method
    Event.run = original_run

    # Assert order is correct
    assert executed_labels == ["A", "B", "C"], f"Expected order ['A', 'B', 'C'], got {executed_labels}"
    print("✅ test_event_ordering passed!")


# -----------------------------------------------------------------------
# SIMULATION METRICS AND ANALYSIS FUNCTIONS 
# -----------------------------------------------------------------------

### `summarize_waiting_times()` Function


def summarize_waiting_times(court_system):
    """
    Computes average and maximum waiting times for trial across queues,
    broken down by representation type ('public', 'private') and crime type.

    Parameters
    ----------
    court_system : JudicialSystem
        The court system object returned by the simulation.

    Returns
    -------
    summary : dict
        Includes breakdown for 'public', 'private', 'combined', and 'by_crime_type'.
    """
    def compute_stats(times):
        """Helper to compute average and max, returns None if empty."""
        return {
            'average': np.mean(times) if times else None,
            'max': np.max(times) if times else None
        }

    # Gather waiting time entries from all queues
    waiting_data = [
        wt for queue in court_system.defense_queues
        for wt in queue.waiting_times
    ]

    # Separate by representation type
    public_times = [d['waiting_time'] for d in waiting_data if d['representation_type'] == 'public']
    private_times = [d['waiting_time'] for d in waiting_data if d['representation_type'] == 'private']
    combined_times = public_times + private_times

    # Breakdown by crime type
    crime_types = set(d['crime_type'] for d in waiting_data)
    crime_type_summary = {
        crime: compute_stats([d['waiting_time'] for d in waiting_data if d['crime_type'] == crime])
        for crime in crime_types
    }

    summary = {
        'public': compute_stats(public_times),
        'private': compute_stats(private_times),
        'combined': compute_stats(combined_times),
        'by_crime_type': crime_type_summary
    }

    return summary

### `summarize_queue_lengths()` Function


def summarize_queue_lengths(court_system):
    """
    Summarizes queue lengths over time across shared queues,
    with breakdowns by representation type and by queue.

    Parameters
    ----------
    court_system : JudicialSystem
        The court system object returned by the simulation.

    Returns
    -------
    result : dict
        Structure:
        {
            'combined': {'times': [...], 'queue_lengths': [...]},
            'by_representation': {
                'public': {'times': [...], 'queue_lengths': [...]},
                'private': {'times': [...], 'queue_lengths': [...]}
            },
            'by_queue': [{'queue_id': ..., 'times': [...], 'queue_lengths': [...]}, ...]
        }
    """
    result = {'by_queue': []}

    # Individual queue breakdown
    for queue in court_system.defense_queues:
        result['by_queue'].append({
            'queue_id': queue.queue_id,
            'times': queue.times,
            'queue_lengths': queue.queue_lengths
        })

    # Gather all unique time points across all queues
    all_times = sorted(set(
        time for queue in court_system.defense_queues
        for time in queue.times
    ))

    combined_lengths = []
    public_lengths = []
    private_lengths = []

    # Compute queue lengths at each time point
    for t in all_times:
        total_length = 0
        public_count = 0
        private_count = 0

        for queue in court_system.defense_queues:
            # Get current queue length at time t (last known value)
            past_lengths = [
                ql for time_point, ql in zip(queue.times, queue.queue_lengths)
                if time_point <= t
            ]
            queue_length = past_lengths[-1] if past_lengths else 0
            total_length += queue_length

            # Disaggregate by representation type
            public_count += sum(
                1 for person in queue.people_in_queue
                if person.representation_type == 'public'
            )
            private_count += sum(
                1 for person in queue.people_in_queue
                if person.representation_type == 'private'
            )

        combined_lengths.append(total_length)
        public_lengths.append(public_count)
        private_lengths.append(private_count)

    result['combined'] = {'times': all_times, 'queue_lengths': combined_lengths}
    result['by_representation'] = {
        'public': {'times': all_times, 'queue_lengths': public_lengths},
        'private': {'times': all_times, 'queue_lengths': private_lengths}
    }

    return result

### `summarize_incarceration()` Summary Function


def summarize_incarceration(court_system):
    """
    Summarizes incarceration dynamics over time, including:
    - pre_trial incarceration (pretrial queue + on-trial + awaiting decision)
    - Total incarcerated population (includes sentenced)
    - Breakdown by crime type

    Parameters
    ----------
    court_system : JudicialSystem
        The court system object returned by the simulation.

    Returns
    -------
    incarceration_summary : dict
        Contains pre_trial, convicted, total, and crime-type breakdowns.
    """
    times = []
    pretrial = []
    on_trial = []
    awaiting_decision = []
    convicted = []
    pre_trial_total = []
    total_population = []
    crime_type_data = {}

    for record in court_system.prison_population_history:
        times.append(record['time'])
        pretrial.append(record['pretrial_queue'])
        on_trial.append(record['on_trial'])
        awaiting_decision.append(record['awaiting_decision'])
        convicted.append(record['serving_sentence'])

        pre_trial = record['pretrial_queue'] + record['on_trial'] + record['awaiting_decision']
        total = pre_trial + record['serving_sentence']

        pre_trial_total.append(pre_trial)
        total_population.append(total)

        # Crime-type breakdown
        for crime, count in record['by_crime_type'].items():
            if crime not in crime_type_data:
                crime_type_data[crime] = {'times': [], 'population': []}
            crime_type_data[crime]['times'].append(record['time'])
            crime_type_data[crime]['population'].append(count)

        # Fill zeros for missing crime types
        for crime in crime_type_data:
            if crime not in record['by_crime_type']:
                crime_type_data[crime]['times'].append(record['time'])
                crime_type_data[crime]['population'].append(0)

    incarceration_summary = {
        'times': times,
        'pre_trial': pre_trial_total,
        'convicted': convicted,
        'total': total_population,
        'by_crime_type': crime_type_data
    }

    return incarceration_summary

### `summarize_service_times()` Function


def summarize_service_times(court_system):
    """
    Summarizes trial durations (service times) for individuals who have completed trial,
    with breakdowns by representation type and crime type.

    Parameters
    ----------
    court_system : JudicialSystem
        The court system object returned by the simulation.

    Returns
    -------
    summary : dict
        Includes overall statistics and breakdowns by representation type and crime type.
    """
    service_data = []

    # Collect service times from all queues
    for queue in court_system.defense_queues:
        service_data += [
            {
                'service_time': person.service_end_time - person.service_start_time,
                'representation_type': person.representation_type,
                'crime_type': person.crime_type
            }
            for person in getattr(queue, 'person_in_service', [])  # current
            if person.service_start_time is not None and person.service_end_time is not None
        ]
        service_data += [
            {
                'service_time': wt['waiting_time'],  # fallback in case person object isn't tracked fully
                'representation_type': wt.get('representation_type'),
                'crime_type': wt.get('crime_type')
            }
            for wt in queue.waiting_times
            if 'waiting_time' in wt  # Sometimes you store dicts, not Person objects
        ]

    def compute_stats(times):
        return {
            'average': np.mean(times) if times else None,
            'max': np.max(times) if times else None,
            'min': np.min(times) if times else None
        }

    combined_times = [d['service_time'] for d in service_data if d['service_time'] is not None]
    public_times = [d['service_time'] for d in service_data if d['representation_type'] == 'public']
    private_times = [d['service_time'] for d in service_data if d['representation_type'] == 'private']

    # Crime-type breakdown
    crime_types = set(d['crime_type'] for d in service_data if d['crime_type'] is not None)
    crime_type_summary = {
        crime: compute_stats(
            [d['service_time'] for d in service_data if d['crime_type'] == crime]
        )
        for crime in crime_types
    }

    summary = {
        'public': compute_stats(public_times),
        'private': compute_stats(private_times),
        'combined': compute_stats(combined_times),
        'by_crime_type': crime_type_summary
    }

    return summary

### `summarize_decision_wait_times()` Function


def summarize_decision_wait_times(court_system):
    """
    Summarizes post-trial decision waiting times across individuals,
    broken down by representation type and crime type.

    Parameters
    ----------
    court_system : JudicialSystem
        The court system object returned by the simulation.

    Returns
    -------
    summary : dict
        Includes overall statistics and breakdowns by representation type and crime type.
    """
    decision_data = []

    # Collect decision wait times from the awaiting decision population
    awaiting_population = getattr(court_system, 'awaiting_decision_population', [])

    decision_data += [
        {
            'decision_wait_time': person.decision_wait_time,
            'representation_type': person.representation_type,
            'crime_type': person.crime_type
        }
        for person in awaiting_population
        if person.decision_wait_time is not None
    ]

    def compute_stats(times):
        return {
            'average': np.mean(times) if times else None,
            'max': np.max(times) if times else None,
            'min': np.min(times) if times else None
        }

    combined_times = [d['decision_wait_time'] for d in decision_data]
    public_times = [d['decision_wait_time'] for d in decision_data if d['representation_type'] == 'public']
    private_times = [d['decision_wait_time'] for d in decision_data if d['representation_type'] == 'private']

    # Breakdown by crime type
    crime_types = set(d['crime_type'] for d in decision_data if d['crime_type'] is not None)
    crime_type_summary = {
        crime: compute_stats(
            [d['decision_wait_time'] for d in decision_data if d['crime_type'] == crime]
        )
        for crime in crime_types
    }

    summary = {
        'public': compute_stats(public_times),
        'private': compute_stats(private_times),
        'combined': compute_stats(combined_times),
        'by_crime_type': crime_type_summary
    }

    return summary


# -----------------------------------------------------------------------
# SIMULATION RUNNER FOR MULTIPLE EXPERIMENTS 
# -----------------------------------------------------------------------

### `run_multiple_simulations()` Function 


def run_multiple_simulations(num_trials,
    run_simulation_func,
    sim_params):
    """
    Runs the simulation multiple times and returns a list of court system objects.

    Parameters
    ----------
    num_trials : int
        Number of simulation experiments to run.
    run_simulation_func : callable
        Function that runs one simulation and returns the court_system object.
    sim_params : dict
        Parameters to pass into the run_simulation_func.

    Returns
    -------
    court_system_list : list
        List of court system objects, one for each simulation run.
    """
    court_system_list = []

    for _ in tqdm(range(num_trials), desc="Running simulations"):
        # Make a copy of sim_params and force progress_bar=False for inner call
        sim_params_no_bar = sim_params.copy()
        sim_params_no_bar['progress_bar'] = False

        court_system = run_simulation_func(**sim_params_no_bar)
        court_system_list.append(court_system)

    return court_system_list

# -----------------------------------------------------------------------
# PLOTTING FUNCTIONS FOR SIMULATION OUTPUTS 
# -----------------------------------------------------------------------

### Plots for a single simulation run

### `plot_incarceration()` Function

import matplotlib.pyplot as plt
import pandas as pd

def plot_incarceration(court_system, breakdowns_to_plot, capacity_threshold=None, pre_trial_capacity_threshold=None,
                       title="Incarceration Over Time", moving_average=True, window=12):
    """
    Plots incarceration dynamics over time for a single simulation trial,
    with optional moving average smoothing.

    Parameters
    ----------
    court_system : JudicialSystem
        The court system object returned by the simulation.
    breakdowns_to_plot : list of str
        Breakdown options to plot. Options: 'total', 'pre-trial', 'convicted', 'by_crime_type'.
    capacity_threshold : int, optional
        Prison system total capacity threshold (drawn as horizontal line).
    pre_trial_capacity_threshold : int, optional
        Pre-trial incarceration capacity threshold (drawn as horizontal line).
    title : str, optional
        Title of the plot.
    moving_average : bool, default=True
        Whether to overlay moving average curves on each line.
    window : int, default=5
        Window size for the moving average (in number of time steps).
    """

    # Global style helpers (kept local to avoid changing the rest of your project)
    BG = "#272122"
    FG = "white"
    AX = "#b0b0b0"   # light gray for axes/spines
    GRID = "#8a8a8a" # light gray grid

    # Core series colors
    C_TOTAL_POP = "#bc6666"
    C_PRETRIAL_POP = "#5e79a4"
    C_CONVICTED_POP = "#f97542"
    C_TOTAL_CAP = "#fa0405"
    C_PRETRIAL_CAP = "#f1dfbc"

    # Crime colors by order (index-based)
    CRIME_COLORS_BY_ORDER = [
        "#b5906d",  # Crimes against property
        "#f97542",  # Drug-related crimes
        "#81c784",  # Crimes against the person
        "#5c6bc0",  # Crimes against sexual dignity
        "#909090",  # Not Informed
        "#f06292",  # Firearms
        "#5e79a4",  # Other crimes
        "#ce93d8",  # Crimes against Public Peace
        "#4dd0e1",  # Crimes against Public Faith (valid hex)
        "#f6d0b0",  # Crimes against Public administration
        "#fdc533",  # Crimes by private person against public administration
        "#c4cce1",  # Traffic crimes
    ]

    def style_axes(ax):
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_color(AX)
        ax.tick_params(axis="both", colors=FG)
        ax.xaxis.label.set_color(FG)
        ax.yaxis.label.set_color(FG)
        ax.title.set_color(FG)
        ax.grid(True, color=GRID, alpha=0.25)

    def legend_outside(ax):
        leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
        if leg is not None:
            for text in leg.get_texts():
                text.set_color(FG)

    # Extract simulation history
    history = court_system.prison_population_history
    times = [entry['time'] for entry in history]

    # Helper: compute moving average safely
    def compute_ma(values):
        return pd.Series(values).rolling(window=window, min_periods=1).mean().tolist()

    # Normalize requested breakdown naming: user might still pass "pre_trial"
    breakdowns_normalized = []
    for b in breakdowns_to_plot:
        breakdowns_normalized.append("pre-trial" if b == "pre_trial" else b)
    breakdowns_to_plot = breakdowns_normalized

    # -------------------
    # Plot 1: Total / Pre-trial / Convicted
    # -------------------
    if any(item in breakdowns_to_plot for item in ['total', 'pre-trial', 'convicted']):
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(BG)
        style_axes(ax)

        if 'total' in breakdowns_to_plot:
            total_values = [entry['total'] for entry in history]
            ax.plot(times, total_values, label='Total Population', color=C_TOTAL_POP)
            if moving_average:
                ax.plot(times, compute_ma(total_values), linestyle="dotted", lw=2,
                        color=C_TOTAL_POP, label=f'Total (MA {window})')

            if capacity_threshold is not None:
                ax.axhline(capacity_threshold, color=C_TOTAL_CAP, linestyle='--', lw=2,
                           label='Total Capacity Threshold')

        if 'pre-trial' in breakdowns_to_plot:
            pretrial_values = [entry['pre_trial_total'] for entry in history]
            ax.plot(times, pretrial_values, label='Pre-trial Population', color=C_PRETRIAL_POP)
            if moving_average:
                ax.plot(times, compute_ma(pretrial_values), linestyle="dotted", lw=2,
                        color=C_PRETRIAL_POP, label=f'Pre-trial (MA {window})')

            if pre_trial_capacity_threshold is not None:
                ax.axhline(pre_trial_capacity_threshold, color=C_PRETRIAL_CAP, linestyle='--', lw=2,
                           label='Pre-trial Capacity Threshold')

        if 'convicted' in breakdowns_to_plot:
            convicted_values = [entry['serving_sentence'] for entry in history]
            ax.plot(times, convicted_values, label='Convicted Population', color=C_CONVICTED_POP)
            if moving_average:
                ax.plot(times, compute_ma(convicted_values), linestyle="dotted", lw=2,
                        color=C_CONVICTED_POP, label=f'Convicted (MA {window})')

        ax.set_title(f"{title} — Total / Pre-trial / Convicted")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Incarcerated Population")

        legend_outside(ax)
        fig.tight_layout(rect=[0, 0, 0.80, 1])  # leave room on the right for legend
        plt.show()

    # -------------------
    # Plot 2: By Crime Type
    # -------------------
    if 'by_crime_type' in breakdowns_to_plot:
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(BG)
        style_axes(ax)

        crime_types = list(court_system.arrests.crime_profiles.keys())

        for i, crime in enumerate(crime_types):
            color = CRIME_COLORS_BY_ORDER[i % len(CRIME_COLORS_BY_ORDER)]
            crime_values = [entry['by_crime_type'].get(crime, 0) for entry in history]
            label = f"{crime.replace('_', ' ').title()}"

            ax.plot(times, crime_values, label=label, color=color)
            if moving_average:
                ax.plot(times, compute_ma(crime_values), linestyle="dotted", lw=2,
                        color=color, label=f"{label} (MA {window})")

        ax.set_title(f"{title} — By Crime Type")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Incarcerated Population")

        legend_outside(ax)
        fig.tight_layout(rect=[0, 0, 0.80, 1])
        plt.show()

## ´plot_queue_lengths()` Function

    
def plot_queue_lengths(court_system, title="People Waiting for Litigation Over Time"):
    """
    Plots combining people waiting for litigation (pre-trial) queue lengths from
    a single simulation trial as scatter plot.

    Parameters
    ----------
    court_system : JudicialSystem
        The court system object returned by the simulation.
    title : str, optional
        Title of the plot.
    """
    summary = summarize_queue_lengths(court_system)
    times = summary['combined']['times']
    queue_lengths = summary['combined']['queue_lengths']

    BG = "#272122"
    FG = "white"
    AX = "#b0b0b0"
    GRID = "#8a8a8a"

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.scatter(times, queue_lengths, alpha=0.5, color=FG)

    ax.set_title(title, color=FG)
    ax.set_xlabel("Time (Months)", color=FG)
    ax.set_ylabel("People Waiting for Litigation", color=FG)

    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)

    ax.grid(True, color=GRID, alpha=0.3)

    fig.tight_layout()
    plt.show()

## `plot_time_before_sentence()` Function

def plot_time_before_sentence(court_system, title="Total Time in System Before Sentencing"):
    """
    Plots a histogram of total system time (waiting in queue for litigation +
      litigationprocess (trial) duration + decision wait time)
      for all individuals who finished their trial (regardless of conviction).

    Parameters
    ----------
    court_system : JudicialSystem
        The court system object returned by the simulation.
    title : str, optional
        Title of the plot.
    """
    total_times = []

    for person in court_system.people:
        if (person.service_start_time is not None and
            person.service_end_time is not None and
            person.decision_time is not None):

            waiting_time = person.service_start_time - person.arrival_time
            service_time = person.service_end_time - person.service_start_time
            decision_wait_time = person.decision_time - person.service_end_time

            total_system_time = waiting_time + service_time + decision_wait_time
            total_times.append(total_system_time)

    mean = np.mean(total_times) if total_times else None
    median = np.median(total_times) if total_times else None

    if total_times:
        # Theme
        BG = "#272122"
        FG = "white"
        AX = "#b0b0b0"
        GRID = "#8a8a8a"

        # Requested colors
        HIST_TICKS = "#f8413e"   # histogram bars
        STATS = "#f1dfbc"        # descriptive stats

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)

        ax.hist(
            total_times,
            bins=30,
            alpha=0.7,
            color=HIST_TICKS,
            edgecolor=BG   # edge color matches background
        )

        ax.axvline(mean, color=STATS, linestyle='--', lw=2,
                   label=f'Mean: {mean:.1f} months ({(mean*30):.0f} days)')
        ax.axvline(median, color=STATS, lw=2,
                   label=f'Median: {median:.1f} ({(median*30):.0f} days)')

        ax.set_title(title, color=FG)
        ax.set_xlabel("Total Time in System Before Sentencing (Months)", color=FG)
        ax.set_ylabel("Number of People", color=FG)

        ax.tick_params(axis="both", colors=FG)
        for spine in ax.spines.values():
            spine.set_color(AX)

        ax.grid(True, color=GRID, alpha=0.3)

        # Legend inside
        leg = ax.legend(frameon=False)
        if leg is not None:
            for text in leg.get_texts():
                text.set_color(FG)

        fig.tight_layout()
        plt.show()
    else:
        print("No people found who completed their trial stage to plot system time histogram.")

### Plots for multiple simulations

## `analyze_waiting_time_vs_queues()` Function

def analyze_waiting_time_vs_queues(
    num_queues_list,
    num_trials,
    run_simulation_func,
    sim_params_base,
    summary_start=None,
    summary_end=None,
    progress_bar=True
):
    """
    Runs simulations varying the number of queues and summarizes average waiting times
    across experiments, including 95% confidence intervals.

    Parameters
    ----------
    num_queues_list : list
        List of different queue (judge) numbers to test.
    num_trials : int
        Number of simulation experiments per configuration.
    run_simulation_func : callable
        Function that runs one simulation and returns the court_system object.
    sim_params_base : dict
        Base parameters for the simulation.
    summary_start : int or float, optional
        Start of the averaging window (in months).
    summary_end : int or float, optional
        End of the averaging window (in months).
    progress_bar : bool, optional
        Whether to display the progress bar.

    Returns
    -------
    results : dict
        Structure:
        {
            num_queues: {
                'mean': {'waiting_time': avg_waiting_time},
                'ci95': {'waiting_time': ci95_waiting_time}
            },
            ...
        }
    """
    results = {}
    iterator = tqdm(num_queues_list, desc="Varying Number of Queues (Litigations)", disable=not progress_bar)

    for num_queues in iterator:
        sim_params = sim_params_base.copy()
        sim_params['num_queues'] = num_queues
        sim_params['is_print'] = False

        court_system_list = run_multiple_simulations(
            num_trials=num_trials,
            run_simulation_func=run_simulation_func,
            sim_params=sim_params
        )

        waiting_times = []

        for trial in court_system_list:
            waiting_data = summarize_waiting_times(trial)
            combined_avg_waiting = waiting_data['combined']['average']
            if combined_avg_waiting is not None:
                waiting_times.append(combined_avg_waiting)

        if waiting_times:
            mean_waiting = np.mean(waiting_times)
            sem = sts.sem(waiting_times)
            ci95 = 1.96 * sem
        else:
            mean_waiting = None
            ci95 = None

        results[num_queues] = {
            'mean': {'waiting_time': mean_waiting},
            'ci95': {'waiting_time': ci95}
        }

    return results

import numpy as np
import scipy.stats as sts
from tqdm import tqdm

## `analyze_queues_vs_incarceration()` Function

def analyze_queues_vs_incarceration(
    num_queues_list,
    num_trials,
    run_simulation_func,
    sim_params_base,
    summary_start=None,
    summary_end=None,
    progress_bar=True
):
    """
    Runs simulations varying the number of queues and tracks average incarceration (total, pre_trial, convicted)
    including 95% confidence intervals, with safe key handling.

    Parameters
    ----------
    num_queues_list : list
        List of queue numbers to test.
    num_trials : int
        Number of simulation experiments per configuration.
    run_simulation_func : callable
        Function that runs one simulation and returns the court_system object.
    sim_params_base : dict
        Base parameters for the simulation.
    summary_start : int or float, optional
        Start of the averaging window (in months).
    summary_end : int or float, optional
        End of the averaging window (in months).
    progress_bar : bool, optional
        Whether to show the progress bar.

    Returns
    -------
    results : dict
        Structure:
        {
            num_queues: {
                'mean': {'total': avg, 'pre_trial': avg, 'convicted': avg},
                'ci95': {'total': ci, 'pre_trial': ci, 'convicted': ci}
            },
            ...
        }
    """
    results = {}
    iterator = tqdm(num_queues_list, desc="Varying Number of Queues (Litigations)", disable=not progress_bar)

    for num_queues in iterator:
        sim_params = sim_params_base.copy()
        sim_params['num_queues'] = num_queues
        sim_params['is_print'] = False
        sim_params['progress_bar'] = False  # Ensure no inner progress bars

        court_system_list = run_multiple_simulations(
            num_trials=num_trials,
            run_simulation_func=run_simulation_func,
            sim_params=sim_params
        )

        total_populations = []
        pre_trial_populations = []
        convicted_populations = []

        for trial in court_system_list:
            history = trial.prison_population_history

            if history:
                times = [entry['time'] for entry in history]
                n_times = len(times)

                start_idx = int(summary_start) if summary_start is not None else 0
                end_idx = int(summary_end) if summary_end is not None else n_times

                total_series = [entry['total'] for entry in history][start_idx:end_idx]
                pre_trial_series = [entry['pre_trial_total'] for entry in history][start_idx:end_idx]
                convicted_series = [entry['serving_sentence'] for entry in history][start_idx:end_idx]

                total_populations.append(np.mean(total_series))
                pre_trial_populations.append(np.mean(pre_trial_series))
                convicted_populations.append(np.mean(convicted_series))

        # Safe computation: check if lists are not empty
        if total_populations:
            def mean_and_ci(data):
                mean_val = np.mean(data)
                sem = sts.sem(data)
                ci95 = 1.96 * sem
                return mean_val, ci95

            total_mean, total_ci = mean_and_ci(total_populations)
            pre_trial_mean, pre_trial_ci = mean_and_ci(pre_trial_populations)
            convicted_mean, convicted_ci = mean_and_ci(convicted_populations)
        else:
            total_mean = pre_trial_mean = convicted_mean = None
            total_ci = pre_trial_ci = convicted_ci = None

        results[num_queues] = {
            'mean': {
                'total': total_mean,
                'pre_trial': pre_trial_mean,
                'convicted': convicted_mean
            },
            'ci95': {
                'total': total_ci,
                'pre_trial': pre_trial_ci,
                'convicted': convicted_ci
            }
        }

    return results

## `plot_incarceration_multiple()` Function

def plot_incarceration_multiple(
    court_system_list,
    breakdowns_to_plot=['total', 'pre-trial', 'convicted'],
    capacity_threshold=None,
    pre_trial_capacity_threshold=None,
    summary_start=None,
    summary_end=None,
    title="Incarceration Over Time (Multiple Trials)"
):
    """
    Plots incarceration dynamics over time for multiple simulation trials.
    Includes mean and 95% confidence intervals across trials, with optional transient exclusion.

    Parameters
    ----------
    court_system_list : list
        List of court system objects from multiple simulation trials.
    breakdowns_to_plot : list of str, optional
        Options: 'total', 'pre_trial', 'convicted'.
    capacity_threshold : int, optional
        Maximum prison system capacity (horizontal line).
    pre_trial_capacity_threshold : int, optional
        pre_trial incarceration capacity (horizontal line).
    summary_start : int or float, optional
        Start of the summary window (in months).
    summary_end : int or float, optional
        End of the summary window (in months).
    title : str, optional
        Title of the plot.
    """
    import numpy as np
    import scipy.stats as sts
    import matplotlib.pyplot as plt

    # Theme
    BG = "#272122"
    FG = "white"
    AX = "#b0b0b0"
    GRID = "#8a8a8a"

    # Color scheme
    C_TOTAL_POP = "#bc6666"
    C_PRETRIAL_POP = "#5e79a4"
    C_CONVICTED_POP = "#f97542"
    C_TOTAL_CAP = "#fa0405"
    C_PRETRIAL_CAP = "#f1dfbc"

    n_trials = len(court_system_list)
    n_times = len(court_system_list[0].prison_population_history)
    time_points = np.arange(n_times)

    def compute_mean_ci(data_matrix):
        """Helper to compute mean and 95% CI across trials for each time point."""
        data_array = np.array(data_matrix)
        mean_values = np.mean(data_array, axis=0)
        sem = sts.sem(data_array, axis=0)
        ci95 = 1.96 * sem
        return mean_values, ci95

    # Determine index range
    start_idx = int(summary_start) if summary_start is not None else 0
    end_idx = int(summary_end) if summary_end is not None else n_times
    time_points = time_points[start_idx:end_idx]

    fig, ax = plt.subplots(figsize=(15, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    if 'total' in breakdowns_to_plot:
        data_matrix = [
            [entry['total'] for entry in trial.prison_population_history][start_idx:end_idx]
            for trial in court_system_list
        ]
        mean_values, ci95 = compute_mean_ci(data_matrix)
        ax.errorbar(
            time_points, mean_values, yerr=ci95,
            fmt='o-', capsize=4, elinewidth=1.6, markersize=5, color=C_TOTAL_POP,
            label='Total Population'
        )

        if capacity_threshold is not None:
            ax.axhline(
                capacity_threshold, color=C_TOTAL_CAP,
                linestyle='--', lw=2,
                label='Total Capacity Threshold'
            )

    if 'pre-trial' in breakdowns_to_plot:
        data_matrix = [
            [entry['pre_trial_total'] for entry in trial.prison_population_history][start_idx:end_idx]
            for trial in court_system_list
        ]
        mean_values, ci95 = compute_mean_ci(data_matrix)
        ax.errorbar(
            time_points, mean_values, yerr=ci95,
            fmt='o-', capsize=4, elinewidth=1.6, markersize=5, color=C_PRETRIAL_POP,
            label='Pre-trial Population'
        )

        if pre_trial_capacity_threshold is not None:
            ax.axhline(
                pre_trial_capacity_threshold, color=C_PRETRIAL_CAP,
                linestyle='--', lw=2,
                label='Pre-trial Capacity Threshold'
            )

    if 'convicted' in breakdowns_to_plot:
        data_matrix = [
            [entry['serving_sentence'] for entry in trial.prison_population_history][start_idx:end_idx]
            for trial in court_system_list
        ]
        mean_values, ci95 = compute_mean_ci(data_matrix)
        ax.errorbar(
            time_points, mean_values, yerr=ci95,
            fmt='o-', capsize=4, elinewidth=1.6, markersize=5, color=C_CONVICTED_POP,
            label='Convicted Population'
        )

    ax.set_title(title, color=FG)
    ax.set_xlabel("Time (Months)", color=FG)
    ax.set_ylabel("Incarcerated Population", color=FG)

    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)

    ax.grid(True, color=GRID, alpha=0.3)

    # Legend outside
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(FG)

    fig.tight_layout(rect=[0, 0, 0.80, 1])
    plt.show()

## `plot_incarceration_by_crime_multiple()` Function

def plot_incarceration_by_crime_multiple(
    court_system_list,
    summary_start=None,
    summary_end=None,
    title="Incarceration Over Time by Crime Type (Multiple Trials)"
):
    """
    Plots incarceration dynamics over time by crime type across multiple simulation trials.
    Includes mean and 95% confidence intervals, with optional transient exclusion.

    Parameters
    ----------
    court_system_list : list
        List of court system objects from multiple simulation trials.
    summary_start : int or float, optional
        Start of the summary window (in months).
    summary_end : int or float, optional
        End of the summary window (in months).
    title : str, optional
        Title of the plot.
    """
    import numpy as np
    import scipy.stats as sts
    import matplotlib.pyplot as plt

    # Theme
    BG = "#272122"
    FG = "white"
    AX = "#b0b0b0"
    GRID = "#8a8a8a"

    # Crime colors by plotting order (index-based)
    CRIME_COLORS_BY_ORDER = [
        "#b5906d",
        "#f97542",
        "#81c784",
        "#5c6bc0",
        "#909090",
        "#f06292",
        "#5e79a4",
        "#ce93d8",
        "#4dd0e1",
        "#f6d0b0",
        "#fdc533",
        "#c4cce1",
    ]

    n_trials = len(court_system_list)
    n_times = len(court_system_list[0].prison_population_history)
    time_points = np.arange(n_times)

    def compute_mean_ci(data_matrix):
        data_array = np.array(data_matrix)
        mean_values = np.mean(data_array, axis=0)
        sem = sts.sem(data_array, axis=0)
        ci95 = 1.96 * sem
        return mean_values, ci95

    # Determine index range
    start_idx = int(summary_start) if summary_start is not None else 0
    end_idx = int(summary_end) if summary_end is not None else n_times
    time_points = time_points[start_idx:end_idx]

    crime_types = list(court_system_list[0].arrests.crime_profiles.keys())

    fig, ax = plt.subplots(figsize=(18, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    for i, crime in enumerate(crime_types):
        color = CRIME_COLORS_BY_ORDER[i % len(CRIME_COLORS_BY_ORDER)]

        data_matrix = [
            [entry['by_crime_type'].get(crime, 0) for entry in trial.prison_population_history][start_idx:end_idx]
            for trial in court_system_list
        ]

        mean_values, ci95 = compute_mean_ci(data_matrix)

        label = f"{crime.replace('_', ' ').title()}"
        ax.plot(time_points, mean_values, color=color, label=label)
        ax.fill_between(
            time_points,
            mean_values - ci95,
            mean_values + ci95,
            color=color,
            alpha=0.3
        )

    ax.set_title(title, color=FG)
    ax.set_xlabel("Time (Months)", color=FG)
    ax.set_ylabel("Incarcerated Population by Crime Type", color=FG)

    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)

    ax.grid(True, color=GRID, alpha=0.3)

    # Legend outside
    leg = ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(FG)

    fig.tight_layout(rect=[0, 0, 0.78, 1])
    plt.show()

## `plot_waiting_times_vs_queues()` Function

def plot_waiting_times_vs_queues(
    results_dict,
    title="Impact of Number of Simultaneous Litigations on Average Waiting Time"
):
    """
    Plots average waiting time against the number of litigation_station (queues) with 95% CI error bars.

    Parameters
    ----------
    results_dict : dict
        Dictionary where keys are numbers of queues and values contain:
        {
            'mean': {'waiting_time': avg_waiting_time},
            'ci95': {'waiting_time': ci95_waiting_time}
        }
        (Assumes this structure was precomputed by an analysis function.)
    title : str, optional
        Title of the plot.
    """
    import matplotlib.pyplot as plt

    # Theme
    BG = "#272122"
    FG = "white"
    AX = "#b0b0b0"
    GRID = "#8a8a8a"

    # Requested colors
    PLOT_COLOR = "#f8413e"
    STATS_COLOR = "#f8413e"

    num_queues = list(results_dict.keys())
    avg_waiting = [results_dict[q]['mean']['waiting_time'] for q in num_queues]
    ci95_waiting = [results_dict[q]['ci95']['waiting_time'] for q in num_queues]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.errorbar(
        num_queues,
        avg_waiting,
        yerr=ci95_waiting,
        fmt='o-',
        capsize=4, elinewidth=1.6, markersize=5,
        color=PLOT_COLOR,
        ecolor=STATS_COLOR,
        label="Average Waiting Time (95% CI)"
    )

    ax.set_title(title, color=FG)
    ax.set_xlabel("Number of Simultaneous Litigations Handled", color=FG)
    ax.set_ylabel("Average Waiting Time (Months)", color=FG)

    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)

    ax.grid(True, color=GRID, alpha=0.3)

    # Legend inside
    leg = ax.legend(frameon=False)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(FG)

    fig.tight_layout()
    plt.show()

## `plot_incarceration_vs_queues()` Function

def plot_incarceration_vs_queues(
    results_dict,
    capacity_threshold=None,
    pre_trial_capacity_threshold=None,
    title="Impact of Number of Litigations Handled on Incarceration Population"
):
    """
    Plots average incarceration population against the number of litigation_station (queues)
    with 95% CI error bars for total, pre_trial, and convicted populations.

    Parameters
    ----------
    results_dict : dict
        Dictionary where keys are numbers of queues and values contain:
        {
            'mean': {'total': avg, 'pre_trial': avg, 'convicted': avg},
            'ci95': {'total': ci, 'pre_trial': ci, 'convicted': ci}
        }
    capacity_threshold : int, optional
        Maximum total prison capacity (horizontal line).
    pre_trial_capacity_threshold : int, optional
        Maximum pre_trial incarceration capacity (horizontal line).
    title : str, optional
        Title of the plot.
    """
    import matplotlib.pyplot as plt

    # Theme
    BG = "#272122"
    FG = "white"
    AX = "#b0b0b0"
    GRID = "#8a8a8a"

    # Color scheme (populations + capacities)
    C_PRETRIAL_POP = "#5e79a4"
    C_CONVICTED_POP = "#f97542"
    C_TOTAL_POP = "#bc6666"
    C_TOTAL_CAP = "#fa0405"
    C_PRETRIAL_CAP = "#f1dfbc"

    num_queues = list(results_dict.keys())

    total_avg = [results_dict[q]['mean']['total'] for q in num_queues]
    total_ci = [results_dict[q]['ci95']['total'] for q in num_queues]

    # Rename pre_trial -> pre-trial (labels); keep keys for backward compatibility
    pre_trial_avg = [results_dict[q]['mean']['pre_trial'] for q in num_queues]
    pre_trial_ci = [results_dict[q]['ci95']['pre_trial'] for q in num_queues]

    convicted_avg = [results_dict[q]['mean']['convicted'] for q in num_queues]
    convicted_ci = [results_dict[q]['ci95']['convicted'] for q in num_queues]

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.errorbar(num_queues, total_avg, yerr=total_ci, fmt='o-', capsize=4, elinewidth=1.6, markersize=5,
                color=C_TOTAL_POP, label='Total Population')
    ax.errorbar(num_queues, pre_trial_avg, yerr=pre_trial_ci, fmt='o-', capsize=4, elinewidth=1.6, markersize=5,
                color=C_PRETRIAL_POP, label='Pre-trial Population')
    ax.errorbar(num_queues, convicted_avg, yerr=convicted_ci, fmt='o-', capsize=4, elinewidth=1.6, markersize=5,
                color=C_CONVICTED_POP, label='Convicted Population')

    if capacity_threshold is not None:
        ax.axhline(capacity_threshold, color=C_TOTAL_CAP, linestyle='--', lw=2,
                   label='Total Capacity Threshold')

    if pre_trial_capacity_threshold is not None:
        ax.axhline(pre_trial_capacity_threshold, color=C_PRETRIAL_CAP, linestyle='--', lw=2,
                   label='Pre-trial Capacity Threshold')

    ax.set_title(title, color=FG)
    ax.set_xlabel("Number of Litigations Handled", color=FG)
    ax.set_ylabel("Average Incarceration Population", color=FG)

    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)

    ax.grid(True, color=GRID, alpha=0.3)

    # Legend INSIDE
    leg = ax.legend(frameon=False)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(FG)

    fig.tight_layout()
    plt.show()


### `analyze_arrival_rate_vs_pretrial_ratio()` Function

def analyze_arrival_rate_vs_pretrial_ratio(
    arrest_rates_list,
    num_trials,
    run_simulation_func,
    sim_params_base,
    summary_start=None,
    summary_end=None,
    progress_bar=True
):
    """
    Analyzes how changing the arrest rate affects the proportion of pre-trial incarceration
    over total incarceration.

    Parameters
    ----------
    arrest_rates_list : list of floats
        List of arrest rates (arrests per month).
    num_trials : int
        Number of simulation experiments per arrest rate.
    run_simulation_func : callable
        Simulation function that returns a court_system object.
    sim_params_base : dict
        Base simulation parameters (excluding arrest_rate_dist).
    summary_start : int or float, optional
        Start of summary interval (to exclude transient).
    summary_end : int or float, optional
        End of summary interval.
    progress_bar : bool, optional
        Whether to display a progress bar.

    Returns
    -------
    results : dict
        Dictionary mapping arrest_rate → {'mean_ratio': ..., 'ci95': ...}.
    """
    results = {}
    iterator = tqdm(arrest_rates_list, desc="Analyzing Arrival Rates", disable=not progress_bar)

    for arrests_per_month in iterator:
        scale_param = 1.0 / arrests_per_month  # Mean inter-arrival time in months
        arrest_rate_dist = expon(scale=scale_param)

        sim_params = sim_params_base.copy()
        sim_params['arrest_rate_dist'] = arrest_rate_dist
        sim_params['is_print'] = False
        sim_params['progress_bar'] = False

        court_system_list = run_multiple_simulations(
            num_trials=num_trials,
            run_simulation_func=run_simulation_func,
            sim_params=sim_params
        )

        pretrial_ratios = []

        for trial in court_system_list:
            history = trial.prison_population_history
            if history:
                times = [entry['time'] for entry in history]
                n_times = len(times)

                start_idx = int(summary_start) if summary_start is not None else 0
                end_idx = int(summary_end) if summary_end is not None else n_times

                # Pretrial = pre_trial (queue + on-trial + awaiting decision)
                pretrial_series = [entry['pre_trial_total'] for entry in history][start_idx:end_idx]
                total_series = [entry['total'] for entry in history][start_idx:end_idx]

                # Avoid division by zero:
                ratio_series = [
                    pretrial / total if total > 0 else 0
                    for pretrial, total in zip(pretrial_series, total_series)
                ]
                pretrial_ratios.append(np.mean(ratio_series))

        if pretrial_ratios:
            mean_ratio = np.mean(pretrial_ratios)
            sem = sts.sem(pretrial_ratios)
            ci95 = 1.96 * sem
        else:
            mean_ratio = None
            ci95 = None

        results[arrests_per_month] = {
            'mean_ratio': mean_ratio,
            'ci95': ci95
        }

    return results

### `plot_arrival_rate_vs_pretrial_ratio()` Function

def plot_arrival_rate_vs_pretrial_ratio(results, title="Arrival Rate vs. Pre-Trial Detention Ratio"):
    """
    Plots the relationship between arrest rate and pre-trial detention ratio.

    Parameters
    ----------
    results : dict
        Output from analyze_arrival_rate_vs_pretrial_ratio.
    title : str, optional
        Title of the plot.
    """
    arrest_rates = list(results.keys())
    mean_ratios = [results[rate]['mean_ratio'] for rate in arrest_rates]
    ci95 = [results[rate]['ci95'] for rate in arrest_rates]

    plt.figure(figsize=(12, 8))
    plt.errorbar(arrest_rates, mean_ratios, yerr=ci95, fmt='o-', capsize=4, elinewidth=1.6, markersize=5, color='black')
    plt.title(title)
    plt.xlabel("Arrests per Month")
    plt.ylabel("Pre-Trial Detention Ratio (Pre-Trial / Total Incarceration)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# `analyze_stations_vs_incarceration()` Function

def analyze_stations_vs_incarceration(
    num_stations_list,
    num_trials,
    run_simulation_func,
    sim_params_base,
    summary_start=None,
    summary_end=None,
    progress_bar=True
):
    """
    Runs simulations varying the number of service stations per litigation queue
    and tracks average incarceration (total, pre_trial, convicted) including 95%
    confidence intervals. The number of queues is held fixed at the value defined
    in `sim_params_base`.
 
    Unlike `analyze_queues_vs_incarceration()`, which varies the number of parallel
    queues while keeping stations per queue fixed, this function varies the number
    of parallel judge servers *within* each queue (i.e., `num_service_stations`)
    while keeping the queue topology fixed. This isolates the effect of increasing
    in-queue parallelism (server capacity) on the overall incarcerated population.
 
    Parameters
    ----------
    num_stations_list : list of int
        List of service station counts to test (e.g., [1, 2, 3, ..., 20]).
    num_trials : int
        Number of simulation experiments per configuration.
    run_simulation_func : callable
        Function that runs one simulation and returns the court_system object.
    sim_params_base : dict
        Base parameters for the simulation. The key 'num_service_stations' will be
        overridden for each entry in `num_stations_list`. The key 'num_queues' is
        kept fixed throughout.
    summary_start : int or float, optional
        Start of the averaging window (in months). Used to exclude transient behaviour.
    summary_end : int or float, optional
        End of the averaging window (in months).
    progress_bar : bool, optional
        Whether to display a progress bar across station configurations.
 
    Returns
    -------
    results : dict
        Nested dictionary keyed by number of service stations:
        {
            num_stations: {
                'mean': {'total': avg, 'pre_trial': avg, 'convicted': avg},
                'ci95': {'total': ci,  'pre_trial': ci,  'convicted': ci}
            },
            ...
        }
    """
    results = {}
    iterator = tqdm(
        num_stations_list,
        desc="Varying Number of Service Stations per Queue",
        disable=not progress_bar
    )
 
    for num_stations in iterator:
        sim_params = sim_params_base.copy()
        sim_params['num_service_stations'] = num_stations
        sim_params['is_print'] = False
        sim_params['progress_bar'] = False  # Suppress inner progress bars
 
        court_system_list = run_multiple_simulations(
            num_trials=num_trials,
            run_simulation_func=run_simulation_func,
            sim_params=sim_params
        )
 
        total_populations = []
        pre_trial_populations = []
        convicted_populations = []
 
        for trial in court_system_list:
            history = trial.prison_population_history
 
            if history:
                n_times = len(history)
 
                start_idx = int(summary_start) if summary_start is not None else 0
                end_idx = int(summary_end) if summary_end is not None else n_times
 
                total_series     = [entry['total']           for entry in history][start_idx:end_idx]
                pre_trial_series = [entry['pre_trial_total'] for entry in history][start_idx:end_idx]
                convicted_series = [entry['serving_sentence'] for entry in history][start_idx:end_idx]
 
                total_populations.append(np.mean(total_series))
                pre_trial_populations.append(np.mean(pre_trial_series))
                convicted_populations.append(np.mean(convicted_series))
 
        # Safe computation: guard against empty lists
        if total_populations:
            def mean_and_ci(data):
                mean_val = np.mean(data)
                sem = sts.sem(data)
                ci95 = 1.96 * sem
                return mean_val, ci95
 
            total_mean,     total_ci     = mean_and_ci(total_populations)
            pre_trial_mean, pre_trial_ci = mean_and_ci(pre_trial_populations)
            convicted_mean, convicted_ci = mean_and_ci(convicted_populations)
        else:
            total_mean = pre_trial_mean = convicted_mean = None
            total_ci   = pre_trial_ci   = convicted_ci   = None
 
        results[num_stations] = {
            'mean': {
                'total':     total_mean,
                'pre_trial': pre_trial_mean,
                'convicted': convicted_mean
            },
            'ci95': {
                'total':     total_ci,
                'pre_trial': pre_trial_ci,
                'convicted': convicted_ci
            }
        }
 
    return results
 
 
## `analyze_waiting_time_vs_stations()` Function

def analyze_waiting_time_vs_stations(
    num_stations_list,
    num_trials,
    run_simulation_func,
    sim_params_base,
    summary_start=None,
    summary_end=None,
    progress_bar=True
):
    """
    Runs simulations varying the number of service stations per litigation queue
    and summarizes average pre-trial waiting times (arrest → trial start) across
    experiments, including 95% confidence intervals.
 
    The number of queues is held fixed at the value defined in `sim_params_base`.
    Only `num_service_stations` is varied. This mirrors `analyze_waiting_time_vs_queues()`
    but sweeps over server capacity instead of queue topology, making it possible to
    isolate the effect of adding parallel judge servers within a fixed queue structure.
 
    Parameters
    ----------
    num_stations_list : list of int
        List of service station counts to test (e.g., [1, 2, 3, ..., 20]).
    num_trials : int
        Number of simulation experiments per configuration.
    run_simulation_func : callable
        Function that runs one simulation and returns the court_system object.
    sim_params_base : dict
        Base parameters for the simulation. The key 'num_service_stations' will be
        overridden for each entry in `num_stations_list`.
    summary_start : int or float, optional
        Start of the averaging window (in months). Used to exclude transient behaviour.
    summary_end : int or float, optional
        End of the averaging window (in months).
    progress_bar : bool, optional
        Whether to display a progress bar across station configurations.
 
    Returns
    -------
    results : dict
        Nested dictionary keyed by number of service stations:
        {
            num_stations: {
                'mean': {'waiting_time': avg_waiting_time},
                'ci95': {'waiting_time': ci95_waiting_time}
            },
            ...
        }
    """
    results = {}
    iterator = tqdm(
        num_stations_list,
        desc="Varying Number of Service Stations per Queue",
        disable=not progress_bar
    )
 
    for num_stations in iterator:
        sim_params = sim_params_base.copy()
        sim_params['num_service_stations'] = num_stations
        sim_params['is_print'] = False
        sim_params['progress_bar'] = False
 
        court_system_list = run_multiple_simulations(
            num_trials=num_trials,
            run_simulation_func=run_simulation_func,
            sim_params=sim_params
        )
 
        waiting_times = []
 
        for trial in court_system_list:
            waiting_data = summarize_waiting_times(trial)
            combined_avg_waiting = waiting_data['combined']['average']
            if combined_avg_waiting is not None:
                waiting_times.append(combined_avg_waiting)
 
        if waiting_times:
            mean_waiting = np.mean(waiting_times)
            sem = sts.sem(waiting_times)
            ci95 = 1.96 * sem
        else:
            mean_waiting = None
            ci95 = None
 
        results[num_stations] = {
            'mean': {'waiting_time': mean_waiting},
            'ci95': {'waiting_time': ci95}
        }
 
    return results
 
 
## `plot_waiting_times_vs_stations()` Function

def plot_waiting_times_vs_stations(
    results_dict,
    title="Impact of Number of Service Stations per Queue on Average Waiting Time"
):
    """
    Plots average pre-trial waiting time against the number of service stations per
    litigation queue, with 95% confidence interval error bars.
 
    Parameters
    ----------
    results_dict : dict
        Output from `analyze_waiting_time_vs_stations()`. Structure:
        {
            num_stations: {
                'mean': {'waiting_time': avg_waiting_time},
                'ci95': {'waiting_time': ci95_waiting_time}
            },
            ...
        }
    title : str, optional
        Title of the plot.
    """
    # Theme
    BG = "#272122"
    FG = "white"
    AX = "#b0b0b0"
    GRID = "#8a8a8a"
 
    PLOT_COLOR  = "#f8413e"
    STATS_COLOR = "#f8413e"
 
    num_stations  = list(results_dict.keys())
    avg_waiting   = [results_dict[s]['mean']['waiting_time'] for s in num_stations]
    ci95_waiting  = [results_dict[s]['ci95']['waiting_time'] for s in num_stations]
 
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
 
    ax.errorbar(
        num_stations,
        avg_waiting,
        yerr=ci95_waiting,
        fmt='o-',
        capsize=4, elinewidth=1.6, markersize=5,
        color=PLOT_COLOR,
        ecolor=STATS_COLOR,
        label="Average Waiting Time (95% CI)"
    )
 
    ax.set_title(title, color=FG)
    ax.set_xlabel("Number of Service Stations per Queue", color=FG)
    ax.set_ylabel("Average Waiting Time (Months)", color=FG)
 
    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)
 
    ax.grid(True, color=GRID, alpha=0.3)
 
    leg = ax.legend(frameon=False)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(FG)
 
    fig.tight_layout()
    plt.show()
 
 
## `plot_incarceration_vs_stations()` Function

def plot_incarceration_vs_stations(
    results_dict,
    capacity_threshold=None,
    pre_trial_capacity_threshold=None,
    title="Impact of Number of Service Stations per Queue on Incarceration Population"
):
    """
    Plots average incarceration population against the number of service stations per
    litigation queue, with 95% confidence interval error bars for the total, pre-trial,
    and convicted sub-populations. Optionally overlays capacity threshold lines.
 
    Parameters
    ----------
    results_dict : dict
        Output from `analyze_stations_vs_incarceration()`. Structure:
        {
            num_stations: {
                'mean': {'total': avg, 'pre_trial': avg, 'convicted': avg},
                'ci95': {'total': ci,  'pre_trial': ci,  'convicted': ci}
            },
            ...
        }
    capacity_threshold : int, optional
        Maximum total prison capacity (drawn as a horizontal dashed line).
    pre_trial_capacity_threshold : int, optional
        Maximum pre-trial incarceration capacity (drawn as a horizontal dashed line).
    title : str, optional
        Title of the plot.
    """
    # Theme
    BG = "#272122"
    FG = "white"
    AX = "#b0b0b0"
    GRID = "#8a8a8a"
 
    # Color scheme — consistent with plot_incarceration_vs_queues()
    C_PRETRIAL_POP  = "#5e79a4"
    C_CONVICTED_POP = "#f97542"
    C_TOTAL_POP     = "#bc6666"
    C_TOTAL_CAP     = "#fa0405"
    C_PRETRIAL_CAP  = "#f1dfbc"
 
    num_stations = list(results_dict.keys())
 
    total_avg     = [results_dict[s]['mean']['total']     for s in num_stations]
    total_ci      = [results_dict[s]['ci95']['total']     for s in num_stations]
    pre_trial_avg = [results_dict[s]['mean']['pre_trial'] for s in num_stations]
    pre_trial_ci  = [results_dict[s]['ci95']['pre_trial'] for s in num_stations]
    convicted_avg = [results_dict[s]['mean']['convicted'] for s in num_stations]
    convicted_ci  = [results_dict[s]['ci95']['convicted'] for s in num_stations]
 
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
 
    ax.errorbar(num_stations, total_avg,     yerr=total_ci,     fmt='o-', capsize=4, elinewidth=1.6, markersize=5,
                color=C_TOTAL_POP,     label='Total Population')
    ax.errorbar(num_stations, pre_trial_avg, yerr=pre_trial_ci, fmt='o-', capsize=4, elinewidth=1.6, markersize=5,
                color=C_PRETRIAL_POP,  label='Pre-trial Population')
    ax.errorbar(num_stations, convicted_avg, yerr=convicted_ci, fmt='o-', capsize=4, elinewidth=1.6, markersize=5,
                color=C_CONVICTED_POP, label='Convicted Population')
 
    if capacity_threshold is not None:
        ax.axhline(capacity_threshold, color=C_TOTAL_CAP, linestyle='--', lw=2,
                   label='Total Capacity Threshold')
 
    if pre_trial_capacity_threshold is not None:
        ax.axhline(pre_trial_capacity_threshold, color=C_PRETRIAL_CAP, linestyle='--', lw=2,
                   label='Pre-trial Capacity Threshold')
 
    ax.set_title(title, color=FG)
    ax.set_xlabel("Number of Service Stations per Queue", color=FG)
    ax.set_ylabel("Average Incarceration Population", color=FG)
 
    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)
 
    ax.grid(True, color=GRID, alpha=0.3)
 
    leg = ax.legend(frameon=False)
    if leg is not None:
        for text in leg.get_texts():
            text.set_color(FG)
 
    fig.tight_layout()
    plt.show()

def create_truncnorm(mean, std, lower, upper):
    """Creates a truncated normal distribution."""
    a, b = (lower - mean) / std, (upper - mean) / std
    return truncnorm(a=a, b=b, loc=mean, scale=std)

from types import SimpleNamespace

def compute_mean_service_time(crime_profiles):
    """
    Computes the weighted average service time (S) across all crime types based on their arrival probabilities and service time distributions.
    """
    S = 0
    for crime, profile in crime_profiles.items():
        p_c = profile['arrival_probability']
        dist = profile['service_time_dist']
        # Check if it's a truncnorm object (your create_truncnorm() outputs this)
        mean_service_time = dist.mean()
        S += p_c * mean_service_time
    return S

sim = SimpleNamespace(
    Arrests=Arrests,
    Event=Event,
    Person=Person,
    Schedule=Schedule,
    LitigationQueue=LitigationQueue,
    JudicialSystem=JudicialSystem,

    run_simulation=run_simulation,
    run_multiple_simulations=run_multiple_simulations,

    summarize_waiting_times=summarize_waiting_times,
    summarize_queue_lengths=summarize_queue_lengths,
    summarize_incarceration=summarize_incarceration,
    summarize_service_times=summarize_service_times,
    summarize_decision_wait_times=summarize_decision_wait_times,

    plot_incarceration=plot_incarceration,
    plot_queue_lengths=plot_queue_lengths,
    plot_time_before_sentence=plot_time_before_sentence,
    plot_incarceration_multiple=plot_incarceration_multiple,
    plot_incarceration_by_crime_multiple=plot_incarceration_by_crime_multiple,

    analyze_waiting_time_vs_queues=analyze_waiting_time_vs_queues,
    analyze_queues_vs_incarceration=analyze_queues_vs_incarceration,
    analyze_arrival_rate_vs_pretrial_ratio=analyze_arrival_rate_vs_pretrial_ratio,
    analyze_stations_vs_incarceration=analyze_stations_vs_incarceration,
    analyze_waiting_time_vs_stations=analyze_waiting_time_vs_stations,

    plot_waiting_times_vs_queues=plot_waiting_times_vs_queues,
    plot_incarceration_vs_queues=plot_incarceration_vs_queues,
    plot_arrival_rate_vs_pretrial_ratio=plot_arrival_rate_vs_pretrial_ratio,
    plot_waiting_times_vs_stations=plot_waiting_times_vs_stations,
    plot_incarceration_vs_stations=plot_incarceration_vs_stations,

    create_truncnorm=create_truncnorm,
    compute_mean_service_time=compute_mean_service_time,
)


# ---------------------------------------------------------------------------
# THEME CONSTANTS  (mirrors the simulation file's palette)
# ---------------------------------------------------------------------------
BG             = "#272122"
FG             = "white"
AX             = "#b0b0b0"
GRID           = "#8a8a8a"
C_TOTAL_POP    = "#bc6666"
C_PRETRIAL_POP = "#5e79a4"
C_CONVICTED    = "#f97542"
C_TOTAL_CAP    = "#fa0405"
C_PRETRIAL_CAP = "#f1dfbc"

# ---------------------------------------------------------------------------
# DEFAULT NATIONAL SENTENCE COUNTS  (Scenario 2 / SENAPPEN 2024)
# ---------------------------------------------------------------------------
DEFAULT_SENTENCE_COUNTS = {
    "not_informed": 48801,
    "0_6mo":        26863,
    "7_12mo":        1089,
    "13mo_2yr":      2497,
    "3_4yr":         7154,
    "5_8yr":        31821,
    "9_15yr":       41933,
    "16_20yr":      23321,
    "21_30yr":      21717,
    "31_50yr":      12212,
    "51_100yr":      4006,
    "gt100":          615,
}

# Human-readable labels for sentence brackets
SENTENCE_LABELS = {
    "not_informed": "Not Informed",
    "0_6mo":        "0 – 6 Months",
    "7_12mo":       "7 – 12 Months",
    "13mo_2yr":     "13 Months – 2 Years",
    "3_4yr":        "3 – 4 Years",
    "5_8yr":        "5 – 8 Years",
    "9_15yr":       "9 – 15 Years",
    "16_20yr":      "16 – 20 Years",
    "21_30yr":      "21 – 30 Years",
    "31_50yr":      "31 – 50 Years",
    "51_100yr":     "51 – 100 Years",
    "gt100":        "More Than 100 Years",
}

# Column name mapping: CSV sentence columns → internal keys
SENTENCE_CSV_COL_MAP = {
    "count per sentence time - not informed": "not_informed",
    "count per sentence time - 0_6mo":        "0_6mo",
    "count per sentence time - 7_12mo":       "7_12mo",
    "count per sentence time - 13mo_2yr":     "13mo_2yr",
    "count per sentence time - 3_4yr":        "3_4yr",
    "count per sentence time - 5_8yr":        "5_8yr",
    "count per sentence time - 9_15yr":       "9_15yr",
    "count per sentence time - 16_20yr":      "16_20yr",
    "count per sentence time - 21_30yr":      "21_30yr",
    "count per sentence time - 31_50yr":      "31_50yr",
    "count per sentence time - 51_100yr":     "51_100yr",
    "count per sentence time - gt100":        "gt100",
}

# ---------------------------------------------------------------------------
# DEFAULT CRIME-PROFILE PARAMETERS  (Scenario 2 values, used as app-side
# source of truth so we can reconstruct truncnorm dists from plain numbers)
# ---------------------------------------------------------------------------
DEFAULT_CRIME_PARAMS = {
    "against_property": {
        "display_name":       "Against Property",
        "arrival_probability": 300109 / 845418,
        "conviction_probability": 0.60,
        "service_time":       {"mean": 0.7, "std": 0.2, "lower": 0.1, "upper": 1.5},
        "public_decision":    {"mean": 5.0, "std": 1.5, "lower": 1.0, "upper": 10.0},
        "private_decision":   {"mean": 2.0, "std": 0.5, "lower": 0.5, "upper":  5.0},
    },
    "drug_related": {
        "display_name":       "Drug-Related",
        "arrival_probability": 232408 / 845418,
        "conviction_probability": 0.75,
        "service_time":       {"mean": 1.0, "std": 0.3, "lower": 0.1, "upper": 2.0},
        "public_decision":    {"mean": 6.0, "std": 2.0, "lower": 1.0, "upper": 12.0},
        "private_decision":   {"mean": 3.0, "std": 1.0, "lower": 0.5, "upper":  6.0},
    },
    "against_the_person": {
        "display_name":       "Against the Person",
        "arrival_probability": 134567 / 845418,
        "conviction_probability": 0.90,
        "service_time":       {"mean": 1.5, "std": 0.5, "lower": 0.3, "upper": 3.0},
        "public_decision":    {"mean": 8.0, "std": 2.0, "lower": 2.0, "upper": 15.0},
        "private_decision":   {"mean": 4.0, "std": 1.0, "lower": 1.0, "upper":  8.0},
    },
    "against_sexual_dignity": {
        "display_name":       "Against Sexual Dignity",
        "arrival_probability": 52585 / 845418,
        "conviction_probability": 0.90,
        "service_time":       {"mean": 1.5, "std": 0.5, "lower": 0.3, "upper": 3.0},
        "public_decision":    {"mean": 8.0, "std": 2.0, "lower": 2.0, "upper": 15.0},
        "private_decision":   {"mean": 4.0, "std": 1.0, "lower": 1.0, "upper":  8.0},
    },
    "not_informed": {
        "display_name":       "Not Informed",
        "arrival_probability": 42001 / 845418,
        "conviction_probability": 0.50,
        "service_time":       {"mean": 1.0, "std": 0.3, "lower": 0.1, "upper": 2.0},
        "public_decision":    {"mean": 6.0, "std": 2.0, "lower": 1.0, "upper": 12.0},
        "private_decision":   {"mean": 3.0, "std": 1.0, "lower": 0.5, "upper":  6.0},
    },
    "firearms": {
        "display_name":       "Firearms",
        "arrival_probability": 41702 / 845418,
        "conviction_probability": 0.80,
        "service_time":       {"mean": 1.0, "std": 0.3, "lower": 0.1, "upper": 2.0},
        "public_decision":    {"mean": 6.0, "std": 2.0, "lower": 1.0, "upper": 12.0},
        "private_decision":   {"mean": 3.0, "std": 1.0, "lower": 0.5, "upper":  6.0},
    },
    "other": {
        "display_name":       "Other",
        "arrival_probability": 18447 / 845418,
        "conviction_probability": 0.60,
        "service_time":       {"mean": 1.0, "std": 0.3, "lower": 0.1, "upper": 2.0},
        "public_decision":    {"mean": 6.0, "std": 2.0, "lower": 1.0, "upper": 12.0},
        "private_decision":   {"mean": 3.0, "std": 1.0, "lower": 0.5, "upper":  6.0},
    },
    "against_public_peace": {
        "display_name":       "Against Public Peace",
        "arrival_probability": 14958 / 845418,
        "conviction_probability": 0.70,
        "service_time":       {"mean": 1.0, "std": 0.3, "lower": 0.1, "upper": 2.0},
        "public_decision":    {"mean": 6.0, "std": 2.0, "lower": 1.0, "upper": 12.0},
        "private_decision":   {"mean": 3.0, "std": 1.0, "lower": 0.5, "upper":  6.0},
    },
    "against_public_faith": {
        "display_name":       "Against Public Faith",
        "arrival_probability": 4743 / 845418,
        "conviction_probability": 0.50,
        "service_time":       {"mean": 1.0, "std": 0.3, "lower": 0.1, "upper": 2.0},
        "public_decision":    {"mean": 6.0, "std": 2.0, "lower": 1.0, "upper": 12.0},
        "private_decision":   {"mean": 3.0, "std": 1.0, "lower": 0.5, "upper":  6.0},
    },
    "against_public_admin": {
        "display_name":       "Against Public Administration",
        "arrival_probability": 2052 / 845418,
        "conviction_probability": 0.70,
        "service_time":       {"mean": 1.0, "std": 0.3, "lower": 0.1, "upper": 2.0},
        "public_decision":    {"mean": 6.0, "std": 2.0, "lower": 1.0, "upper": 12.0},
        "private_decision":   {"mean": 3.0, "std": 1.0, "lower": 0.5, "upper":  6.0},
    },
    "by_private_person_against_public_admin": {
        "display_name":       "By Private Person Against Public Admin",
        "arrival_probability": 1368 / 845418,
        "conviction_probability": 0.70,
        "service_time":       {"mean": 1.0, "std": 0.3, "lower": 0.1, "upper": 2.0},
        "public_decision":    {"mean": 6.0, "std": 2.0, "lower": 1.0, "upper": 12.0},
        "private_decision":   {"mean": 3.0, "std": 1.0, "lower": 0.5, "upper":  6.0},
    },
    "traffic": {
        "display_name":       "Traffic",
        "arrival_probability": 478 / 845418,
        "conviction_probability": 0.60,
        "service_time":       {"mean": 0.7, "std": 0.2, "lower": 0.1, "upper": 1.5},
        "public_decision":    {"mean": 5.0, "std": 1.5, "lower": 1.0, "upper": 10.0},
        "private_decision":   {"mean": 2.0, "std": 0.5, "lower": 0.5, "upper":  5.0},
    },
}

# CSV arrival-probability column name → crime-profile key mapping
ARRIVAL_CSV_COL_MAP = {
    "arrival probability - Against Property":                       "against_property",
    "arrival probability - Drug-related":                           "drug_related",
    "arrival probability - Against the person":                     "against_the_person",
    "arrival probability - Against sexual dignity":                 "against_sexual_dignity",
    "arrival probability - Not informed":                           "not_informed",
    "arrival probability - Firearms":                               "firearms",
    "arrival probability - Other crimes":                           "other",
    "arrival probability - Against public peace":                   "against_public_peace",
    "arrival probability - Against public faith":                   "against_public_faith",
    "arrival probability - Against public administration":          "against_public_admin",
    "arrival probability - By private person against public administration": "by_private_person_against_public_admin",
    "arrival probability - Traffic":                                "traffic",
}

# ---------------------------------------------------------------------------
# HELPER: COMARCA CSV LOADER
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Loading comarca dataset…")
def load_comarca_csv(path: str) -> pd.DataFrame:
    """
    Loads and lightly cleans the comarca-level parameter CSV.

    Parameters
    ----------
    path : str
        File-system path to the CSV.

    Returns
    -------
    df : pd.DataFrame
        Cleaned dataframe with comarca data, or an empty dataframe if loading fails.
    """
    try:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        return df
    except Exception as exc:
        st.warning(f"Could not load comarca dataset: {exc}")
        return pd.DataFrame()


def get_comarca_names(df: pd.DataFrame) -> list:
    """
    Returns a sorted list of comarca names from the dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Comarca parameter dataframe.

    Returns
    -------
    list of str
        Comarca display names, or an empty list if the column is missing.
    """
    for col in ("nome_comarca", "Nome_comarca", "comarca", "name"):
        if col in df.columns:
            return sorted(df[col].dropna().astype(str).unique().tolist())
    return []


def get_comarca_row(df: pd.DataFrame, comarca_name: str) -> pd.Series:
    """
    Returns the DataFrame row corresponding to the selected comarca name.

    Parameters
    ----------
    df : pd.DataFrame
        Comarca parameter dataframe.
    comarca_name : str
        Name of the selected comarca.

    Returns
    -------
    pd.Series
        The matching row, or an empty Series if not found.
    """
    for col in ("nome_comarca", "Nome_comarca", "comarca", "name"):
        if col in df.columns:
            matches = df[df[col].astype(str) == comarca_name]
            if not matches.empty:
                return matches.iloc[0]
    return pd.Series(dtype=float)


def safe_float(value, fallback: float) -> float:
    """
    Safely converts a value to float, returning a fallback on failure.

    Parameters
    ----------
    value : any
        The value to convert.
    fallback : float
        Value returned when conversion fails or value is NaN.

    Returns
    -------
    float
    """
    try:
        result = float(value)
        return fallback if (np.isnan(result) or np.isinf(result)) else result
    except (TypeError, ValueError):
        return fallback

# -----------------------------------------------------------------------------
# COMARCA PARAMETERS .CSV
# -----------------------------------------------------------------------------
from io import StringIO

EMBEDDED_COMARCA_CSV = """
cod_mun_ibge_comarca,nome_comarca,count per sentence time - not informed,count per sentence time - 0_6mo,count per sentence time - 7_12mo,count per sentence time - 13mo_2yr,count per sentence time - 3_4yr,count per sentence time - 5_8yr,count per sentence time - 9_15yr,count per sentence time - 16_20yr,count per sentence time - 21_30yr,count per sentence time - 31_50yr,count per sentence time - 51_100yr,count per sentence time - gt100,new_admissions_per_semester,total_capacity_threshold,arrests_per_month,pre_trial_capacity_threshold,count per crime group - Against Property,count per crime group - Drug-related,count per crime group - Against the person,count per crime group - Against sexual dignity,count per crime group - Not informed,count per crime group - Firearms,count per crime group - Other crimes,count per crime group - Against public peace,count per crime group - Against public faith,count per crime group - Against public administration,count per crime group - By private person against public administration,count per crime group - Traffic,total across all crime groups,arrival probability - Against Property,arrival probability - Drug-related,arrival probability - Against the person,arrival probability - Against sexual dignity,arrival probability - Not informed,arrival probability - Firearms,arrival probability - Other crimes,arrival probability - Against public peace,arrival probability - Against public faith,arrival probability - Against public administration,arrival probability - By private person against public administration,arrival probability - Traffic,Endereço Comarca,count of cities covered,count of prisons covered,sum arrival probabilities,total count per sentence time
1100015,Alta Floresta d'Oeste,35,0,0,0,0,0,21,16,3,3,0,0,89,272,14.83333333,0,20,13,8,12,35,0,0,0,0,0,0,0,88,0.2272727273,0.1477272727,0.09090909091,0.1363636364,0.3977272727,0,0,0,0,0,0,0,"Alta Floresta d'Oeste, RO, Brazil",2,2,1,78
1100023,Ariquemes,839,0,0,0,5,42,68,31,28,38,15,3,103,591,17.16666667,0,321,165,294,87,362,8,30,5,0,0,1,0,1273,0.2521602514,0.1296150825,0.2309505106,0.06834249804,0.2843676355,0.006284367636,0.02356637863,0.003927729772,0,0,0.0007855459544,0,"Ariquemes, RO, Brazil",6,4,1,1069
1100049,Cacoal,445,9,43,36,23,16,31,9,21,9,1,0,830,302,138.3333333,36,184,64,177,10,254,5,8,0,1,0,0,1,704,0.2613636364,0.09090909091,0.2514204545,0.01420454545,0.3607954545,0.007102272727,0.01136363636,0,0.001420454545,0,0,0.001420454545,"Cacoal, RO, Brazil",2,3,1,643
1100056,Cerejeiras,175,5,0,0,0,11,6,0,0,0,0,0,116,153,19.33333333,31,28,11,15,22,153,0,4,0,0,0,0,0,233,0.1201716738,0.04721030043,0.0643776824,0.09442060086,0.6566523605,0,0.01716738197,0,0,0,0,0,"Cerejeiras, RO, Brazil",2,3,1,197
1100064,Colorado do Oeste,54,0,0,0,1,3,17,9,4,2,3,0,14,107,2.333333333,0,20,19,24,5,42,1,2,0,0,0,0,1,114,0.1754385965,0.1666666667,0.2105263158,0.04385964912,0.3684210526,0.008771929825,0.01754385965,0,0,0,0,0.008771929825,"Colorado do Oeste, RO, Brazil",3,2,1,93
1100080,Costa Marques,75,2,0,0,3,1,4,3,5,1,1,0,0,94,0,42,31,25,23,16,47,1,5,0,0,0,0,0,148,0.2094594595,0.1689189189,0.1554054054,0.1081081081,0.3175675676,0.006756756757,0.03378378378,0,0,0,0,0,"Costa Marques, RO, Brazil",1,3,1,95
1100106,Guajará-Mirim,321,0,1,1,26,40,72,40,45,24,1,2,390,426,65,169,447,229,124,40,57,34,17,4,4,0,0,0,956,0.4675732218,0.239539749,0.129707113,0.04184100418,0.05962343096,0.03556485356,0.01778242678,0.004184100418,0.004184100418,0,0,0,"Guajará-Mirim, RO, Brazil",2,6,1,573
1100114,Jaru,180,0,0,1,7,21,21,11,10,4,2,0,8,534,1.333333333,107,83,49,59,28,137,4,9,2,0,0,0,1,372,0.2231182796,0.1317204301,0.1586021505,0.0752688172,0.3682795699,0.01075268817,0.02419354839,0.005376344086,0,0,0,0.002688172043,"Jaru, RO, Brazil",3,4,1,257
1100122,Ji-Paraná,484,0,2,5,7,46,97,46,35,21,7,1,842,782,140.3333333,112,602,271,187,47,146,39,17,19,6,0,0,1,1335,0.4509363296,0.2029962547,0.1400749064,0.03520599251,0.1093632959,0.02921348315,0.0127340824,0.01423220974,0.004494382022,0,0,0.0007490636704,"Ji-Paraná, RO, Brazil",1,5,1,751
1100130,Machadinho d'Oeste,129,0,0,0,0,3,18,14,8,0,0,0,121,211,20.16666667,0,31,16,28,17,105,3,1,0,1,2,0,0,204,0.1519607843,0.07843137255,0.137254902,0.08333333333,0.5147058824,0.01470588235,0.004901960784,0,0.004901960784,0.009803921569,0,0,"Machadinho d'Oeste, RO, Brazil",2,2,1,172
1100155,Ouro Preto do Oeste,48,0,1,2,3,19,33,22,19,19,1,0,190,202,31.66666667,0,145,60,61,27,58,11,10,3,2,2,0,0,379,0.382585752,0.1583113456,0.1609498681,0.07124010554,0.1530343008,0.0290237467,0.02638522427,0.007915567282,0.005277044855,0.005277044855,0,0,"Ouro Preto do Oeste, RO, Brazil",5,2,1,167
1100189,Pimenta Bueno,471,0,0,0,0,0,0,0,0,0,0,0,0,373,0,105,78,10,69,31,356,5,4,2,0,1,0,1,557,0.1400359066,0.01795332136,0.1238779174,0.05565529623,0.6391382406,0.008976660682,0.007181328546,0.003590664273,0,0.001795332136,0,0.001795332136,"Pimenta Bueno, RO, Brazil",3,3,1,471
1100205,Porto Velho,5936,1,2,7,12,81,180,139,210,173,95,25,6550,7226,1091.666667,520,1728,838,673,343,3878,134,103,26,31,0,8,6,7768,0.2224510814,0.1078784758,0.08663748713,0.04415550978,0.4992276004,0.01725025747,0.01325952626,0.003347064882,0.003990731205,0,0.001029866117,0.0007723995881,"Porto Velho, RO, Brazil",3,14,1,6861
1100254,Presidente Médici,41,0,0,0,2,4,13,9,5,3,1,0,61,78,10.16666667,44,32,23,19,6,34,1,2,0,0,0,0,0,117,0.2735042735,0.1965811966,0.1623931624,0.05128205128,0.2905982906,0.008547008547,0.01709401709,0,0,0,0,0,"Presidente Médici, RO, Brazil",2,2,1,78
1100288,Rolim de Moura,389,3,0,1,2,13,20,10,10,4,0,0,147,405,24.5,36,133,56,66,44,228,6,13,1,0,0,0,0,547,0.2431444241,0.1023765996,0.1206581353,0.08043875686,0.4168190128,0.01096892139,0.02376599634,0.001828153565,0,0,0,0,"Rolim de Moura, RO, Brazil",1,5,1,452
1100304,Vilhena,359,6,37,46,41,11,0,0,0,0,0,0,1416,592,236,86,348,93,137,59,135,22,31,8,1,2,0,0,836,0.4162679426,0.1112440191,0.1638755981,0.07057416268,0.1614832536,0.02631578947,0.03708133971,0.00956937799,0.001196172249,0.002392344498,0,0,"Vilhena, RO, Brazil",2,5,1,500
1100320,São Miguel do Guaporé,149,0,0,0,0,0,0,0,0,0,0,0,33,150,5.5,28,47,21,41,28,55,3,5,1,1,0,0,0,202,0.2326732673,0.103960396,0.202970297,0.1386138614,0.2722772277,0.01485148515,0.02475247525,0.00495049505,0.00495049505,0,0,0,"São Miguel do Guaporé, RO, Brazil",2,2,1,149
1100346,Alvorada d'Oeste,212,0,0,0,0,0,0,0,0,0,0,0,0,194,0,0,57,37,70,44,135,4,6,3,0,0,0,0,356,0.1601123596,0.1039325843,0.1966292135,0.1235955056,0.3792134831,0.01123595506,0.01685393258,0.008426966292,0,0,0,0,"Alvorada d'Oeste, RO, Brazil",2,3,1,212
1100452,Buritis,299,0,0,0,0,0,0,0,0,0,0,0,84,112,14,32,115,17,86,32,177,10,6,1,0,0,1,0,445,0.2584269663,0.03820224719,0.193258427,0.07191011236,0.397752809,0.02247191011,0.01348314607,0.002247191011,0,0,0.002247191011,0,"Buritis, RO, Brazil",2,3,1,299
1101492,São Francisco do Guaporé,47,0,0,0,0,8,9,6,7,1,0,0,0,96,0,41,6,19,8,7,55,1,1,0,0,0,0,0,97,0.0618556701,0.1958762887,0.0824742268,0.07216494845,0.5670103093,0.01030927835,0.01030927835,0,0,0,0,0,"São Francisco do Guaporé, RO, Brazil",1,2,1,78
1200104,Brasiléia,0,0,0,0,0,0,0,0,0,0,0,0,32,34,5.333333333,0,0,0,0,0,24,0,0,0,0,0,0,0,24,0,0,0,0,1,0,0,0,0,0,0,0,"Brasiléia, AC, Brazil",1,1,1,0
1200203,Cruzeiro do Sul,54,4,2,1,27,136,236,120,140,77,15,0,432,813,72,218,0,0,0,0,83,0,0,0,0,0,0,0,83,0,0,0,0,1,0,0,0,0,0,0,0,"Cruzeiro do Sul, AC, Brazil",3,3,1,812
1200302,Feijó,0,0,0,0,0,0,0,0,0,0,0,0,90,205,15,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Feijó, AC, Brazil",1,1,0,0
1200401,Rio Branco,47,0,4,3,10,239,560,364,585,406,145,21,3443,3259,573.8333333,791,1851,1151,816,275,19,194,75,14,8,14,3,4,4424,0.4183996383,0.2601717902,0.1844484629,0.06216094033,0.004294755877,0.0438517179,0.01695298373,0.003164556962,0.001808318264,0.003164556962,0.000678119349,0.000904159132,"Rio Branco, AC, Brazil",1,6,1,2384
1200450,Senador Guiomard,0,0,0,0,0,49,126,123,98,65,21,4,116,796,19.33333333,0,229,141,268,156,0,16,0,0,0,0,0,0,810,0.2827160494,0.1740740741,0.3308641975,0.1925925926,0,0.01975308642,0,0,0,0,0,0,"Senador Guiomard, AC, Brazil",1,1,1,486
1200500,Sena Madureira,204,0,0,0,0,30,78,57,42,35,11,4,131,511,21.83333333,80,0,0,0,0,129,0,0,0,0,0,0,0,129,0,0,0,0,1,0,0,0,0,0,0,0,"Sena Madureira, AC, Brazil",2,2,1,461
1200609,Tarauacá,11,0,0,1,18,75,79,46,46,18,5,1,93,360,15.5,140,144,162,161,25,4,3,1,0,0,0,0,0,500,0.288,0.324,0.322,0.05,0.008,0.006,0.002,0,0,0,0,0,"Tarauacá, AC, Brazil",2,3,1,300
1301209,Coari,0,1,1,7,9,53,41,14,13,2,0,1,65,50,10.83333333,25,105,46,74,31,0,17,0,11,0,0,2,0,286,0.3671328671,0.1608391608,0.2587412587,0.1083916084,0,0.05944055944,0,0.03846153846,0,0,0.006993006993,0,"Coari, AM, Brazil",1,2,1,142
1301704,Humaitá,14,0,1,0,4,2,0,0,0,0,0,0,88,33,14.66666667,17,71,38,41,19,0,8,0,0,0,0,0,0,177,0.4011299435,0.2146892655,0.2316384181,0.1073446328,0,0.04519774011,0,0,0,0,0,0,"Humaitá, AM, Brazil",1,2,1,21
1301902,Itacoatiara,0,0,0,0,0,7,11,2,8,0,0,0,51,144,8.5,84,54,90,52,26,0,24,0,4,0,0,0,0,250,0.216,0.36,0.208,0.104,0,0.096,0,0.016,0,0,0,0,"Itacoatiara, AM, Brazil",1,2,1,28
1302603,Manaus,346,97,212,200,1673,2070,1465,595,629,331,81,6,7169,11607,1194.833333,2170,5142,3412,1710,795,0,572,79,120,34,17,6,0,11887,0.4325733995,0.2870362581,0.1438546311,0.06687978464,0,0.04811979473,0.006645915706,0.01009506183,0.002860267519,0.00143013376,0.0005047530916,0,"Manaus, AM, Brazil",1,21,1,7705
1302900,Maués,27,0,40,34,38,5,9,0,6,0,0,0,32,45,5.333333333,23,91,117,47,0,0,0,0,0,0,0,0,0,255,0.3568627451,0.4588235294,0.1843137255,0,0,0,0,0,0,0,0,0,"Maués, AM, Brazil",1,2,1,159
1303403,Parintins,36,0,0,0,16,49,30,18,10,1,0,0,49,36,8.166666667,18,35,42,44,38,0,2,0,0,0,0,0,0,161,0.2173913043,0.2608695652,0.2732919255,0.2360248447,0,0.01242236025,0,0,0,0,0,0,"Parintins, AM, Brazil",1,2,1,160
1304062,Tabatinga,42,0,0,0,5,16,30,14,4,0,0,0,83,108,13.83333333,70,25,63,26,31,0,11,3,0,2,0,0,0,161,0.1552795031,0.3913043478,0.1614906832,0.1925465839,0,0.06832298137,0.01863354037,0,0.01242236025,0,0,0,"Tabatinga, AM, Brazil",1,2,1,111
1304203,Tefé,0,0,0,0,2,41,52,14,9,3,1,0,126,125,21,90,85,71,37,40,0,9,9,6,0,0,2,0,259,0.3281853282,0.2741312741,0.1428571429,0.1544401544,0,0.03474903475,0.03474903475,0.02316602317,0,0,0.007722007722,0,"Tefé, AM, Brazil",1,2,1,122
1400100,Boa Vista,0,89,127,134,799,1059,681,183,186,69,21,0,2078,2430,346.3333333,544,1907,1212,1247,602,0,47,25,15,2,2,4,1,5064,0.3765797788,0.2393364929,0.2462480253,0.118878357,0,0.009281200632,0.004936808847,0.002962085308,0.0003949447077,0.0003949447077,0.0007898894155,0.0001974723539,"Boa Vista, RR, Brazil",3,7,1,3348
1400472,Rorainópolis,0,0,0,0,0,0,0,0,0,0,0,0,45,175,7.5,22,50,0,66,13,0,0,0,0,0,0,0,0,129,0.3875968992,0,0.511627907,0.1007751938,0,0,0,0,0,0,0,0,"Rorainópolis, RR, Brazil",1,1,1,0
1500107,Abaetetuba,0,0,0,0,5,15,141,114,45,8,3,0,355,580,59.16666667,0,264,158,92,89,0,61,8,14,0,2,0,0,688,0.3837209302,0.2296511628,0.1337209302,0.1293604651,0,0.0886627907,0.01162790698,0.02034883721,0,0.002906976744,0,0,"Abaetetuba, PA, Brazil",1,2,1,331
1500602,Altamira,0,5,29,15,57,72,60,44,22,26,8,2,111,805,18.5,0,221,128,193,114,0,23,14,18,4,0,0,0,715,0.3090909091,0.179020979,0.2699300699,0.1594405594,0,0.03216783217,0.01958041958,0.02517482517,0.005594405594,0,0,0,"Altamira, PA, Brazil",2,3,1,340
1500800,Ananindeua,0,1,1,4,34,130,234,90,77,35,6,0,707,902,117.8333333,274,232,445,117,384,0,26,15,18,4,8,5,0,1254,0.1850079745,0.3548644338,0.09330143541,0.3062200957,0,0.02073365231,0.01196172249,0.01435406699,0.003189792663,0.006379585327,0.003987240829,0,"Ananindeua, PA, Brazil",1,3,1,612
1501402,Belém,7,125,153,100,104,832,506,213,218,124,38,3,6149,6679,1024.833333,167,2260,1214,499,580,0,146,69,94,21,7,11,5,4906,0.4606604158,0.2474520995,0.1017121892,0.1182225846,0,0.02975947819,0.01406441093,0.01916021199,0.00428047289,0.001426824297,0.002242152466,0.001019160212,"Belém, PA, Brazil",1,6,1,2423
1501709,Bragança,0,0,0,0,62,51,29,12,9,0,0,0,293,122,48.83333333,0,108,45,35,30,0,13,6,0,0,0,0,0,237,0.4556962025,0.1898734177,0.1476793249,0.1265822785,0,0.05485232068,0.0253164557,0,0,0,0,0,"Bragança, PA, Brazil",2,1,1,163
1501808,Breves,0,0,0,0,0,15,68,19,23,6,1,0,132,152,22,0,51,45,36,58,0,1,0,0,0,0,0,0,191,0.2670157068,0.2356020942,0.1884816754,0.3036649215,0,0.005235602094,0,0,0,0,0,0,"Breves, PA, Brazil",3,1,1,132
1502103,Cametá,0,0,0,0,9,15,35,22,1,3,3,0,76,64,12.66666667,0,74,15,19,7,0,3,0,0,0,0,0,0,118,0.6271186441,0.1271186441,0.1610169492,0.0593220339,0,0.02542372881,0,0,0,0,0,0,"Cametá, PA, Brazil",1,1,1,88
1502202,Capanema,0,0,0,0,0,3,40,20,24,2,0,0,366,60,61,0,57,52,66,81,0,4,0,1,0,0,0,3,264,0.2159090909,0.196969697,0.25,0.3068181818,0,0.01515151515,0,0.003787878788,0,0,0,0.01136363636,"Capanema, PA, Brazil",2,1,1,89
1502400,Castanhal,0,0,0,0,0,54,97,63,54,10,5,0,256,197,42.66666667,0,317,112,115,172,0,2,0,0,0,0,0,0,718,0.4415041783,0.1559888579,0.1601671309,0.2395543175,0,0.00278551532,0,0,0,0,0,0,"Castanhal, PA, Brazil",1,1,1,283
1503606,Itaituba,0,0,9,42,39,38,25,27,13,9,0,0,258,196,43,0,109,43,100,53,0,9,9,4,1,0,0,0,328,0.3323170732,0.131097561,0.3048780488,0.1615853659,0,0.02743902439,0.02743902439,0.01219512195,0.003048780488,0,0,0,"Itaituba, PA, Brazil",2,1,1,202
1504208,Marabá,0,0,0,0,44,180,346,148,143,64,1,0,666,1418,111,292,566,235,437,160,0,31,4,17,6,0,0,0,1456,0.3887362637,0.1614010989,0.3001373626,0.1098901099,0,0.02129120879,0.002747252747,0.01167582418,0.004120879121,0,0,0,"Marabá, PA, Brazil",2,6,1,926
1504422,Marituba,0,0,0,1,16,55,173,195,221,125,12,2,283,996,47.16666667,0,753,229,388,155,0,126,109,66,5,0,20,0,1851,0.4068071313,0.1237169098,0.2096164236,0.08373851972,0,0.0680713128,0.05888708806,0.03565640194,0.002701242572,0,0.01080497029,0,"Marituba, PA, Brazil",1,3,1,800
1504604,Mocajuba,0,0,0,0,4,12,26,10,7,2,0,0,93,64,15.5,0,50,18,29,15,0,4,0,0,0,0,0,0,116,0.4310344828,0.1551724138,0.25,0.1293103448,0,0.03448275862,0,0,0,0,0,0,"Mocajuba, PA, Brazil",1,1,1,61
1505502,Paragominas,0,0,35,40,66,59,59,71,39,30,0,0,214,715,35.66666667,0,223,87,131,75,0,29,1,13,30,15,0,0,604,0.369205298,0.1440397351,0.2168874172,0.1241721854,0,0.04801324503,0.001655629139,0.02152317881,0.04966887417,0.02483443709,0,0,"Paragominas, PA, Brazil",1,3,1,399
1505536,Parauapebas,0,0,5,2,27,36,46,26,10,1,0,0,403,460,67.16666667,0,106,69,82,54,0,11,8,0,2,0,0,0,332,0.3192771084,0.2078313253,0.2469879518,0.1626506024,0,0.03313253012,0.02409638554,0,0.006024096386,0,0,0,"Parauapebas, PA, Brazil",1,1,1,153
1506138,Redenção,0,0,0,0,0,8,121,70,29,9,3,0,432,460,72,0,165,71,204,64,0,9,4,3,3,2,0,0,525,0.3142857143,0.1352380952,0.3885714286,0.1219047619,0,0.01714285714,0.007619047619,0.005714285714,0.005714285714,0.00380952381,0,0,"Redenção, PA, Brazil",3,1,1,240
1506203,Salinópolis,0,0,0,0,0,18,48,16,20,4,2,0,141,120,23.5,0,43,48,32,58,0,8,3,3,1,0,1,0,197,0.2182741117,0.2436548223,0.1624365482,0.2944162437,0,0.04060913706,0.0152284264,0.0152284264,0.005076142132,0,0.005076142132,0,"Salinópolis, PA, Brazil",1,1,1,108
1506500,Santa Isabel do Pará,64,0,0,79,145,1384,1008,1052,771,193,53,2,3990,4459,665,608,3976,2179,2868,128,0,1852,8,891,29,6,30,2,11969,0.3321914947,0.1820536386,0.2396190158,0.01069429359,0,0.1547330604,0.0006683933495,0.0744423093,0.002422925892,0.0005012950121,0.002506475061,0.0001670983374,"Santa Isabel do Pará, PA, Brazil",1,11,1,4751
1506807,Santarém,0,38,38,82,121,322,289,261,152,41,26,0,1174,1811,195.6666667,316,386,536,1011,378,0,81,37,44,0,0,0,0,2473,0.1560857258,0.2167408006,0.4088152042,0.1528507885,0,0.0327537404,0.01496158512,0.01779215528,0,0,0,0,"Santarém, PA, Brazil",3,4,1,1370
1507300,São Félix do Xingu,0,0,0,0,0,16,15,4,4,1,1,0,76,128,12.66666667,0,50,10,88,31,0,2,0,0,0,3,0,0,184,0.2717391304,0.05434782609,0.4782608696,0.1684782609,0,0.01086956522,0,0,0,0.01630434783,0,0,"São Félix do Xingu, PA, Brazil",1,1,1,41
1508001,Tomé-Açu,0,0,0,1,0,11,38,22,6,2,1,0,101,58,16.83333333,0,59,83,22,43,0,2,10,7,0,0,0,0,226,0.2610619469,0.3672566372,0.09734513274,0.1902654867,0,0.008849557522,0.04424778761,0.03097345133,0,0,0,0,"Tomé-Açu, PA, Brazil",1,1,1,81
1508100,Tucuruí,0,0,0,0,83,81,73,38,60,18,6,2,312,434,52,0,618,343,686,212,0,74,68,30,1,0,2,0,2034,0.3038348083,0.168633235,0.33726647,0.1042281219,0,0.03638151426,0.03343166175,0.01474926254,0.0004916420846,0,0.0009832841691,0,"Tucuruí, PA, Brazil",1,2,1,361
1600303,Macapá,300,13,15,49,154,1011,1725,753,803,347,30,4,1633,3012,272.1666667,459,2816,1523,1429,210,0,189,33,19,14,2,4,5,6244,0.4509929532,0.2439141576,0.2288597053,0.033632287,0,0.0302690583,0.005285073671,0.003042921204,0.002242152466,0.0003203074952,0.0006406149904,0.000800768738,"Macapá, AP, Brazil",1,10,1,5204
1600501,Oiapoque,0,0,0,0,0,0,0,0,0,0,0,0,66,52,11,52,1,16,7,0,0,0,0,0,0,0,0,0,24,0.04166666667,0.6666666667,0.2916666667,0,0,0,0,0,0,0,0,0,"Oiapoque, AP, Brazil",1,1,1,0
1700707,Alvorada,0,0,0,0,0,0,0,0,0,0,0,0,40,48,6.666666667,0,0,17,2,0,4,0,0,0,0,0,0,0,23,0,0.7391304348,0.08695652174,0,0.1739130435,0,0,0,0,0,0,0,"Alvorada, TO, Brazil",2,1,1,0
1701002,Ananás,0,0,0,0,2,6,4,2,0,0,0,0,42,41,7,26,10,13,17,1,0,0,0,0,0,0,0,0,41,0.243902439,0.3170731707,0.4146341463,0.0243902439,0,0,0,0,0,0,0,0,"Ananás, TO, Brazil",4,1,1,14
1702109,Araguaína,35,7,0,0,4,24,111,70,121,72,30,1,775,858,129.1666667,115,726,258,434,109,0,129,122,16,17,1,1,1,1814,0.4002205072,0.1422271224,0.2392502756,0.06008820287,0,0.07111356119,0.06725468578,0.008820286659,0.009371554576,0.0005512679162,0.0005512679162,0.0005512679162,"Araguaína, TO, Brazil",7,3,1,475
1702208,Araguatins,3,0,0,0,0,1,5,2,1,0,0,0,71,73,11.83333333,61,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Araguatins, TO, Brazil",3,1,0,12
1702406,Arraias,10,0,0,0,0,0,29,6,11,9,3,0,64,100,10.66666667,100,14,54,26,18,0,1,1,0,0,0,0,0,114,0.1228070175,0.4736842105,0.2280701754,0.1578947368,0,0.008771929825,0.008771929825,0,0,0,0,0,"Arraias, TO, Brazil",1,1,1,68
1702554,Augustinópolis,52,0,0,0,0,0,2,1,0,1,0,0,74,150,12.33333333,150,58,31,40,20,0,0,0,2,0,0,0,0,151,0.3841059603,0.2052980132,0.2649006623,0.1324503311,0,0,0,0.01324503311,0,0,0,0,"Augustinópolis, TO, Brazil",6,1,1,56
1705508,Colinas do Tocantins,0,0,0,0,0,0,0,0,0,0,0,0,160,60,26.66666667,60,42,10,33,18,0,0,0,0,0,0,0,0,103,0.4077669903,0.09708737864,0.3203883495,0.1747572816,0,0,0,0,0,0,0,0,"Colinas do Tocantins, TO, Brazil",6,1,1,0
1707009,Dianópolis,0,0,0,0,4,8,5,6,6,2,0,0,155,68,25.83333333,37,21,15,24,13,9,11,2,0,0,0,0,0,95,0.2210526316,0.1578947368,0.2526315789,0.1368421053,0.09473684211,0.1157894737,0.02105263158,0,0,0,0,0,"Dianópolis, TO, Brazil",5,1,1,31
1708205,Formoso do Araguaia,0,0,0,0,0,0,0,0,0,0,0,0,33,48,5.5,20,8,8,25,8,0,0,0,0,0,0,0,0,49,0.1632653061,0.1632653061,0.5102040816,0.1632653061,0,0,0,0,0,0,0,0,"Formoso do Araguaia, TO, Brazil",1,1,1,0
1709302,Guaraí,0,12,9,9,9,14,20,6,4,3,1,0,0,161,0,69,0,0,0,0,75,0,0,0,0,0,0,0,75,0,0,0,0,1,0,0,0,0,0,0,0,"Guaraí, TO, Brazil",2,1,1,87
1709500,Gurupi,0,0,0,2,8,156,85,84,74,73,1,0,780,766,130,161,801,230,443,113,0,65,11,6,8,0,1,0,1678,0.4773539928,0.137067938,0.2640047676,0.0673420739,0,0.03873659118,0.006555423123,0.00357568534,0.004767580453,0,0.0005959475566,0,"Gurupi, TO, Brazil",5,4,1,483
1713205,Miracema do Tocantins,0,0,0,0,7,22,28,23,20,6,0,0,58,100,9.666666667,50,40,30,33,36,0,0,0,0,0,0,0,0,139,0.2877697842,0.2158273381,0.2374100719,0.2589928058,0,0,0,0,0,0,0,0,"Miracema do Tocantins, TO, Brazil",1,1,1,106
1713304,Miranorte,0,0,0,0,0,1,3,2,7,2,0,0,28,48,4.666666667,0,1,17,6,0,0,0,0,0,0,0,0,0,24,0.04166666667,0.7083333333,0.25,0,0,0,0,0,0,0,0,0,"Miranorte, TO, Brazil",4,1,1,15
1714203,Natividade,0,16,6,5,9,2,0,0,0,0,0,0,0,12,0,12,6,8,8,11,2,1,1,0,1,0,0,0,38,0.1578947368,0.2105263158,0.2105263158,0.2894736842,0.05263157895,0.02631578947,0.02631578947,0,0.02631578947,0,0,0,"Natividade, TO, Brazil",3,1,1,38
1715754,Palmeirópolis,0,0,0,0,0,0,0,0,0,0,0,0,7,45,1.166666667,0,5,8,15,15,0,2,0,0,0,0,0,0,45,0.1111111111,0.1777777778,0.3333333333,0.3333333333,0,0.04444444444,0,0,0,0,0,0,"Palmeirópolis, TO, Brazil",2,1,1,0
1716109,Paraíso do Tocantins,0,0,0,0,0,0,0,0,0,0,0,0,352,267,58.66666667,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Paraíso do Tocantins, TO, Brazil",6,1,0,0
1716703,Colméia,0,0,0,0,0,0,0,0,0,0,0,0,24,50,4,0,3,0,20,3,0,0,0,0,0,0,0,0,26,0.1153846154,0,0.7692307692,0.1153846154,0,0,0,0,0,0,0,0,"Colméia, TO, Brazil",5,1,1,0
1718204,Porto Nacional,9,0,0,0,0,4,29,7,13,7,3,0,163,96,27.16666667,0,25,9,32,34,0,0,0,0,0,0,0,0,100,0.25,0.09,0.32,0.34,0,0,0,0,0,0,0,0,"Porto Nacional, TO, Brazil",8,1,1,72
1720903,Taguatinga,0,0,0,0,0,0,0,0,0,0,0,0,11,25,1.833333333,0,7,14,18,5,0,0,2,0,0,0,0,0,46,0.152173913,0.3043478261,0.3913043478,0.1086956522,0,0,0.04347826087,0,0,0,0,0,"Taguatinga, TO, Brazil",2,1,1,0
1721000,Palmas,0,1,2,0,2,3,7,7,1,1,0,0,316,1663,52.66666667,0,6,31,5,1,0,0,0,0,0,0,0,0,43,0.1395348837,0.7209302326,0.1162790698,0.02325581395,0,0,0,0,0,0,0,0,"Palmas, TO, Brazil",1,3,1,24
1721208,Tocantinópolis,6,0,0,0,0,3,13,2,7,4,0,0,142,88,23.66666667,70,13,18,26,24,24,0,0,0,0,0,0,0,105,0.1238095238,0.1714285714,0.2476190476,0.2285714286,0.2285714286,0,0,0,0,0,0,0,"Tocantinópolis, TO, Brazil",6,1,1,35
2100055,Açailândia,144,1,3,11,7,53,48,16,15,2,0,0,241,293,40.16666667,73,254,141,202,81,4,67,35,10,14,0,3,1,812,0.3128078818,0.1736453202,0.2487684729,0.09975369458,0.004926108374,0.08251231527,0.04310344828,0.01231527094,0.01724137931,0,0.003694581281,0.001231527094,"Açailândia, MA, Brazil",3,1,1,300
2101202,Bacabal,157,11,4,12,14,43,72,29,7,4,0,0,299,362,49.83333333,80,396,129,253,56,19,88,54,32,20,2,2,0,1051,0.3767840152,0.1227402474,0.2407231208,0.05328258801,0.01807802093,0.08372978116,0.05137963844,0.03044719315,0.01902949572,0.001902949572,0.001902949572,0,"Bacabal, MA, Brazil",4,2,1,353
2101400,Balsas,150,10,3,18,15,54,45,12,14,5,0,0,253,289,42.16666667,72,345,154,198,50,7,85,55,13,4,0,1,0,912,0.3782894737,0.1688596491,0.2171052632,0.0548245614,0.007675438596,0.09320175439,0.06030701754,0.01425438596,0.004385964912,0,0.001096491228,0,"Balsas, MA, Brazil",3,1,1,326
2101608,Barra do Corda,47,2,0,3,0,9,11,5,7,0,0,0,108,15,18,0,41,26,60,24,4,17,8,4,3,0,0,0,187,0.2192513369,0.1390374332,0.320855615,0.128342246,0.02139037433,0.09090909091,0.04278074866,0.02139037433,0.01604278075,0,0,0,"Barra do Corda, MA, Brazil",3,1,1,84
2102606,Cândido Mendes,45,0,0,0,1,14,8,0,2,0,0,0,66,165,11,42,55,33,39,17,3,14,2,3,0,0,0,0,166,0.3313253012,0.1987951807,0.234939759,0.1024096386,0.01807228916,0.0843373494,0.01204819277,0.01807228916,0,0,0,0,"Cândido Mendes, MA, Brazil",2,1,1,70
2102804,Carolina,38,0,0,3,3,6,2,0,0,0,0,0,10,70,1.666666667,21,25,37,19,0,4,8,4,8,0,0,0,0,105,0.2380952381,0.3523809524,0.180952381,0,0.0380952381,0.07619047619,0.0380952381,0.07619047619,0,0,0,0,"Carolina, MA, Brazil",1,1,1,52
2103000,Caxias,61,10,4,6,4,22,34,9,5,1,0,0,225,167,37.5,44,179,39,106,33,13,41,11,14,9,0,0,0,445,0.402247191,0.08764044944,0.2382022472,0.07415730337,0.02921348315,0.09213483146,0.02471910112,0.03146067416,0.0202247191,0,0,0,"Caxias, MA, Brazil",3,1,1,156
2103208,Chapadinha,103,9,6,4,10,33,34,15,18,1,0,0,316,208,52.66666667,52,214,95,153,63,16,60,25,15,3,0,1,1,646,0.3312693498,0.1470588235,0.2368421053,0.09752321981,0.02476780186,0.09287925697,0.0386996904,0.02321981424,0.004643962848,0,0.001547987616,0.001547987616,"Chapadinha, MA, Brazil",2,1,1,233
2103307,Codó,45,4,4,5,7,46,38,12,8,2,0,0,221,185,36.83333333,48,166,42,111,25,9,46,19,12,10,0,0,0,440,0.3772727273,0.09545454545,0.2522727273,0.05681818182,0.02045454545,0.1045454545,0.04318181818,0.02727272727,0.02272727273,0,0,0,"Codó, MA, Brazil",1,1,1,171
2103505,Colinas,42,1,4,8,9,20,33,13,3,1,0,0,63,129,10.5,15,159,23,126,39,5,17,20,6,3,0,0,1,399,0.3984962406,0.05764411028,0.3157894737,0.0977443609,0.01253132832,0.04260651629,0.05012531328,0.01503759398,0.007518796992,0,0,0.002506265664,"Colinas, MA, Brazil",2,1,1,134
2103604,Coroatá,77,1,3,2,4,41,26,11,10,2,0,0,62,217,10.33333333,64,158,84,102,18,7,57,17,10,12,0,1,0,466,0.339055794,0.1802575107,0.2188841202,0.03862660944,0.01502145923,0.1223175966,0.0364806867,0.02145922747,0.02575107296,0,0.002145922747,0,"Coroatá, MA, Brazil",2,1,1,177
2104677,Governador Nunes Freire,190,5,2,6,10,56,94,30,31,3,0,0,81,458,13.5,76,247,87,338,272,18,73,28,6,10,0,1,0,1080,0.2287037037,0.08055555556,0.312962963,0.2518518519,0.01666666667,0.06759259259,0.02592592593,0.005555555556,0.009259259259,0,0.0009259259259,0,"Governador Nunes Freire, MA, Brazil",3,1,1,427
2104800,Grajaú,45,7,0,1,2,2,8,2,3,0,0,0,103,50,17.16666667,0,36,16,90,10,9,28,3,7,1,0,0,0,200,0.18,0.08,0.45,0.05,0.045,0.14,0.015,0.035,0.005,0,0,0,"Grajaú, MA, Brazil",3,1,1,70
2105302,Imperatriz,290,24,12,32,51,205,196,68,50,14,0,0,768,1019,128,198,1051,376,586,148,51,272,107,105,52,3,4,2,2757,0.3812114617,0.1363801233,0.2125498731,0.0536815379,0.01849836779,0.09865796155,0.03881030105,0.03808487486,0.01886108089,0.001088139282,0.001450852376,0.0007254261879,"Imperatriz, MA, Brazil",4,4,1,942
2105401,Itapecuru Mirim,111,6,0,0,3,32,41,15,8,3,0,0,301,236,50.16666667,54,168,114,146,59,16,50,42,15,6,2,3,0,621,0.270531401,0.1835748792,0.2351046699,0.09500805153,0.02576489533,0.08051529791,0.06763285024,0.02415458937,0.009661835749,0.003220611916,0.004830917874,0,"Itapecuru Mirim, MA, Brazil",3,2,1,219
2107506,Paço do Lumiar,7,0,0,2,5,12,25,22,12,2,0,0,60,138,10,43,70,31,86,16,1,16,6,12,5,0,0,0,243,0.2880658436,0.1275720165,0.353909465,0.0658436214,0.004115226337,0.0658436214,0.02469135802,0.04938271605,0.02057613169,0,0,0,"Paço do Lumiar, MA, Brazil",1,2,1,87
2108207,Pedreiras,126,9,10,17,13,74,80,43,33,15,0,0,186,574,31,108,426,105,339,118,12,90,39,12,10,0,1,0,1152,0.3697916667,0.09114583333,0.2942708333,0.1024305556,0.01041666667,0.078125,0.03385416667,0.01041666667,0.008680555556,0,0.0008680555556,0,"Pedreiras, MA, Brazil",4,2,1,420
2108603,Pinheiro,309,14,3,6,28,88,97,33,19,10,0,0,523,576,87.16666667,156,480,323,383,51,16,180,77,69,24,7,4,0,1614,0.2973977695,0.2001239157,0.2372986369,0.03159851301,0.009913258984,0.1115241636,0.04770755886,0.04275092937,0.01486988848,0.004337050805,0.002478314746,0,"Pinheiro, MA, Brazil",3,2,1,607
2109007,Porto Franco,39,1,0,2,7,16,19,10,8,3,0,0,43,102,7.166666667,25,95,4,92,28,5,27,18,5,8,0,0,0,282,0.3368794326,0.01418439716,0.3262411348,0.09929078014,0.01773049645,0.09574468085,0.06382978723,0.01773049645,0.02836879433,0,0,0,"Porto Franco, MA, Brazil",4,1,1,105
2109106,Presidente Dutra,91,8,1,3,5,10,25,3,4,1,0,0,189,129,31.5,48,54,3,69,37,27,9,8,3,5,0,0,0,215,0.2511627907,0.01395348837,0.3209302326,0.1720930233,0.1255813953,0.04186046512,0.03720930233,0.01395348837,0.02325581395,0,0,0,"Presidente Dutra, MA, Brazil",1,1,1,151
2109601,Rosário,93,0,2,3,3,27,37,9,7,5,0,0,196,162,32.66666667,36,84,63,72,80,6,35,12,1,2,1,0,0,356,0.2359550562,0.1769662921,0.202247191,0.2247191011,0.01685393258,0.09831460674,0.03370786517,0.002808988764,0.005617977528,0.002808988764,0,0,"Rosário, MA, Brazil",2,1,1,186
2109908,Santa Inês,99,32,6,9,6,51,44,14,10,3,0,0,311,212,51.83333333,54,124,94,157,37,31,65,32,22,7,1,1,0,571,0.2171628722,0.1646234676,0.2749562172,0.06479859895,0.05429071804,0.1138353765,0.05604203152,0.03852889667,0.0122591944,0.001751313485,0.001751313485,0,"Santa Inês, MA, Brazil",2,1,1,274
2111102,São João dos Patos,76,9,5,5,4,6,11,3,7,2,0,0,209,101,34.83333333,30,130,39,156,29,7,28,24,2,4,0,0,0,419,0.3102625298,0.09307875895,0.3723150358,0.0692124105,0.01670644391,0.06682577566,0.05727923628,0.00477326969,0.009546539379,0,0,0,"São João dos Patos, MA, Brazil",2,1,1,128
2111300,São Luís,1075,60,66,182,245,1479,1142,365,320,125,0,0,2787,6579,464.5,1486,7123,3197,3278,521,595,4260,1002,1260,333,20,49,5,21643,0.3291133392,0.1477151966,0.1514577462,0.02407244837,0.02749156771,0.196830384,0.04629672411,0.05821743751,0.01538603706,0.0009240863097,0.002264011459,0.0002310215774,"São Luís, MA, Brazil",1,16,1,5059
2112209,Timon,230,29,7,20,48,225,174,49,35,18,0,0,484,1032,80.66666667,189,1232,390,488,94,44,333,180,148,48,0,0,1,2958,0.4164976335,0.1318458418,0.1649763354,0.03177822853,0.01487491548,0.1125760649,0.06085192698,0.05003380663,0.01622718053,0,0,0.000338066261,"Timon, MA, Brazil",1,3,1,835
2112803,Viana,52,4,5,2,8,28,44,19,14,6,0,0,201,213,33.5,34,166,74,99,47,7,50,36,14,11,0,2,0,506,0.3280632411,0.1462450593,0.1956521739,0.09288537549,0.01383399209,0.09881422925,0.07114624506,0.02766798419,0.02173913043,0,0.00395256917,0,"Viana, MA, Brazil",2,3,1,182
2114007,Zé Doca,61,2,2,4,6,21,24,10,9,5,0,0,134,125,22.33333333,31,100,63,74,32,10,42,36,19,7,0,1,0,384,0.2604166667,0.1640625,0.1927083333,0.08333333333,0.02604166667,0.109375,0.09375,0.04947916667,0.01822916667,0,0.002604166667,0,"Zé Doca, MA, Brazil",3,1,1,144
2200400,Altos,514,1,0,6,66,338,197,88,100,27,11,0,1766,1126,294.3333333,603,1080,504,497,173,165,64,10,7,1,0,0,2,2503,0.4314822213,0.20135837,0.1985617259,0.06911705953,0.06592089493,0.02556931682,0.003995205753,0.002796644027,0.0003995205753,0,0,0.0007990411506,"Altos, PI, Brazil",3,4,1,1348
2201903,Bom Jesus,18,0,0,0,0,5,22,3,10,1,0,0,195,76,32.5,0,63,0,82,35,29,0,0,0,0,0,0,0,209,0.3014354067,0,0.3923444976,0.1674641148,0.1387559809,0,0,0,0,0,0,0,"Bom Jesus, PI, Brazil",2,1,1,59
2202208,Campo Maior,4,0,1,4,5,28,58,28,32,5,1,1,187,144,31.16666667,0,121,117,72,23,0,19,0,4,1,0,0,0,357,0.3389355742,0.3277310924,0.2016806723,0.06442577031,0,0.05322128852,0,0.01120448179,0.002801120448,0,0,0,"Campo Maior, PI, Brazil",4,1,1,167
2203701,Esperantina,0,0,0,0,0,6,78,18,9,8,1,0,202,143,33.66666667,0,136,45,109,75,0,23,0,4,0,0,0,0,392,0.3469387755,0.1147959184,0.2780612245,0.1913265306,0,0.05867346939,0,0.01020408163,0,0,0,0,"Esperantina, PI, Brazil",2,1,1,120
2203909,Floriano,0,0,0,0,3,33,81,36,33,11,4,0,164,200,27.33333333,0,165,54,93,56,0,16,20,1,21,0,0,0,426,0.3873239437,0.1267605634,0.2183098592,0.1314553991,0,0.03755868545,0.04694835681,0.00234741784,0.04929577465,0,0,0,"Floriano, PI, Brazil",1,1,1,201
2207009,Oeiras,3,0,0,0,2,6,14,14,9,2,1,0,63,47,10.5,0,34,20,38,17,0,3,4,2,0,0,0,0,118,0.2881355932,0.1694915254,0.3220338983,0.1440677966,0,0.02542372881,0.03389830508,0.01694915254,0,0,0,0,"Oeiras, PI, Brazil",7,1,1,51
2207702,Parnaíba,17,0,0,4,27,79,132,49,49,32,4,0,364,176,60.66666667,0,359,302,257,91,148,27,6,5,0,1,0,1,1197,0.2999164578,0.2522974102,0.2147034252,0.07602339181,0.1236424394,0.02255639098,0.005012531328,0.00417710944,0,0.0008354218881,0,0.0008354218881,"Parnaíba, PI, Brazil",2,1,1,393
2208007,Picos,151,0,0,0,0,20,38,13,28,14,2,0,258,306,43,0,323,115,185,70,0,13,3,2,0,0,2,2,715,0.4517482517,0.1608391608,0.2587412587,0.0979020979,0,0.01818181818,0.004195804196,0.002797202797,0,0,0.002797202797,0.002797202797,"Picos, PI, Brazil",9,2,1,266
2210607,São Raimundo Nonato,0,0,0,0,0,12,21,11,14,4,3,0,164,154,27.33333333,0,92,43,74,10,0,8,5,4,0,0,0,0,236,0.3898305085,0.1822033898,0.313559322,0.04237288136,0,0.03389830508,0.02118644068,0.01694915254,0,0,0,0,"São Raimundo Nonato, PI, Brazil",9,1,1,65
2211001,Teresina,1137,0,0,0,15,176,203,98,97,39,9,2,1944,2765,324,41,1656,785,477,208,378,178,30,22,7,10,1,6,3758,0.4406599255,0.2088877062,0.1269292177,0.05534858968,0.1005854178,0.04736562001,0.007982969665,0.005854177754,0.001862692922,0.002660989888,0.0002660989888,0.001596593933,"Teresina, PI, Brazil",2,4,1,1776
2300309,Acopiara,0,0,0,0,0,0,0,0,0,0,0,0,147,47,24.5,47,3,4,1,0,28,0,0,0,0,0,0,0,36,0.08333333333,0.1111111111,0.02777777778,0,0.7777777778,0,0,0,0,0,0,0,"Acopiara, CE, Brazil",1,1,1,0
2301000,Aquiraz,333,0,1,5,25,251,514,225,208,107,43,10,5016,2734,836,950,2281,1875,1163,365,0,494,338,179,102,7,7,3,6814,0.3347519812,0.2751687702,0.1706780158,0.05356618726,0,0.07249779865,0.04960375697,0.02626944526,0.0149691811,0.001027296742,0.001027296742,0.0004402700323,"Aquiraz, CE, Brazil",1,5,1,1722
2303709,Caucaia,0,0,0,0,6,141,194,109,135,13,4,0,0,1212,0,1212,1109,531,628,255,0,264,176,84,32,4,13,1,3097,0.3580884727,0.171456248,0.2027768809,0.08233774621,0,0.08524378431,0.05682918954,0.02712302228,0.01033257992,0.00129157249,0.004197610591,0.0003228931224,"Caucaia, CE, Brazil",1,1,1,602
2303808,Cedro,0,0,0,0,0,0,5,0,0,0,0,0,135,70,22.5,50,4,3,2,1,24,0,0,0,0,0,0,0,34,0.1176470588,0.08823529412,0.05882352941,0.02941176471,0.7058823529,0,0,0,0,0,0,0,"Cedro, CE, Brazil",1,1,1,5
2304202,Crato,0,0,0,0,2,9,21,7,3,0,0,0,81,140,13.5,84,33,92,20,2,0,5,4,2,2,0,0,0,160,0.20625,0.575,0.125,0.0125,0,0.03125,0.025,0.0125,0.0125,0,0,0,"Crato, CE, Brazil",1,1,1,42
2304400,Fortaleza,6,7,6,45,406,2136,2228,745,668,95,10,0,9452,9854,1575.333333,0,7192,7272,1522,272,0,1142,109,623,192,10,41,4,18379,0.3913161761,0.39566897,0.08281190489,0.01479949943,0,0.06213613363,0.005930681756,0.03389738288,0.01044670548,0.0005440992437,0.002230806899,0.0002176396975,"Fortaleza, CE, Brazil",1,2,1,6352
2304459,Fortim,0,0,0,0,0,1,1,0,0,0,0,0,206,34,34.33333333,34,6,4,6,6,8,0,0,0,1,0,0,0,31,0.1935483871,0.1290322581,0.1935483871,0.1935483871,0.2580645161,0,0,0,0.03225806452,0,0,0,"Fortim, CE, Brazil",1,1,1,2
2305407,Icó,0,0,1,1,0,7,9,5,1,0,2,0,160,70,26.66666667,70,31,38,48,8,0,12,7,0,0,0,0,0,144,0.2152777778,0.2638888889,0.3333333333,0.05555555556,0,0.08333333333,0.04861111111,0,0,0,0,0,"Icó, CE, Brazil",1,1,1,26
2306256,Itaitinga,458,1,7,41,140,1360,2794,1496,1483,898,187,49,6938,8598,1156.333333,4466,14371,7247,6622,1205,0,3943,3146,1457,361,13,90,15,38470,0.373563816,0.1883805563,0.1721341305,0.03132310892,0,0.102495451,0.08177800884,0.03787366779,0.009383935534,0.0003379256564,0.002339485313,0.0003899142189,"Itaitinga, CE, Brazil",1,10,1,8914
2307304,Juazeiro do Norte,66,2,3,9,19,125,351,155,136,65,17,3,1311,997,218.5,272,947,620,890,286,0,241,145,44,11,3,6,3,3196,0.2963078849,0.1939924906,0.2784730914,0.08948685857,0,0.07540675845,0.04536921151,0.01376720901,0.003441802253,0.0009386733417,0.001877346683,0.0009386733417,"Juazeiro do Norte, CE, Brazil",1,2,1,951
2309409,Novo Oriente,7,0,0,0,0,0,1,0,0,0,0,0,342,35,57,35,24,15,27,9,0,10,4,1,0,2,0,0,92,0.2608695652,0.1630434783,0.2934782609,0.09782608696,0,0.1086956522,0.04347826087,0.01086956522,0,0.02173913043,0,0,"Novo Oriente, CE, Brazil",1,1,1,8
2309706,Pacatuba,81,1,1,3,10,140,299,151,137,52,16,4,663,1305,110.5,652,1784,656,606,5,0,331,269,127,34,0,6,0,3818,0.4672603457,0.1718177056,0.1587218439,0.001309586171,0,0.0866946045,0.07045573599,0.03326348874,0.008905185961,0,0.001571503405,0,"Pacatuba, CE, Brazil",1,1,1,895
2311306,Quixadá,0,0,0,2,1,0,3,5,1,0,0,0,384,138,64,138,38,0,28,14,44,0,0,2,3,0,0,0,129,0.2945736434,0,0.2170542636,0.1085271318,0.3410852713,0,0,0.01550387597,0.02325581395,0,0,0,"Quixadá, CE, Brazil",3,1,1,12
2312908,Sobral,0,0,0,5,7,67,121,44,26,11,4,0,916,1506,152.6666667,162,1244,759,948,260,0,341,163,76,9,5,5,2,3812,0.3263378804,0.1991080797,0.2486883526,0.06820566632,0,0.08945435467,0.04275970619,0.01993704092,0.002360965373,0.001311647429,0.001311647429,0.0005246589717,"Sobral, CE, Brazil",1,2,1,285
2313401,Tianguá,0,0,0,0,0,2,3,0,0,1,0,0,336,153,56,93,35,11,64,21,0,5,6,2,0,1,0,0,145,0.2413793103,0.07586206897,0.4413793103,0.1448275862,0,0.03448275862,0.04137931034,0.01379310345,0,0.006896551724,0,0,"Tianguá, CE, Brazil",1,1,1,6
2313500,Trairi,0,0,0,0,0,1,0,1,0,0,0,0,89,54,14.83333333,54,5,8,6,2,13,3,3,0,0,0,0,0,40,0.125,0.2,0.15,0.05,0.325,0.075,0.075,0,0,0,0,0,"Trairi, CE, Brazil",1,1,1,2
2401008,Apodi,0,0,0,0,0,0,0,0,0,0,0,0,102,100,17,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Apodi, RN, Brazil",5,2,0,0
2402006,Caicó,0,0,17,3,17,70,175,75,66,16,21,1,525,553,87.5,0,36,20,31,12,0,1,0,6,0,0,0,0,106,0.3396226415,0.1886792453,0.2924528302,0.1132075472,0,0.009433962264,0,0.05660377358,0,0,0,0,"Caicó, RN, Brazil",3,2,1,461
2402303,Caraúbas,0,0,0,0,0,0,0,0,0,0,0,0,249,264,41.5,264,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Caraúbas, RN, Brazil",1,1,0,0
2402600,Ceará-Mirim,0,0,0,0,0,0,0,0,0,0,0,0,548,1364,91.33333333,1364,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ceará-Mirim, RN, Brazil",3,1,0,0
2403251,Parnamirim,0,0,0,0,0,0,0,0,0,0,0,0,2443,992,407.1666667,168,19,44,51,1,0,0,0,1,0,0,0,0,116,0.1637931034,0.3793103448,0.4396551724,0.008620689655,0,0,0,0.008620689655,0,0,0,0,"Parnamirim, RN, Brazil",1,5,1,0
2408003,Mossoró,8,0,0,1,0,3,3,6,5,13,7,5,1039,1676,173.1666667,526,20,18,13,0,0,7,0,3,2,1,0,0,64,0.3125,0.28125,0.203125,0,0,0.109375,0,0.046875,0.03125,0.015625,0,0,"Mossoró, RN, Brazil",2,6,1,51
2408102,Natal,0,13,191,0,0,0,0,0,0,0,0,0,2654,10659,442.3333333,374,7,1,9,2,49,0,0,0,0,0,0,0,68,0.1029411765,0.01470588235,0.1323529412,0.02941176471,0.7205882353,0,0,0,0,0,0,0,"Natal, RN, Brazil",1,8,1,204
2408201,Nísia Floresta,0,0,0,0,0,0,0,0,0,0,0,0,499,2834,83.16666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Nísia Floresta, RN, Brazil",1,2,0,0
2408300,Nova Cruz,0,0,0,0,0,0,0,0,0,0,0,0,251,252,41.83333333,252,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Nova Cruz, RN, Brazil",4,1,0,0
2409407,Pau dos Ferros,0,0,0,0,0,0,0,0,0,0,0,0,232,360,38.66666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Pau dos Ferros, RN, Brazil",7,2,0,0
2500304,Alagoa Grande,59,0,0,0,0,0,0,0,0,0,0,0,0,59,0,12,0,0,0,0,59,0,0,0,0,0,0,0,59,0,0,0,0,1,0,0,0,0,0,0,0,"Alagoa Grande, PB, Brazil",2,1,1,59
2500403,Alagoa Nova,9,0,0,5,15,14,8,10,7,0,0,0,17,68,2.833333333,21,41,12,25,8,0,8,0,0,0,0,0,1,95,0.4315789474,0.1263157895,0.2631578947,0.08421052632,0,0.08421052632,0,0,0,0,0,0.01052631579,"Alagoa Nova, PB, Brazil",3,1,1,68
2500502,Alagoinha,0,2,0,3,1,5,2,4,1,2,0,0,23,10,3.833333333,5,20,2,19,7,0,3,2,1,0,0,0,0,54,0.3703703704,0.03703703704,0.3518518519,0.1296296296,0,0.05555555556,0.03703703704,0.01851851852,0,0,0,0,"Alagoinha, PB, Brazil",2,1,1,20
2500601,Alhandra,0,0,0,0,0,16,20,12,5,0,0,0,19,75,3.166666667,50,10,49,25,11,0,12,0,0,0,0,0,0,107,0.09345794393,0.4579439252,0.2336448598,0.1028037383,0,0.1121495327,0,0,0,0,0,0,"Alhandra, PB, Brazil",1,1,1,53
2500700,São João do Rio do Peixe,24,1,2,4,12,15,16,7,1,1,0,0,24,42,4,12,14,16,32,10,0,8,3,0,0,0,2,0,85,0.1647058824,0.1882352941,0.3764705882,0.1176470588,0,0.09411764706,0.03529411765,0,0,0,0.02352941176,0,"São João do Rio do Peixe, PB, Brazil",5,1,1,83
2501005,Araruna,0,0,0,0,2,4,24,2,1,0,0,0,35,64,5.833333333,48,9,8,8,5,12,7,0,0,0,0,0,0,49,0.1836734694,0.1632653061,0.1632653061,0.1020408163,0.2448979592,0.1428571429,0,0,0,0,0,0,"Araruna, PB, Brazil",3,1,1,33
2501104,Areia,0,4,3,4,5,7,13,8,4,0,0,0,11,63,1.833333333,8,34,18,34,11,0,6,2,4,0,3,2,0,114,0.298245614,0.1578947368,0.298245614,0.09649122807,0,0.05263157895,0.01754385965,0.0350877193,0,0.02631578947,0.01754385965,0,"Areia, PB, Brazil",1,1,1,48
2501302,Aroeiras,20,0,0,0,0,0,0,0,0,0,0,0,10,18,1.666666667,18,4,5,9,4,0,2,4,2,0,1,0,1,32,0.125,0.15625,0.28125,0.125,0,0.0625,0.125,0.0625,0,0.03125,0,0.03125,"Aroeiras, PB, Brazil",2,1,1,20
2501500,Bananeiras,6,2,2,3,6,12,11,10,5,3,2,0,22,63,3.666666667,3,30,6,16,13,1,4,1,3,0,0,0,2,76,0.3947368421,0.07894736842,0.2105263158,0.1710526316,0.01315789474,0.05263157895,0.01315789474,0.03947368421,0,0,0,0.02631578947,"Bananeiras, PB, Brazil",2,1,1,62
2501807,Bayeux,0,0,0,0,0,0,0,0,0,0,0,0,142,90,23.66666667,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Bayeux, PB, Brazil",1,1,0,0
2501906,Belém,0,0,0,1,8,13,12,7,3,1,0,0,18,78,3,34,26,0,18,6,0,0,0,6,0,0,0,0,56,0.4642857143,0,0.3214285714,0.1071428571,0,0,0,0.1071428571,0,0,0,0,"Belém, PB, Brazil",2,1,1,45
2503704,Cajazeiras,184,1,2,3,14,36,118,47,38,35,13,3,151,320,25.16666667,15,256,163,117,23,0,49,4,0,0,1,1,0,614,0.4169381107,0.2654723127,0.1905537459,0.03745928339,0,0.07980456026,0.00651465798,0,0,0.001628664495,0.001628664495,0,"Cajazeiras, PB, Brazil",3,2,1,494
2504009,Campina Grande,0,0,0,3,5,42,64,30,34,11,0,0,1303,736,217.1666667,185,1262,603,569,137,185,434,14,36,56,17,1,1,3315,0.380693816,0.1819004525,0.1716440422,0.04132730015,0.05580693816,0.1309200603,0.004223227753,0.01085972851,0.01689291101,0.005128205128,0.0003016591252,0.0003016591252,"Campina Grande, PB, Brazil",4,4,1,189
2504306,Catolé do Rocha,207,0,0,0,0,0,0,0,0,0,0,0,106,207,17.66666667,55,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Catolé do Rocha, PB, Brazil",6,1,0,207
2504405,Conceição,0,2,3,4,2,5,18,2,9,2,0,0,46,66,7.666666667,25,5,30,17,5,0,4,1,0,2,0,0,0,64,0.078125,0.46875,0.265625,0.078125,0,0.0625,0.015625,0,0.03125,0,0,0,"Conceição, PB, Brazil",4,1,1,47
2504801,Coremas,0,0,0,0,1,4,17,3,4,7,0,0,14,36,2.333333333,14,34,20,19,5,0,7,11,1,0,0,0,0,97,0.3505154639,0.206185567,0.1958762887,0.05154639175,0,0.07216494845,0.1134020619,0.01030927835,0,0,0,0,"Coremas, PB, Brazil",1,1,1,36
2504900,Cruz do Espírito Santo,2,0,0,0,0,0,5,3,2,0,0,0,2,23,0.3333333333,23,11,13,2,1,0,1,2,0,0,0,0,0,30,0.3666666667,0.4333333333,0.06666666667,0.03333333333,0,0.03333333333,0.06666666667,0,0,0,0,0,"Cruz do Espírito Santo, PB, Brazil",1,1,1,12
2505105,Cuité,81,0,0,0,0,1,16,4,3,2,0,0,30,36,5,36,36,11,28,12,0,9,10,4,0,0,0,0,110,0.3272727273,0.1,0.2545454545,0.1090909091,0,0.08181818182,0.09090909091,0.03636363636,0,0,0,0,"Cuité, PB, Brazil",2,1,1,107
2506004,Esperança,0,0,0,0,0,4,7,5,1,0,1,0,24,24,4,6,7,4,8,5,0,0,0,0,0,0,0,0,24,0.2916666667,0.1666666667,0.3333333333,0.2083333333,0,0,0,0,0,0,0,0,"Esperança, PB, Brazil",3,1,1,18
2506301,Guarabira,80,4,8,13,26,68,140,48,86,39,11,1,166,342,27.66666667,120,368,110,177,37,0,22,15,12,4,0,0,0,745,0.4939597315,0.1476510067,0.2375838926,0.04966442953,0,0.02953020134,0.02013422819,0.01610738255,0.005369127517,0,0,0,"Guarabira, PB, Brazil",3,2,1,524
2506400,Gurinhém,0,0,0,0,0,0,0,0,0,0,0,0,17,38,2.833333333,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Gurinhém, PB, Brazil",2,1,0,0
2506806,Ingá,0,0,0,0,0,4,6,3,0,0,0,0,11,35,1.833333333,10,0,12,2,2,0,3,2,0,0,0,0,0,21,0,0.5714285714,0.09523809524,0.09523809524,0,0.1428571429,0.09523809524,0,0,0,0,0,"Ingá, PB, Brazil",4,1,1,13
2506905,Itabaiana,38,0,0,0,0,9,20,10,5,0,0,0,31,72,5.166666667,42,10,40,20,10,0,2,0,0,0,0,0,0,82,0.1219512195,0.487804878,0.243902439,0.1219512195,0,0.0243902439,0,0,0,0,0,0,"Itabaiana, PB, Brazil",4,1,1,82
2507002,Itaporanga,56,0,0,0,0,4,11,7,2,2,0,0,51,36,8.5,36,49,21,28,8,30,5,7,0,0,0,0,0,148,0.3310810811,0.1418918919,0.1891891892,0.05405405405,0.2027027027,0.03378378378,0.0472972973,0,0,0,0,0,"Itaporanga, PB, Brazil",7,1,1,82
2507309,Jacaraú,61,0,0,0,0,0,0,0,0,0,0,0,13,61,2.166666667,21,21,16,9,7,0,6,4,1,0,0,0,0,64,0.328125,0.25,0.140625,0.109375,0,0.09375,0.0625,0.015625,0,0,0,0,"Jacaraú, PB, Brazil",4,1,1,61
2507507,João Pessoa,2685,0,7,16,77,227,285,206,95,48,31,2,2845,5665,474.1666667,764,3659,1485,1212,243,2646,629,231,224,74,10,9,7,10429,0.3508485953,0.1423914086,0.1162144021,0.02330041231,0.2537156007,0.06031258989,0.02214977467,0.02147856937,0.007095598811,0.0009588647042,0.0008629782338,0.0006712052929,"João Pessoa, PB, Brazil",1,10,1,3679
2507705,Juazeirinho,0,0,0,0,7,12,13,8,1,1,0,0,29,28,4.833333333,14,29,10,17,9,0,4,5,0,1,3,0,0,78,0.3717948718,0.1282051282,0.2179487179,0.1153846154,0,0.05128205128,0.0641025641,0,0.01282051282,0.03846153846,0,0,"Juazeirinho, PB, Brazil",3,1,1,42
2508802,Malta,4,1,3,1,1,3,9,5,5,3,0,0,72,36,12,36,1,3,34,23,5,0,0,0,0,0,0,0,66,0.01515151515,0.04545454545,0.5151515152,0.3484848485,0.07575757576,0,0,0,0,0,0,0,"Malta, PB, Brazil",3,1,1,35
2508901,Mamanguape,0,0,0,0,0,0,0,0,0,0,0,0,50,42,8.333333333,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Mamanguape, PB, Brazil",5,1,0,0
2509701,Monteiro,0,0,3,17,58,70,39,37,11,2,1,0,48,231,8,68,116,156,159,24,0,9,0,3,3,0,1,2,473,0.245243129,0.3298097252,0.3361522199,0.05073995772,0,0.01902748414,0,0.006342494715,0.006342494715,0,0.002114164905,0.00422832981,"Monteiro, PB, Brazil",5,1,1,238
2510808,Patos,0,0,0,0,1,4,16,6,10,0,0,0,249,300,41.5,0,277,221,176,4,0,60,18,0,7,6,1,1,771,0.3592736706,0.2866407263,0.2282749676,0.005188067445,0,0.07782101167,0.0233463035,0,0.009079118029,0.007782101167,0.001297016861,0.001297016861,"Patos, PB, Brazil",9,2,1,37
2511202,Pedras de Fogo,0,0,0,0,0,0,10,7,0,1,0,0,6,70,1,13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Pedras de Fogo, PB, Brazil",1,1,0,18
2511301,Piancó,0,0,0,1,0,5,13,1,6,7,0,0,41,63,6.833333333,38,6,6,9,11,0,1,1,0,0,0,0,1,35,0.1714285714,0.1714285714,0.2571428571,0.3142857143,0,0.02857142857,0.02857142857,0,0,0,0,0.02857142857,"Piancó, PB, Brazil",6,1,1,33
2512101,Pombal,65,0,0,0,0,9,33,15,9,3,0,0,45,70,7.5,50,38,32,27,15,0,12,7,1,2,0,0,0,134,0.2835820896,0.2388059701,0.2014925373,0.1119402985,0,0.08955223881,0.05223880597,0.007462686567,0.01492537313,0,0,0,"Pombal, PB, Brazil",5,1,1,134
2512309,Princesa Isabel,0,0,0,0,2,6,19,21,4,0,0,0,76,60,12.66666667,44,25,18,16,18,0,3,0,0,0,0,0,0,80,0.3125,0.225,0.2,0.225,0,0.0375,0,0,0,0,0,0,"Princesa Isabel, PB, Brazil",4,1,1,52
2512507,Queimadas,0,3,1,12,18,32,18,7,6,3,0,0,57,125,9.5,25,33,15,48,20,0,7,1,0,0,0,0,1,125,0.264,0.12,0.384,0.16,0,0.056,0.008,0,0,0,0,0.008,"Queimadas, PB, Brazil",2,1,1,100
2512705,Remígio,0,0,2,4,6,0,2,1,0,0,0,0,5,40,0.8333333333,16,11,0,5,3,18,3,0,0,0,0,0,0,40,0.275,0,0.125,0.075,0.45,0.075,0,0,0,0,0,0,"Remígio, PB, Brazil",2,1,1,15
2513406,Santa Luzia,3,0,0,0,0,1,7,2,0,0,0,0,22,50,3.666666667,50,5,1,5,3,4,1,0,0,0,0,0,0,19,0.2631578947,0.05263157895,0.2631578947,0.1578947368,0.2105263158,0.05263157895,0,0,0,0,0,0,"Santa Luzia, PB, Brazil",4,1,1,13
2513703,Santa Rita,0,0,4,1,4,52,74,36,44,16,4,1,219,170,36.5,0,182,164,97,34,0,9,2,1,1,0,1,0,491,0.3706720978,0.33401222,0.1975560081,0.06924643585,0,0.0183299389,0.004073319756,0.002036659878,0.002036659878,0,0.002036659878,0,"Santa Rita, PB, Brazil",1,1,1,236
2514008,São João do Cariri,0,0,0,0,0,0,0,0,0,0,0,0,20,40,3.333333333,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São João do Cariri, PB, Brazil",3,1,0,0
2514503,São José de Piranhas,28,5,6,7,8,9,10,6,6,0,0,0,73,44,12.16666667,19,53,18,56,0,0,10,8,0,0,0,0,0,145,0.3655172414,0.124137931,0.3862068966,0,0,0.06896551724,0.05517241379,0,0,0,0,0,"São José de Piranhas, PB, Brazil",2,1,1,85
2515302,Sapé,26,0,0,0,5,29,51,17,29,8,3,0,84,87,14,0,59,60,28,22,0,6,3,0,0,0,0,0,178,0.3314606742,0.3370786517,0.1573033708,0.1235955056,0,0.03370786517,0.01685393258,0,0,0,0,0,"Sapé, PB, Brazil",3,1,1,168
2515500,Serra Branca,37,1,3,7,2,2,8,11,6,1,0,0,53,50,8.833333333,38,16,18,17,6,0,7,9,0,0,1,1,0,75,0.2133333333,0.24,0.2266666667,0.08,0,0.09333333333,0.12,0,0,0.01333333333,0.01333333333,0,"Serra Branca, PB, Brazil",4,1,1,78
2516003,Solânea,0,0,0,0,51,49,27,11,11,3,2,0,37,60,6.166666667,60,98,18,30,12,0,15,12,7,2,0,0,0,194,0.5051546392,0.09278350515,0.1546391753,0.0618556701,0,0.07731958763,0.0618556701,0.03608247423,0.01030927835,0,0,0,"Solânea, PB, Brazil",2,1,1,154
2516102,Soledade,45,0,0,1,0,5,10,2,1,0,0,0,45,64,7.5,45,24,7,20,16,0,6,3,2,2,2,0,0,82,0.2926829268,0.08536585366,0.243902439,0.1951219512,0,0.07317073171,0.03658536585,0.0243902439,0.0243902439,0.0243902439,0,0,"Soledade, PB, Brazil",4,2,1,64
2516201,Sousa,0,28,34,66,64,167,134,77,43,27,4,0,557,262,92.83333333,50,365,122,287,73,0,128,96,4,3,0,2,0,1080,0.337962963,0.112962963,0.2657407407,0.06759259259,0,0.1185185185,0.08888888889,0.003703703704,0.002777777778,0,0.001851851852,0,"Sousa, PB, Brazil",8,3,1,644
2516706,Teixeira,0,0,0,0,0,7,9,4,4,7,0,0,19,40,3.166666667,20,22,16,30,9,0,4,8,0,0,0,0,0,89,0.2471910112,0.1797752809,0.3370786517,0.1011235955,0,0.04494382022,0.08988764045,0,0,0,0,0,"Teixeira, PB, Brazil",5,1,1,31
2516904,Uiraúna,0,0,0,0,7,5,2,1,2,0,0,0,19,3,3.166666667,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Uiraúna, PB, Brazil",3,1,0,17
2517001,Umbuzeiro,0,0,0,4,3,8,6,2,3,0,1,0,4,28,0.6666666667,10,0,9,11,4,0,1,0,0,0,0,0,0,25,0,0.36,0.44,0.16,0,0.04,0,0,0,0,0,0,"Umbuzeiro, PB, Brazil",3,1,1,27
2600054,Abreu e Lima,0,0,0,0,0,0,0,0,0,0,0,0,3798,1138,633,946,3141,21,940,108,0,434,90,188,49,139,13,0,5123,0.6131173141,0.004099160648,0.1834862385,0.02108139762,0,0.08471598673,0.01756783135,0.03669724771,0.009564708179,0.02713253953,0.002537575639,0,"Abreu e Lima, PE, Brazil",1,3,1,0
2600104,Afogados da Ingazeira,0,0,0,0,0,0,0,0,0,0,0,0,0,24,0,24,10,13,13,0,0,0,0,0,0,0,0,0,36,0.2777777778,0.3611111111,0.3611111111,0,0,0,0,0,0,0,0,0,"Afogados da Ingazeira, PE, Brazil",2,1,1,0
2600203,Afrânio,0,0,0,0,0,0,0,0,0,0,0,0,5,50,0.8333333333,50,0,0,7,4,0,0,0,0,0,0,0,0,11,0,0,0.6363636364,0.3636363636,0,0,0,0,0,0,0,0,"Afrânio, PE, Brazil",2,1,1,0
2600302,Agrestina,0,0,0,0,0,0,0,0,0,0,0,0,12,32,2,32,12,4,4,2,0,0,0,0,0,0,0,0,22,0.5454545455,0.1818181818,0.1818181818,0.09090909091,0,0,0,0,0,0,0,0,"Agrestina, PE, Brazil",1,2,1,0
2600807,Altinho,0,0,0,0,0,0,0,0,0,0,0,0,0,30,0,30,1,0,0,1,0,0,3,0,0,0,0,0,5,0.2,0,0,0.2,0,0,0.6,0,0,0,0,0,"Altinho, PE, Brazil",1,1,1,0
2601102,Araripina,0,0,0,0,0,0,0,0,0,0,0,0,78,45,13,45,24,21,40,5,0,13,2,3,0,0,0,1,109,0.2201834862,0.1926605505,0.3669724771,0.04587155963,0,0.119266055,0.01834862385,0.02752293578,0,0,0,0.009174311927,"Araripina, PE, Brazil",1,1,1,0
2601201,Arcoverde,0,0,0,46,41,161,268,85,112,37,21,0,387,452,64.5,452,606,425,632,195,100,94,1,1,4,1,0,0,2059,0.2943176299,0.2064108791,0.306945119,0.09470616804,0.04856726566,0.04565322972,0.0004856726566,0.0004856726566,0.001942690627,0.0004856726566,0,0,"Arcoverde, PE, Brazil",1,2,1,771
2601904,Bezerros,0,0,0,0,0,0,0,0,0,0,0,0,27,30,4.5,30,1,11,3,1,0,0,0,0,0,0,0,0,16,0.0625,0.6875,0.1875,0.0625,0,0,0,0,0,0,0,0,"Bezerros, PE, Brazil",1,1,1,0
2602100,Bom Conselho,0,0,0,0,0,0,0,0,0,0,0,0,10,46,1.666666667,46,4,3,7,2,0,0,0,1,0,0,0,0,17,0.2352941176,0.1764705882,0.4117647059,0.1176470588,0,0,0,0.05882352941,0,0,0,0,"Bom Conselho, PE, Brazil",2,1,1,0
2602803,Buíque,0,0,0,1,2,21,30,19,22,4,2,1,110,107,18.33333333,107,131,248,91,12,0,12,9,4,0,0,0,0,507,0.258382643,0.4891518738,0.1794871795,0.02366863905,0,0.02366863905,0.01775147929,0.007889546351,0,0,0,0,"Buíque, PE, Brazil",1,1,1,102
2603108,Cachoeirinha,0,4,0,0,0,0,0,0,0,0,0,0,4,12,0.6666666667,12,1,0,2,1,0,0,0,0,0,0,0,0,4,0.25,0,0.5,0.25,0,0,0,0,0,0,0,0,"Cachoeirinha, PE, Brazil",1,1,1,4
2603702,Canhotinho,0,0,0,0,0,0,0,0,0,0,0,0,520,492,86.66666667,0,1280,1811,1153,393,0,170,40,4,18,0,30,3,4902,0.2611179111,0.3694410445,0.2352101183,0.08017135863,0,0.03467972256,0.008159934721,0.0008159934721,0.003671970624,0,0.00611995104,0.000611995104,"Canhotinho, PE, Brazil",1,2,1,0
2603900,Carnaíba,0,0,0,0,0,0,0,0,0,0,0,0,11,40,1.833333333,40,0,0,6,0,0,0,0,0,0,0,0,0,6,0,0,1,0,0,0,0,0,0,0,0,0,"Carnaíba, PE, Brazil",2,1,1,0
2604007,Carpina,0,0,0,0,0,0,0,0,0,0,0,0,61,74,10.16666667,74,54,37,31,2,0,12,0,3,1,0,2,0,142,0.3802816901,0.2605633803,0.2183098592,0.01408450704,0,0.08450704225,0,0.02112676056,0.007042253521,0,0.01408450704,0,"Carpina, PE, Brazil",2,2,1,0
2604106,Caruaru,0,0,0,0,0,0,0,0,0,0,0,0,540,774,90,0,1646,849,1274,210,0,302,174,92,32,16,5,0,4600,0.357826087,0.1845652174,0.2769565217,0.04565217391,0,0.06565217391,0.03782608696,0.02,0.006956521739,0.00347826087,0.001086956522,0,"Caruaru, PE, Brazil",1,1,1,0
2605202,Escada,0,0,0,0,0,0,0,0,0,0,0,0,13,47,2.166666667,47,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Escada, PE, Brazil",1,1,0,0
2605301,Exu,0,0,0,0,0,0,0,0,0,0,0,0,44,32,7.333333333,32,5,1,23,8,2,2,1,0,0,0,0,0,42,0.119047619,0.02380952381,0.5476190476,0.1904761905,0.04761904762,0.04761904762,0.02380952381,0,0,0,0,0,"Exu, PE, Brazil",1,2,1,0
2605608,Flores,0,0,0,0,0,0,0,0,0,0,0,0,8,7,1.333333333,7,0,0,3,6,2,0,0,0,0,0,0,0,11,0,0,0.2727272727,0.5454545455,0.1818181818,0,0,0,0,0,0,0,"Flores, PE, Brazil",2,1,1,0
2606002,Garanhuns,0,0,0,0,0,0,0,0,0,0,0,0,204,96,34,96,129,99,80,18,0,23,2,1,3,0,0,0,355,0.3633802817,0.2788732394,0.2253521127,0.05070422535,0,0.06478873239,0.005633802817,0.002816901408,0.008450704225,0,0,0,"Garanhuns, PE, Brazil",1,1,1,0
2606101,Glória do Goitá,0,0,0,0,0,0,0,0,0,0,0,0,4,48,0.6666666667,48,1,0,6,0,0,2,1,0,0,0,0,0,10,0.1,0,0.6,0,0,0.2,0.1,0,0,0,0,0,"Glória do Goitá, PE, Brazil",2,1,1,0
2606200,Goiana,0,0,0,0,0,0,0,0,0,0,0,0,127,100,21.16666667,100,35,75,46,1,0,19,2,10,0,0,0,1,189,0.1851851852,0.3968253968,0.2433862434,0.005291005291,0,0.1005291005,0.01058201058,0.05291005291,0,0,0,0.005291005291,"Goiana, PE, Brazil",1,2,1,0
2606408,Gravatá,0,0,0,0,0,0,0,0,0,0,0,0,54,68,9,68,29,23,14,0,0,0,0,0,0,0,0,0,66,0.4393939394,0.3484848485,0.2121212121,0,0,0,0,0,0,0,0,0,"Gravatá, PE, Brazil",1,1,1,0
2606606,Ibimirim,0,0,0,0,0,0,0,0,0,0,0,0,5,36,0.8333333333,36,1,5,1,1,0,0,0,0,0,0,0,0,8,0.125,0.625,0.125,0.125,0,0,0,0,0,0,0,0,"Ibimirim, PE, Brazil",1,1,1,0
2607307,Ipubi,0,0,0,0,0,0,0,0,0,0,0,0,33,18,5.5,18,5,27,35,7,0,8,0,0,0,0,0,0,82,0.06097560976,0.3292682927,0.4268292683,0.08536585366,0,0.09756097561,0,0,0,0,0,0,"Ipubi, PE, Brazil",1,1,1,0
2607604,Ilha de Itamaracá,111,6,4,21,32,90,42,103,85,72,50,0,1165,1757,194.1666667,0,1174,2145,1217,292,632,433,92,190,49,160,6,0,6390,0.1837245696,0.3356807512,0.1904538341,0.04569640063,0.09890453834,0.06776212833,0.01439749609,0.02973395931,0.007668231612,0.02503912363,0.0009389671362,0,"Ilha de Itamaracá, PE, Brazil",1,4,1,616
2607752,Itapissuma,0,0,0,0,0,0,0,0,0,0,0,0,1433,1226,238.8333333,1226,5672,3376,1295,543,2302,975,230,398,66,226,27,1,15111,0.3753557011,0.2234134075,0.08569915955,0.03593408775,0.1523393554,0.06452253325,0.01522070015,0.02633842896,0.004367679174,0.01495599232,0.001786777844,6.62E-05,"Itapissuma, PE, Brazil",1,2,1,0
2607802,Itaquitinga,0,0,0,0,0,0,0,0,0,0,0,0,496,1824,82.66666667,0,1490,42,2067,119,399,312,32,47,49,100,4,1,4662,0.3196053196,0.009009009009,0.4433719434,0.02552552553,0.08558558559,0.06692406692,0.006864006864,0.01008151008,0.01051051051,0.02145002145,0.000858000858,0.0002145002145,"Itaquitinga, PE, Brazil",1,3,1,0
2608800,Lajedo,0,0,0,0,0,0,0,0,0,0,0,0,43,105,7.166666667,105,59,18,23,5,0,15,1,1,0,0,0,0,122,0.4836065574,0.1475409836,0.1885245902,0.04098360656,0,0.1229508197,0.008196721311,0.008196721311,0,0,0,0,"Lajedo, PE, Brazil",1,1,1,0
2608909,Limoeiro,0,0,0,0,0,0,0,0,0,0,0,0,412,723,68.66666667,0,1895,1061,1596,212,0,406,47,158,20,82,5,1,5483,0.3456137151,0.1935072041,0.2910815247,0.03866496444,0,0.07404705453,0.008571949663,0.02881634142,0.003647638154,0.01495531643,0.0009119095386,0.0001823819077,"Limoeiro, PE, Brazil",1,2,1,0
2609006,Macaparana,0,0,0,0,0,0,0,0,0,0,0,0,8,50,1.333333333,50,1,11,13,5,0,0,0,0,0,0,0,0,30,0.03333333333,0.3666666667,0.4333333333,0.1666666667,0,0,0,0,0,0,0,0,"Macaparana, PE, Brazil",1,1,1,0
2609501,Nazaré da Mata,0,0,0,0,0,0,0,0,0,0,0,0,12,18,2,18,4,2,3,0,0,0,1,0,0,0,0,0,10,0.4,0.2,0.3,0,0,0,0.1,0,0,0,0,0,"Nazaré da Mata, PE, Brazil",1,1,1,0
2610004,Palmares,0,0,0,0,0,0,0,0,0,0,0,0,178,532,29.66666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Palmares, PE, Brazil",1,1,0,0
2610400,Parnamirim,0,0,0,0,0,0,0,0,0,0,0,0,7,7,1.166666667,7,1,0,4,1,0,0,0,0,0,0,0,0,6,0.1666666667,0,0.6666666667,0.1666666667,0,0,0,0,0,0,0,0,"Parnamirim, PE, Brazil",1,1,1,0
2610806,Pedra,0,0,0,0,0,0,0,0,0,0,0,0,3,36,0.5,36,2,4,6,3,0,1,0,0,0,0,0,0,16,0.125,0.25,0.375,0.1875,0,0.0625,0,0,0,0,0,0,"Pedra, PE, Brazil",1,1,1,0
2610905,Pesqueira,560,0,0,4,20,90,234,106,134,70,36,0,484,163,80.66666667,0,336,230,474,166,0,30,2,6,4,4,2,0,1254,0.2679425837,0.1834130781,0.3779904306,0.1323763955,0,0.02392344498,0.001594896332,0.004784688995,0.003189792663,0.003189792663,0.001594896332,0,"Pesqueira, PE, Brazil",1,2,1,1254
2611002,Petrolândia,0,0,0,0,0,0,0,0,0,0,0,0,60,12,10,12,7,7,24,8,1,4,0,1,0,0,0,0,52,0.1346153846,0.1346153846,0.4615384615,0.1538461538,0.01923076923,0.07692307692,0,0.01923076923,0,0,0,0,"Petrolândia, PE, Brazil",2,2,1,0
2611101,Petrolina,0,0,0,5,15,147,343,217,223,104,31,1,465,877,77.5,20,142,262,501,150,0,14,4,6,3,1,0,0,1083,0.1311172669,0.241920591,0.4626038781,0.1385041551,0,0.01292705448,0.003693444137,0.005540166205,0.002770083102,0.0009233610342,0,0,"Petrolina, PE, Brazil",1,2,1,1086
2611606,Recife,1,1082,18,106,229,705,628,278,231,131,57,2,297,9944,49.5,579,1653,1119,1069,149,1827,264,64,155,35,64,4,4,6407,0.2579990635,0.1746527236,0.1668487592,0.02325581395,0.2851568597,0.04120493211,0.00998907445,0.02419228968,0.00546277509,0.00998907445,0.0006243171531,0.0006243171531,"Recife, PE, Brazil",2,9,1,3468
2611705,Riacho das Almas,0,0,0,0,0,0,0,0,0,0,0,0,6,4,1,4,0,1,2,1,0,0,0,0,0,0,0,0,4,0,0.25,0.5,0.25,0,0,0,0,0,0,0,0,"Riacho das Almas, PE, Brazil",1,1,1,0
2611804,Ribeirão,0,0,5,0,0,0,0,0,0,0,0,0,1,35,0.1666666667,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ribeirão, PE, Brazil",1,1,0,5
2612208,Salgueiro,332,10,50,6,10,20,54,163,50,60,0,0,213,404,35.5,202,13,246,264,4,131,6,1,10,1,0,4,1,681,0.01908957416,0.3612334802,0.3876651982,0.005873715125,0.1923641703,0.008810572687,0.001468428781,0.01468428781,0.001468428781,0,0.005873715125,0.001468428781,"Salgueiro, PE, Brazil",1,1,1,755
2612307,Saloá,0,0,0,0,0,0,0,0,0,0,0,0,39,42,6.5,42,13,14,38,10,0,7,0,3,0,0,0,0,85,0.1529411765,0.1647058824,0.4470588235,0.1176470588,0,0.08235294118,0,0.03529411765,0,0,0,0,"Saloá, PE, Brazil",2,1,1,0
2612505,Santa Cruz do Capibaribe,0,0,0,0,0,0,0,0,0,0,0,0,151,228,25.16666667,228,537,265,344,49,0,96,56,19,2,4,2,0,1374,0.3908296943,0.19286754,0.250363901,0.03566229985,0,0.06986899563,0.04075691412,0.01382823872,0.001455604076,0.002911208151,0.001455604076,0,"Santa Cruz do Capibaribe, PE, Brazil",1,1,1,0
2613305,São Joaquim do Monte,0,0,0,0,0,0,0,0,0,0,0,0,6,24,1,24,4,4,4,2,0,0,0,0,0,0,0,0,14,0.2857142857,0.2857142857,0.2857142857,0.1428571429,0,0,0,0,0,0,0,0,"São Joaquim do Monte, PE, Brazil",1,1,1,0
2613503,São José do Belmonte,0,0,0,0,0,0,0,0,0,0,0,0,13,30,2.166666667,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São José do Belmonte, PE, Brazil",1,1,0,0
2613602,São José do Egito,0,0,0,0,0,0,0,0,0,0,0,0,38,25,6.333333333,25,8,8,12,2,0,0,0,0,0,0,0,0,30,0.2666666667,0.2666666667,0.4,0.06666666667,0,0,0,0,0,0,0,0,"São José do Egito, PE, Brazil",2,1,1,0
2613909,Serra Talhada,0,0,0,0,0,0,0,0,0,0,0,0,129,50,21.5,50,42,26,78,15,0,27,0,1,0,0,0,0,189,0.2222222222,0.1375661376,0.4126984127,0.07936507937,0,0.1428571429,0,0.005291005291,0,0,0,0,"Serra Talhada, PE, Brazil",1,2,1,0
2614105,Sertânia,0,0,0,0,0,0,0,0,0,0,0,0,5,35,0.8333333333,35,2,2,3,0,0,0,0,0,0,0,0,0,7,0.2857142857,0.2857142857,0.4285714286,0,0,0,0,0,0,0,0,0,"Sertânia, PE, Brazil",1,1,1,0
2614303,Moreilândia,0,0,0,0,0,0,0,0,0,0,0,0,10,30,1.666666667,30,0,0,10,0,2,0,0,0,0,0,0,0,12,0,0,0.8333333333,0,0.1666666667,0,0,0,0,0,0,0,"Moreilândia, PE, Brazil",1,1,1,0
2614600,Tabira,0,0,0,0,0,0,0,0,0,0,0,0,19,48,3.166666667,48,9,6,5,2,0,0,0,0,0,0,0,0,22,0.4090909091,0.2727272727,0.2272727273,0.09090909091,0,0,0,0,0,0,0,0,"Tabira, PE, Brazil",2,1,1,0
2614709,Tacaimbó,0,0,0,0,0,0,0,0,0,0,0,0,302,676,50.33333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Tacaimbó, PE, Brazil",1,1,0,0
2615300,Timbaúba,0,0,0,0,0,0,0,0,0,0,0,0,8,25,1.333333333,25,2,6,2,0,0,0,0,0,0,0,0,0,10,0.2,0.6,0.2,0,0,0,0,0,0,0,0,0,"Timbaúba, PE, Brazil",1,1,1,0
2615904,Tuparetama,0,0,0,0,0,0,0,0,0,0,0,0,6,18,1,18,3,2,3,0,0,0,0,0,0,0,0,0,8,0.375,0.25,0.375,0,0,0,0,0,0,0,0,0,"Tuparetama, PE, Brazil",2,1,1,0
2616001,Venturosa,0,0,0,0,0,0,0,0,0,0,0,0,0,36,0,36,2,0,3,1,0,0,0,0,0,0,0,0,6,0.3333333333,0,0.5,0.1666666667,0,0,0,0,0,0,0,0,"Venturosa, PE, Brazil",1,1,1,0
2616100,Verdejante,0,0,0,0,0,0,0,0,0,0,0,0,45,25,7.5,25,9,18,16,1,0,2,3,0,0,0,0,0,49,0.1836734694,0.3673469388,0.3265306122,0.02040816327,0,0.04081632653,0.0612244898,0,0,0,0,0,"Verdejante, PE, Brazil",1,2,1,0
2616308,Vicência,0,0,0,0,0,0,0,0,0,0,0,0,4,36,0.6666666667,36,1,5,3,3,0,0,0,0,0,0,0,0,12,0.08333333333,0.4166666667,0.25,0.25,0,0,0,0,0,0,0,0,"Vicência, PE, Brazil",1,1,1,0
2616407,Vitória de Santo Antão,0,0,0,1,5,27,31,15,8,4,3,0,235,157,39.16666667,157,153,213,238,31,0,54,9,15,5,0,2,1,721,0.2122052705,0.2954230236,0.3300970874,0.04299583911,0,0.07489597781,0.01248266297,0.02080443828,0.00693481276,0,0.002773925104,0.001386962552,"Vitória de Santo Antão, PE, Brazil",1,2,1,94
2700300,Arapiraca,0,0,0,0,0,0,0,0,0,0,0,0,102,48,17,48,1,4,7,2,0,0,2,0,0,0,0,0,16,0.0625,0.25,0.4375,0.125,0,0,0.125,0,0,0,0,0,"Arapiraca, AL, Brazil",1,1,1,0
2702900,Girau do Ponciano,0,0,4,14,11,100,149,97,116,68,35,8,526,960,87.66666667,960,306,344,385,44,0,205,2,48,6,10,2,1,1353,0.2261640798,0.2542498152,0.2845528455,0.0325203252,0,0.1515151515,0.0014781966,0.0354767184,0.0044345898,0.007390983001,0.0014781966,0.0007390983001,"Girau do Ponciano, AL, Brazil",2,1,1,602
2704302,Maceió,2037,3,8,121,294,3508,2076,848,830,381,118,6,5468,4671,911.3333333,1714,4597,4675,3608,1638,9,1219,203,760,52,28,20,6,16815,0.273386857,0.2780255724,0.2145703241,0.09741302409,0.0005352363961,0.07249479631,0.01207255427,0.04519774011,0.003092476955,0.001665179899,0.001189414213,0.000356824264,"Maceió, AL, Brazil",1,11,1,10230
2800308,Aracaju,0,0,0,0,217,298,129,28,20,6,5,1,1620,2350,270,180,713,854,678,92,200,145,16,40,19,1,1,0,2759,0.2584269663,0.3095324393,0.2457412106,0.03334541501,0.07249003262,0.05255527365,0.00579920261,0.01449800652,0.006886553099,0.0003624501631,0.0003624501631,0,"Aracaju, SE, Brazil",1,3,1,704
2802106,Estância,0,0,0,0,0,0,0,0,0,0,0,0,144,245,24,245,164,195,69,9,0,37,0,0,2,0,0,0,476,0.3445378151,0.4096638655,0.1449579832,0.01890756303,0,0.07773109244,0,0,0.004201680672,0,0,0,"Estância, SE, Brazil",1,1,1,0
2803609,Laranjeiras,235,4,5,24,25,278,308,162,221,110,4,0,777,1122,129.5,490,549,579,298,109,0,47,0,2,4,0,0,0,1588,0.3457178841,0.3646095718,0.1876574307,0.06863979849,0,0.02959697733,0,0.001259445844,0.002518891688,0,0,0,"Laranjeiras, SE, Brazil",2,2,1,1376
2804508,Nossa Senhora da Glória,0,0,0,0,17,58,114,84,42,36,5,0,86,201,14.33333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Nossa Senhora da Glória, SE, Brazil",3,1,0,356
2804805,Nossa Senhora do Socorro,74,0,0,0,0,19,29,9,7,5,0,0,536,335,89.33333333,260,209,120,50,8,17,19,4,1,4,0,0,0,432,0.4837962963,0.2777777778,0.1157407407,0.01851851852,0.03935185185,0.04398148148,0.009259259259,0.002314814815,0.009259259259,0,0,0,"Nossa Senhora do Socorro, SE, Brazil",1,2,1,143
2806701,São Cristóvão,0,0,0,0,0,0,0,0,0,0,0,0,899,800,149.8333333,720,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São Cristóvão, SE, Brazil",1,1,0,0
2807402,Tobias Barreto,0,0,0,0,87,146,150,70,60,20,8,0,170,346,28.33333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Tobias Barreto, SE, Brazil",1,1,0,541
2903201,Barreiras,0,0,2,3,4,8,109,39,24,9,3,0,456,533,76,195,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Barreiras, BA, Brazil",2,1,0,201
2904605,Brumado,0,0,0,4,22,100,40,30,2,0,0,0,187,467,31.16666667,261,247,130,453,166,3,45,0,2,0,0,0,0,1046,0.2361376673,0.1242829828,0.4330783939,0.1586998088,0.002868068834,0.0430210325,0,0.001912045889,0,0,0,0,"Brumado, BA, Brazil",3,1,1,198
2910727,Eunápolis,0,0,0,0,0,0,0,0,0,0,0,0,181,457,30.16666667,228,241,365,221,44,0,69,30,23,2,0,2,0,997,0.2417251755,0.3660982949,0.221664995,0.04413239719,0,0.06920762287,0.03009027081,0.02306920762,0.002006018054,0,0.002006018054,0,"Eunápolis, BA, Brazil",1,1,1,0
2910800,Feira De Santana,0,0,1,2,28,228,276,133,127,390,15,2,606,1280,101,730,576,457,484,335,0,19,0,2,2,1,0,0,1876,0.3070362473,0.2436034115,0.2579957356,0.1785714286,0,0.01012793177,0,0.001066098081,0.001066098081,0.0005330490405,0,0,"Feira De Santana, BA, Brazil",1,1,1,1202
2913606,Ilhéus,0,0,0,0,0,0,0,0,0,0,0,0,0,80,0,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ilhéus, BA, Brazil",1,1,0,0
2914604,Irecê,0,0,0,255,0,0,0,0,0,0,0,0,0,467,0,213,235,151,356,130,10,6,16,9,1,2,1,0,917,0.2562704471,0.1646673937,0.3882224646,0.1417666303,0.01090512541,0.006543075245,0.01744820065,0.009814612868,0.001090512541,0.002181025082,0.001090512541,0,"Irecê, BA, Brazil",1,1,1,255
2914802,Itabuna,0,0,0,0,0,0,0,0,0,0,0,0,473,670,78.83333333,258,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Itabuna, BA, Brazil",3,1,0,0
2918001,Jequié,13,0,0,3,1,35,73,43,65,30,7,1,202,416,33.66666667,112,102,121,142,63,0,30,10,1,2,0,1,0,472,0.2161016949,0.2563559322,0.3008474576,0.1334745763,0,0.06355932203,0.02118644068,0.002118644068,0.004237288136,0,0.002118644068,0,"Jequié, BA, Brazil",3,1,1,271
2918407,Juazeiro,0,0,0,0,0,81,203,116,88,42,10,0,586,756,97.66666667,408,272,270,336,85,0,65,17,13,1,0,0,0,1059,0.2568460812,0.2549575071,0.3172804533,0.08026440038,0,0.06137865911,0.01605288008,0.01227573182,0.0009442870633,0,0,0,"Juazeiro, BA, Brazil",1,1,1,540
2919207,Lauro De Freitas,0,0,0,0,0,0,0,0,0,0,0,0,294,430,49,0,238,300,62,19,0,69,1,0,0,0,0,0,689,0.3454281567,0.435413643,0.08998548621,0.02757619739,0,0.1001451379,0.00145137881,0,0,0,0,0,"Lauro De Freitas, BA, Brazil",1,1,1,0
2924009,Paulo Afonso,0,0,0,0,0,0,0,0,0,0,0,0,384,410,64,288,306,302,416,156,0,97,24,23,6,0,1,0,1331,0.2299023291,0.2268970699,0.3125469572,0.1172051089,0,0.07287753569,0.01803155522,0.01728024042,0.004507888805,0,0.0007513148009,0,"Paulo Afonso, BA, Brazil",2,1,1,0
2927408,Salvador,1237,16,37,29,76,346,518,304,263,95,40,6,3258,5787,543,1092,1951,2098,1452,505,713,332,62,50,27,8,5,0,7203,0.2708593642,0.2912675274,0.2015826739,0.07010967652,0.09898653339,0.04609190615,0.008607524643,0.006941552131,0.003748438151,0.001110648341,0.0006941552131,0,"Salvador, BA, Brazil",2,10,1,2967
2930501,Serrinha,121,0,0,0,0,11,10,11,20,5,2,0,96,476,16,144,27,69,52,9,0,22,0,1,0,0,0,0,180,0.15,0.3833333333,0.2888888889,0.05,0,0.1222222222,0,0.005555555556,0,0,0,0,"Serrinha, BA, Brazil",3,1,1,180
2930709,Simões Filho,46,0,0,0,0,88,20,0,0,0,0,0,127,220,21.16666667,0,119,79,24,15,0,25,0,0,0,0,0,0,262,0.4541984733,0.3015267176,0.09160305344,0.0572519084,0,0.09541984733,0,0,0,0,0,0,"Simões Filho, BA, Brazil",1,1,1,154
2931350,Teixeira De Freitas,0,0,0,0,0,47,68,27,22,31,6,0,6,316,1,138,119,172,234,63,0,2,2,0,0,4,0,0,596,0.1996644295,0.288590604,0.3926174497,0.105704698,0,0.003355704698,0.003355704698,0,0,0.006711409396,0,0,"Teixeira De Freitas, BA, Brazil",1,1,1,201
2932903,Valença,0,0,0,0,0,0,0,0,0,0,0,0,227,268,37.83333333,268,50,51,233,20,0,48,0,2,2,0,0,0,406,0.1231527094,0.1256157635,0.5738916256,0.04926108374,0,0.118226601,0,0.004926108374,0.004926108374,0,0,0,"Valença, BA, Brazil",3,1,1,0
2933307,Vitória Da Conquista,0,0,0,0,0,187,0,0,0,0,0,0,617,779,102.8333333,384,50,96,28,17,0,0,0,0,0,0,0,0,191,0.2617801047,0.502617801,0.1465968586,0.0890052356,0,0,0,0,0,0,0,0,"Vitória Da Conquista, BA, Brazil",1,2,1,187
3100203,Abaeté,0,0,0,0,0,0,0,0,0,0,0,0,13,112,2.166666667,94,50,43,47,9,59,5,0,0,0,0,0,0,213,0.234741784,0.2018779343,0.220657277,0.04225352113,0.2769953052,0.0234741784,0,0,0,0,0,0,"Abaeté, MG, Brazil",3,1,1,0
3100302,Abre Campo,0,0,0,1,0,8,41,16,18,2,0,0,171,138,28.5,84,45,55,62,37,0,6,0,0,0,1,0,0,206,0.2184466019,0.2669902913,0.3009708738,0.1796116505,0,0.02912621359,0,0,0,0.004854368932,0,0,"Abre Campo, MG, Brazil",6,1,1,86
3100500,Açucena,0,0,0,0,0,0,0,0,0,0,0,0,0,88,0,30,28,45,30,6,18,3,1,0,0,1,0,0,132,0.2121212121,0.3409090909,0.2272727273,0.04545454545,0.1363636364,0.02272727273,0.007575757576,0,0,0.007575757576,0,0,"Açucena, MG, Brazil",3,1,1,0
3100906,Águas Formosas,0,0,0,0,0,0,0,0,0,0,0,0,72,58,12,20,5,14,33,5,48,2,2,0,0,0,0,0,109,0.04587155963,0.128440367,0.3027522936,0.04587155963,0.4403669725,0.01834862385,0.01834862385,0,0,0,0,0,"Águas Formosas, MG, Brazil",7,1,1,0
3101102,Aimorés,0,0,2,0,9,35,28,13,9,3,0,0,33,84,5.5,84,58,114,31,5,1,11,11,0,0,0,1,0,232,0.25,0.4913793103,0.1336206897,0.02155172414,0.004310344828,0.0474137931,0.0474137931,0,0,0,0.004310344828,0,"Aimorés, MG, Brazil",1,1,1,99
3101508,Além Paraíba,0,0,0,0,0,57,23,25,0,0,0,0,39,166,6.5,59,73,59,21,12,0,0,0,0,0,0,0,0,165,0.4424242424,0.3575757576,0.1272727273,0.07272727273,0,0,0,0,0,0,0,0,"Além Paraíba, MG, Brazil",3,1,1,105
3101607,Alfenas,0,0,0,0,0,0,0,0,0,0,0,0,113,318,18.83333333,176,56,115,66,6,113,6,3,2,1,0,0,0,368,0.152173913,0.3125,0.1793478261,0.01630434783,0.3070652174,0.01630434783,0.008152173913,0.005434782609,0.002717391304,0,0,0,"Alfenas, MG, Brazil",2,2,1,0
3101706,Almenara,0,0,0,0,0,0,0,0,0,0,0,0,19,203,3.166666667,60,53,73,122,11,37,9,1,0,0,0,0,0,306,0.1732026144,0.2385620915,0.3986928105,0.03594771242,0.1209150327,0.02941176471,0.003267973856,0,0,0,0,0,"Almenara, MG, Brazil",7,2,1,0
3102605,Andradas,0,0,0,0,0,0,0,0,0,0,0,0,16,85,2.666666667,83,0,0,0,0,27,0,1,0,0,0,0,0,28,0,0,0,0,0.9642857143,0,0.03571428571,0,0,0,0,0,"Andradas, MG, Brazil",2,1,1,0
3103405,Araçuaí,0,0,0,1,7,17,29,4,1,2,0,0,151,63,25.16666667,63,23,14,42,15,43,3,1,0,0,1,0,0,142,0.161971831,0.0985915493,0.2957746479,0.1056338028,0.3028169014,0.02112676056,0.007042253521,0,0,0.007042253521,0,0,"Araçuaí, MG, Brazil",6,1,1,61
3103504,Araguari,0,0,0,0,0,0,0,0,0,0,0,0,32,291,5.333333333,117,115,120,74,20,42,3,4,0,0,0,0,0,378,0.3042328042,0.3174603175,0.1957671958,0.05291005291,0.1111111111,0.007936507937,0.01058201058,0,0,0,0,0,"Araguari, MG, Brazil",2,1,1,0
3104007,Araxá,0,0,0,0,0,0,0,0,0,0,0,0,71,396,11.83333333,186,117,112,63,20,77,1,3,1,0,0,0,2,396,0.2954545455,0.2828282828,0.1590909091,0.05050505051,0.1944444444,0.002525252525,0.007575757576,0.002525252525,0,0,0,0.005050505051,"Araxá, MG, Brazil",2,2,1,0
3104205,Arcos,0,0,0,0,0,0,0,0,0,0,0,0,61,261,10.16666667,101,79,76,40,15,62,0,3,0,0,0,0,0,275,0.2872727273,0.2763636364,0.1454545455,0.05454545455,0.2254545455,0,0.01090909091,0,0,0,0,0,"Arcos, MG, Brazil",2,2,1,0
3105400,Barão de Cocais,0,0,0,0,0,0,0,0,0,0,0,0,117,67,19.5,64,42,37,25,9,27,3,6,0,0,0,0,0,149,0.2818791946,0.2483221477,0.1677852349,0.06040268456,0.1812080537,0.02013422819,0.04026845638,0,0,0,0,0,"Barão de Cocais, MG, Brazil",2,1,1,0
3105608,Barbacena,0,4,0,9,7,45,28,10,4,0,0,0,93,389,15.5,347,115,74,121,25,119,5,4,0,4,0,2,0,469,0.2452025586,0.157782516,0.2579957356,0.05330490405,0.2537313433,0.01066098081,0.008528784648,0,0.008528784648,0,0.004264392324,0,"Barbacena, MG, Brazil",12,2,1,107
3106200,Belo Horizonte,0,0,0,0,0,0,0,0,0,0,0,0,919,14176,153.1666667,665,365,310,173,16,200,28,12,5,6,3,4,0,1122,0.325311943,0.2762923351,0.1541889483,0.01426024955,0.1782531194,0.02495543672,0.01069518717,0.004456327986,0.005347593583,0.002673796791,0.003565062389,0,"Belo Horizonte, MG, Brazil",1,5,1,0
3106705,Betim,785,0,1,9,3,3,4,0,0,0,0,0,1604,545,267.3333333,404,269,231,191,32,149,36,4,4,3,5,5,0,929,0.2895586652,0.2486544672,0.2055974166,0.03444564047,0.1603875135,0.03875134553,0.004305705059,0.004305705059,0.003229278794,0.005382131324,0.005382131324,0,"Betim, MG, Brazil",1,2,1,805
3106903,Bicas,0,0,0,0,0,0,0,0,0,0,0,0,63,101,10.5,99,55,45,21,9,55,1,1,0,0,2,0,0,189,0.291005291,0.2380952381,0.1111111111,0.04761904762,0.291005291,0.005291005291,0.005291005291,0,0,0.01058201058,0,0,"Bicas, MG, Brazil",4,1,1,0
3107109,Boa Esperança,0,0,0,0,0,0,0,0,0,0,0,0,101,109,16.83333333,109,37,34,35,10,28,3,0,0,0,1,0,0,148,0.25,0.2297297297,0.2364864865,0.06756756757,0.1891891892,0.02027027027,0,0,0,0.006756756757,0,0,"Boa Esperança, MG, Brazil",3,1,1,0
3107307,Bocaiúva,0,0,0,0,0,0,0,0,0,0,0,0,46,60,7.666666667,57,28,52,0,5,60,0,2,0,0,0,0,0,147,0.1904761905,0.3537414966,0,0.03401360544,0.4081632653,0,0.01360544218,0,0,0,0,0,"Bocaiúva, MG, Brazil",5,1,1,0
3107406,Bom Despacho,0,0,0,0,0,0,0,0,0,0,0,0,0,75,0,0,35,81,24,6,28,1,0,0,0,0,0,0,175,0.2,0.4628571429,0.1371428571,0.03428571429,0.16,0.005714285714,0,0,0,0,0,0,"Bom Despacho, MG, Brazil",2,1,1,0
3109303,Buritis,0,0,0,0,0,0,0,0,0,0,0,0,148,115,24.66666667,66,34,15,35,8,26,4,2,0,1,0,0,0,125,0.272,0.12,0.28,0.064,0.208,0.032,0.016,0,0.008,0,0,0,"Buritis, MG, Brazil",2,1,1,0
3110004,Caeté,0,0,0,0,0,0,0,0,0,0,0,0,4,49,0.6666666667,49,23,23,28,0,14,3,2,0,0,0,0,0,93,0.247311828,0.247311828,0.3010752688,0,0.1505376344,0.03225806452,0.02150537634,0,0,0,0,0,"Caeté, MG, Brazil",3,1,1,0
3111200,Campo Belo,0,0,0,0,0,0,0,0,0,0,0,0,154,205,25.66666667,42,90,139,46,8,56,4,1,0,0,1,0,0,345,0.2608695652,0.4028985507,0.1333333333,0.0231884058,0.1623188406,0.0115942029,0.002898550725,0,0,0.002898550725,0,0,"Campo Belo, MG, Brazil",4,2,1,0
3111606,Campos Gerais,0,0,0,0,0,0,0,0,0,0,0,0,133,110,22.16666667,57,49,71,23,18,22,0,1,0,0,0,0,0,184,0.2663043478,0.3858695652,0.125,0.09782608696,0.1195652174,0,0.005434782609,0,0,0,0,0,"Campos Gerais, MG, Brazil",2,1,1,0
3111804,Canápolis,0,0,0,0,0,0,0,0,0,0,0,0,94,97,15.66666667,31,35,18,11,16,0,2,0,0,0,0,0,0,82,0.4268292683,0.2195121951,0.1341463415,0.1951219512,0,0.0243902439,0,0,0,0,0,0,"Canápolis, MG, Brazil",2,1,1,0
3112307,Capelinha,0,0,0,0,0,0,0,0,0,0,0,0,246,95,41,66,40,42,12,45,0,4,2,1,1,0,0,0,147,0.2721088435,0.2857142857,0.08163265306,0.306122449,0,0.02721088435,0.01360544218,0.006802721088,0.006802721088,0,0,0,"Capelinha, MG, Brazil",3,1,1,0
3113305,Carangola,0,0,1,4,10,46,18,8,2,0,0,0,175,132,29.16666667,120,45,48,42,6,14,4,0,0,0,0,0,0,159,0.2830188679,0.3018867925,0.2641509434,0.03773584906,0.08805031447,0.0251572327,0,0,0,0,0,0,"Carangola, MG, Brazil",4,1,1,89
3113404,Caratinga,0,0,0,0,0,0,0,0,0,0,0,0,137,464,22.83333333,214,135,122,91,12,210,13,5,0,0,0,1,0,589,0.2292020374,0.2071307301,0.1544991511,0.02037351443,0.3565365025,0.0220713073,0.008488964346,0,0,0,0.001697792869,0,"Caratinga, MG, Brazil",11,2,1,0
3113701,Carlos Chagas,0,0,0,0,0,9,17,6,5,1,0,0,1,48,0.1666666667,45,27,83,38,7,45,130,26,10,3,0,0,0,369,0.07317073171,0.2249322493,0.1029810298,0.0189701897,0.1219512195,0.352303523,0.07046070461,0.027100271,0.008130081301,0,0,0,"Carlos Chagas, MG, Brazil",1,1,1,38
3114303,Carmo do Paranaíba,0,0,0,0,0,0,0,0,0,0,0,0,120,229,20,24,150,84,86,23,82,6,15,0,1,0,0,0,447,0.3355704698,0.1879194631,0.192393736,0.0514541387,0.1834451902,0.01342281879,0.03355704698,0,0.002237136465,0,0,0,"Carmo do Paranaíba, MG, Brazil",1,1,1,0
3115300,Cataguases,0,0,0,0,0,0,0,0,0,0,0,0,113,134,18.83333333,97,71,71,57,3,52,4,0,0,0,2,0,0,260,0.2730769231,0.2730769231,0.2192307692,0.01153846154,0.2,0.01538461538,0,0,0,0.007692307692,0,0,"Cataguases, MG, Brazil",5,1,1,0
3117306,Conceição das Alagoas,0,0,0,0,0,0,0,0,0,0,0,0,50,84,8.333333333,0,0,0,1,0,75,0,0,0,0,0,0,0,76,0,0,0.01315789474,0,0.9868421053,0,0,0,0,0,0,0,"Conceição das Alagoas, MG, Brazil",2,1,1,0
3118007,Congonhas,0,0,0,0,0,0,0,0,0,0,0,0,29,146,4.833333333,102,48,71,42,11,79,4,2,0,0,0,1,0,258,0.1860465116,0.2751937984,0.1627906977,0.04263565891,0.3062015504,0.01550387597,0.007751937984,0,0,0,0.003875968992,0,"Congonhas, MG, Brazil",1,1,1,0
3118304,Conselheiro Lafaiete,0,0,0,0,0,0,0,0,0,0,0,0,58,368,9.666666667,110,119,170,101,30,94,23,12,2,3,2,0,0,556,0.214028777,0.3057553957,0.1816546763,0.05395683453,0.1690647482,0.04136690647,0.02158273381,0.003597122302,0.005395683453,0.003597122302,0,0,"Conselheiro Lafaiete, MG, Brazil",9,3,1,0
3118403,Conselheiro Pena,0,0,0,0,0,0,0,0,0,0,0,0,106,85,17.66666667,37,16,34,35,7,12,1,5,0,0,0,0,0,110,0.1454545455,0.3090909091,0.3181818182,0.06363636364,0.1090909091,0.009090909091,0.04545454545,0,0,0,0,0,"Conselheiro Pena, MG, Brazil",5,1,1,0
3118601,Contagem,0,0,0,0,0,0,0,0,0,0,0,0,1825,1759,304.1666667,236,693,804,361,0,731,62,23,4,17,4,15,0,2714,0.2553426676,0.2962417097,0.1330140015,0,0.2693441415,0.02284450995,0.008474576271,0.001473839352,0.006263817244,0.001473839352,0.005526897568,0,"Contagem, MG, Brazil",1,2,1,0
3119104,Corinto,0,0,0,0,0,0,0,0,0,0,0,0,68,78,11.33333333,65,41,31,32,0,38,2,2,0,0,0,0,0,146,0.2808219178,0.2123287671,0.2191780822,0,0.2602739726,0.01369863014,0.01369863014,0,0,0,0,0,"Corinto, MG, Brazil",2,1,1,0
3119302,Coromandel,98,1,4,8,7,7,3,3,0,0,0,2,75,132,12.5,80,73,37,42,8,14,1,2,0,0,0,0,0,177,0.4124293785,0.209039548,0.2372881356,0.04519774011,0.0790960452,0.005649717514,0.01129943503,0,0,0,0,0,"Coromandel, MG, Brazil",2,1,1,133
3119401,Coronel Fabriciano,0,0,0,0,0,0,0,0,0,0,0,0,247,213,41.16666667,92,0,0,0,0,166,0,1,0,0,0,0,0,167,0,0,0,0,0.994011976,0,0.005988023952,0,0,0,0,0,"Coronel Fabriciano, MG, Brazil",2,1,1,0
3120904,Curvelo,0,0,0,0,0,0,0,0,0,0,0,0,389,112,64.83333333,60,45,47,40,15,76,8,6,1,0,0,0,0,238,0.1890756303,0.1974789916,0.1680672269,0.06302521008,0.3193277311,0.03361344538,0.02521008403,0.004201680672,0,0,0,0,"Curvelo, MG, Brazil",5,1,1,0
3121605,Diamantina,0,0,0,0,0,0,0,0,0,0,0,0,62,115,10.33333333,113,32,70,39,22,28,2,3,0,0,0,0,0,196,0.1632653061,0.3571428571,0.1989795918,0.112244898,0.1428571429,0.01020408163,0.01530612245,0,0,0,0,0,"Diamantina, MG, Brazil",9,1,1,0
3122306,Divinópolis,0,0,0,0,0,0,0,0,0,0,0,0,127,543,21.16666667,435,348,315,113,21,19,32,33,19,5,0,3,0,908,0.3832599119,0.3469162996,0.1244493392,0.0231277533,0.02092511013,0.03524229075,0.03634361233,0.02092511013,0.00550660793,0,0.003303964758,0,"Divinópolis, MG, Brazil",1,2,1,0
3124005,Ervália,0,0,0,0,0,0,0,0,0,0,0,0,50,142,8.333333333,142,43,45,18,4,73,6,7,1,0,0,0,0,197,0.2182741117,0.2284263959,0.09137055838,0.02030456853,0.3705583756,0.03045685279,0.03553299492,0.005076142132,0,0,0,0,"Ervália, MG, Brazil",2,1,1,0
3124906,Eugenópolis,0,0,0,0,0,0,0,0,0,0,0,0,47,114,7.833333333,60,36,52,35,13,18,3,0,0,0,0,0,0,157,0.2292993631,0.3312101911,0.2229299363,0.08280254777,0.1146496815,0.01910828025,0,0,0,0,0,0,"Eugenópolis, MG, Brazil",3,1,1,0
3125101,Extrema,0,0,0,0,0,0,0,0,0,0,0,0,155,43,25.83333333,40,15,27,23,0,9,1,2,1,0,0,0,0,78,0.1923076923,0.3461538462,0.2948717949,0,0.1153846154,0.01282051282,0.02564102564,0.01282051282,0,0,0,0,"Extrema, MG, Brazil",2,1,1,0
3126109,Formiga,0,0,0,0,0,0,0,0,0,0,0,0,340,396,56.66666667,31,154,206,103,18,302,18,8,4,5,0,2,0,820,0.187804878,0.2512195122,0.1256097561,0.02195121951,0.3682926829,0.02195121951,0.009756097561,0.00487804878,0.006097560976,0,0.00243902439,0,"Formiga, MG, Brazil",3,1,1,0
3126703,Francisco Sá,0,0,0,0,0,0,0,0,0,0,0,0,85,332,14.16666667,0,73,61,52,2,169,17,3,7,11,0,4,4,403,0.1811414392,0.1513647643,0.1290322581,0.004962779156,0.4193548387,0.04218362283,0.007444168734,0.01736972705,0.02729528536,0,0.009925558313,0.009925558313,"Francisco Sá, MG, Brazil",2,1,1,0
3127107,Frutal,0,0,0,0,0,0,0,0,0,0,0,0,321,486,53.5,0,165,117,98,24,132,2,13,1,2,0,0,0,554,0.297833935,0.2111913357,0.1768953069,0.04332129964,0.238267148,0.003610108303,0.02346570397,0.001805054152,0.003610108303,0,0,0,"Frutal, MG, Brazil",4,3,1,0
3127701,Governador Valadares,0,0,0,0,0,0,0,0,0,0,0,0,607,1150,101.1666667,361,369,540,385,71,563,63,38,6,8,2,0,0,2045,0.1804400978,0.2640586797,0.1882640587,0.03471882641,0.2753056235,0.03080684597,0.01858190709,0.00293398533,0.00391198044,0.00097799511,0,0,"Governador Valadares, MG, Brazil",6,3,1,0
3128006,Guanhães,3,0,1,1,1,18,22,7,3,1,0,0,277,214,46.16666667,101,50,21,12,16,37,11,2,4,3,0,0,2,158,0.3164556962,0.1329113924,0.07594936709,0.1012658228,0.2341772152,0.06962025316,0.01265822785,0.0253164557,0.01898734177,0,0,0.01265822785,"Guanhães, MG, Brazil",3,2,1,57
3128303,Guaranésia,0,0,0,0,0,0,0,0,0,0,0,0,37,170,6.166666667,122,137,95,61,10,91,2,5,0,2,0,0,0,403,0.3399503722,0.2357320099,0.1513647643,0.02481389578,0.2258064516,0.004962779156,0.01240694789,0,0.004962779156,0,0,0,"Guaranésia, MG, Brazil",1,1,1,0
3129806,Ibirité,0,0,0,0,0,21,34,13,8,2,0,0,212,102,35.33333333,80,37,88,26,0,0,13,8,0,0,0,1,0,173,0.2138728324,0.5086705202,0.1502890173,0,0,0.07514450867,0.04624277457,0,0,0,0.005780346821,0,"Ibirité, MG, Brazil",3,1,1,78
3130101,Igarapé,131,0,0,1,6,310,968,490,378,132,120,0,3787,1986,631.1666667,445,1436,1442,593,12,667,46,15,26,32,0,5,0,4274,0.3359850257,0.3373888629,0.1387459055,0.00280767431,0.1560598971,0.01076275152,0.003509592887,0.006083294338,0.007487131493,0,0.001169864296,0,"Igarapé, MG, Brazil",2,3,1,2536
3130903,Inhapim,0,0,0,0,0,0,0,0,0,0,0,0,40,204,6.666666667,78,50,64,43,13,74,9,4,0,0,0,0,0,257,0.1945525292,0.2490272374,0.1673151751,0.05058365759,0.2879377432,0.03501945525,0.01556420233,0,0,0,0,0,"Inhapim, MG, Brazil",7,2,1,0
3131307,Ipatinga,0,0,0,0,0,0,0,0,0,0,0,0,766,900,127.6666667,427,460,449,220,77,421,48,24,3,4,5,2,0,1713,0.2685347344,0.2621132516,0.1284296556,0.04495037945,0.2457676591,0.02802101576,0.01401050788,0.001751313485,0.002335084647,0.002918855809,0.001167542323,0,"Ipatinga, MG, Brazil",3,2,1,0
3131703,Itabira,0,0,0,0,0,0,0,0,0,0,0,0,1,96,0.1666666667,0,7,14,17,21,30,2,0,0,0,0,0,0,91,0.07692307692,0.1538461538,0.1868131868,0.2307692308,0.3296703297,0.02197802198,0,0,0,0,0,0,"Itabira, MG, Brazil",4,1,1,0
3131901,Itabirito,0,0,0,0,0,0,0,0,0,0,0,0,0,84,0,0,16,33,19,16,5,1,11,0,0,0,0,0,101,0.1584158416,0.3267326733,0.1881188119,0.1584158416,0.0495049505,0.009900990099,0.1089108911,0,0,0,0,0,"Itabirito, MG, Brazil",1,1,1,0
3132404,Itajubá,0,0,0,0,0,0,0,0,0,0,0,0,383,641,63.83333333,489,222,375,115,26,387,10,11,6,0,8,1,0,1161,0.1912144703,0.322997416,0.09905254091,0.02239448751,0.3333333333,0.008613264427,0.00947459087,0.005167958656,0,0.006890611542,0.0008613264427,0,"Itajubá, MG, Brazil",5,1,1,0
3132503,Itamarandiba,0,0,0,0,0,0,0,0,0,0,0,0,17,58,2.833333333,35,13,21,16,11,18,2,0,0,0,0,0,0,81,0.1604938272,0.2592592593,0.1975308642,0.1358024691,0.2222222222,0.02469135802,0,0,0,0,0,0,"Itamarandiba, MG, Brazil",3,1,1,0
3132701,Itambacuri,0,0,0,0,0,0,0,0,0,0,0,0,37,78,6.166666667,78,10,8,18,5,140,1,0,0,0,0,0,0,182,0.05494505495,0.04395604396,0.0989010989,0.02747252747,0.7692307692,0.005494505495,0,0,0,0,0,0,"Itambacuri, MG, Brazil",7,1,1,0
3133402,Itapagipe,0,0,0,0,0,0,0,0,0,0,0,0,292,65,48.66666667,65,43,79,30,11,0,2,0,0,0,0,0,0,165,0.2606060606,0.4787878788,0.1818181818,0.06666666667,0,0.01212121212,0,0,0,0,0,0,"Itapagipe, MG, Brazil",2,1,1,0
3133808,Itaúna,0,0,0,0,0,0,0,0,0,0,0,0,88,305,14.66666667,34,89,71,63,9,70,7,1,0,0,0,0,0,310,0.2870967742,0.2290322581,0.2032258065,0.02903225806,0.2258064516,0.02258064516,0.003225806452,0,0,0,0,0,"Itaúna, MG, Brazil",2,3,1,0
3134202,Ituiutaba,0,0,0,0,0,0,0,0,0,0,0,0,58,298,9.666666667,208,92,102,49,17,35,4,1,0,1,3,1,0,305,0.3016393443,0.3344262295,0.1606557377,0.05573770492,0.1147540984,0.0131147541,0.003278688525,0,0.003278688525,0.009836065574,0.003278688525,0,"Ituiutaba, MG, Brazil",2,2,1,0
3134608,Jaboticatubas,0,0,0,0,0,0,11,8,2,5,1,0,56,56,9.333333333,25,1,0,16,11,0,0,11,0,1,0,0,0,40,0.025,0,0.4,0.275,0,0,0.275,0,0.025,0,0,0,"Jaboticatubas, MG, Brazil",2,1,1,27
3134707,Jacinto,6,0,0,1,3,9,15,5,8,3,0,0,0,40,0,34,9,22,28,23,1,1,2,0,1,0,0,0,87,0.1034482759,0.2528735632,0.3218390805,0.2643678161,0.01149425287,0.01149425287,0.02298850575,0,0.01149425287,0,0,0,"Jacinto, MG, Brazil",5,1,1,50
3135100,Janaúba,0,0,0,0,0,0,0,0,0,0,0,0,409,388,68.16666667,159,54,67,52,16,175,8,1,2,0,0,1,0,376,0.1436170213,0.1781914894,0.1382978723,0.04255319149,0.4654255319,0.02127659574,0.002659574468,0.005319148936,0,0,0.002659574468,0,"Janaúba, MG, Brazil",3,1,1,0
3135209,Januária,0,0,0,0,0,18,14,7,5,5,1,0,194,212,32.33333333,82,62,55,39,26,130,14,2,0,0,0,0,0,328,0.1890243902,0.1676829268,0.118902439,0.07926829268,0.3963414634,0.04268292683,0.006097560976,0,0,0,0,0,"Januária, MG, Brazil",5,3,1,50
3135803,Jequitinhonha,24,0,0,0,2,21,41,19,15,5,36,33,93,83,15.5,80,34,45,48,11,0,4,3,0,0,0,0,0,145,0.2344827586,0.3103448276,0.3310344828,0.07586206897,0,0.0275862069,0.02068965517,0,0,0,0,0,"Jequitinhonha, MG, Brazil",4,1,1,196
3136207,João Monlevade,0,0,0,0,0,0,0,0,0,0,0,0,750,102,125,101,41,55,35,13,0,12,0,0,1,0,0,0,157,0.2611464968,0.3503184713,0.2229299363,0.08280254777,0,0.07643312102,0,0,0.006369426752,0,0,0,"João Monlevade, MG, Brazil",1,1,1,0
3136306,João Pinheiro,0,0,0,0,0,0,0,0,0,0,0,0,151,215,25.16666667,194,58,44,40,8,69,12,3,0,0,0,0,0,234,0.2478632479,0.188034188,0.1709401709,0.03418803419,0.2948717949,0.05128205128,0.01282051282,0,0,0,0,0,"João Pinheiro, MG, Brazil",2,1,1,0
3136702,Juiz de Fora,0,0,0,0,0,0,0,0,0,0,0,0,1664,1898,277.3333333,510,830,1246,660,168,369,122,8,6,5,6,0,2,3422,0.2425482174,0.3641145529,0.1928696669,0.04909409702,0.1078316774,0.03565166569,0.002337814144,0.001753360608,0.00146113384,0.001753360608,0,0.0005844535359,"Juiz de Fora, MG, Brazil",3,3,1,0
3137205,Lagoa da Prata,0,0,0,0,0,0,0,0,0,0,0,0,134,313,22.33333333,147,115,57,68,11,93,6,8,1,0,1,1,0,361,0.3185595568,0.1578947368,0.188365651,0.03047091413,0.2576177285,0.01662049861,0.02216066482,0.002770083102,0,0.002770083102,0.002770083102,0,"Lagoa da Prata, MG, Brazil",2,2,1,0
3137601,Lagoa Santa,0,0,0,0,0,0,0,0,0,0,0,0,9,46,1.5,46,22,32,20,0,15,6,5,0,0,0,1,0,101,0.2178217822,0.3168316832,0.198019802,0,0.1485148515,0.05940594059,0.0495049505,0,0,0,0.009900990099,0,"Lagoa Santa, MG, Brazil",1,1,1,0
3138203,Lavras,0,0,0,0,0,0,0,0,0,0,0,0,15,122,2.5,83,77,50,35,12,48,5,1,2,0,0,0,0,230,0.3347826087,0.2173913043,0.152173913,0.05217391304,0.2086956522,0.02173913043,0.004347826087,0.008695652174,0,0,0,0,"Lavras, MG, Brazil",4,1,1,0
3138401,Leopoldina,0,0,0,0,0,0,0,0,0,0,0,0,6,52,1,52,45,53,14,6,17,3,4,1,0,0,0,0,143,0.3146853147,0.3706293706,0.0979020979,0.04195804196,0.1188811189,0.02097902098,0.02797202797,0.006993006993,0,0,0,0,"Leopoldina, MG, Brazil",3,1,1,0
3138807,Luz,0,25,12,0,0,52,0,0,12,0,0,0,56,176,9.333333333,156,71,49,31,7,32,1,1,0,0,0,0,0,192,0.3697916667,0.2552083333,0.1614583333,0.03645833333,0.1666666667,0.005208333333,0.005208333333,0,0,0,0,0,"Luz, MG, Brazil",2,1,1,101
3139003,Machado,0,134,0,0,0,0,0,0,0,0,0,0,4,134,0.6666666667,90,40,23,22,20,23,0,2,2,0,2,0,0,134,0.2985074627,0.171641791,0.1641791045,0.1492537313,0.171641791,0,0.01492537313,0.01492537313,0,0.01492537313,0,0,"Machado, MG, Brazil",2,1,1,134
3139201,Malacacheta,0,0,0,0,0,0,0,0,0,0,0,0,39,78,6.5,62,21,33,21,11,30,3,6,2,0,0,0,0,127,0.1653543307,0.2598425197,0.1653543307,0.08661417323,0.2362204724,0.02362204724,0.04724409449,0.0157480315,0,0,0,0,"Malacacheta, MG, Brazil",3,1,1,0
3139300,Manga,0,0,0,0,0,0,0,0,0,0,0,0,3,94,0.5,63,24,19,31,10,68,3,2,0,0,0,0,0,157,0.152866242,0.1210191083,0.1974522293,0.06369426752,0.4331210191,0.01910828025,0.0127388535,0,0,0,0,0,"Manga, MG, Brazil",5,1,1,0
3139409,Manhuaçu,0,0,0,0,0,0,0,0,0,0,0,0,456,286,76,109,88,126,87,17,58,9,8,0,1,0,0,0,394,0.2233502538,0.3197969543,0.2208121827,0.04314720812,0.1472081218,0.02284263959,0.02030456853,0,0.002538071066,0,0,0,"Manhuaçu, MG, Brazil",6,2,1,0
3139508,Manhumirim,0,0,0,0,0,0,0,0,0,0,0,0,280,271,46.66666667,124,37,64,41,5,17,3,0,0,2,0,0,0,169,0.2189349112,0.3786982249,0.2426035503,0.02958579882,0.100591716,0.01775147929,0,0,0.01183431953,0,0,0,"Manhumirim, MG, Brazil",5,2,1,0
3139607,Mantena,0,0,0,0,0,0,0,0,0,0,0,0,96,134,16,108,29,68,50,2,25,2,3,0,0,0,0,0,179,0.1620111732,0.3798882682,0.2793296089,0.01117318436,0.1396648045,0.01117318436,0.01675977654,0,0,0,0,0,"Mantena, MG, Brazil",7,1,1,0
3140001,Mariana,132,0,0,0,0,0,0,0,0,0,0,0,94,129,15.66666667,68,39,44,19,1,34,8,1,0,0,0,0,0,146,0.2671232877,0.301369863,0.1301369863,0.006849315068,0.2328767123,0.05479452055,0.006849315068,0,0,0,0,0,"Mariana, MG, Brazil",2,1,1,132
3140704,Mateus Leme,0,0,0,0,0,0,0,0,0,0,0,0,413,109,68.83333333,0,31,81,19,0,2,4,2,0,2,0,1,0,142,0.2183098592,0.5704225352,0.1338028169,0,0.01408450704,0.02816901408,0.01408450704,0,0.01408450704,0,0.007042253521,0,"Mateus Leme, MG, Brazil",2,1,1,0
3140803,Matias Barbosa,0,0,0,0,0,0,0,0,0,0,0,0,48,73,8,73,17,18,22,1,13,2,1,0,0,0,0,0,74,0.2297297297,0.2432432432,0.2972972973,0.01351351351,0.1756756757,0.02702702703,0.01351351351,0,0,0,0,0,"Matias Barbosa, MG, Brazil",4,1,1,0
3141108,Matozinhos,0,0,0,0,0,1,7,4,1,3,0,0,12,91,2,36,1,1,6,2,8,0,1,0,0,10,0,0,29,0.03448275862,0.03448275862,0.2068965517,0.06896551724,0.275862069,0,0.03448275862,0,0,0.3448275862,0,0,"Matozinhos, MG, Brazil",3,1,1,16
3141405,Medina,0,0,0,0,0,0,0,0,0,0,0,0,7,91,1.166666667,34,19,27,26,4,53,3,0,3,0,0,0,0,135,0.1407407407,0.2,0.1925925926,0.02962962963,0.3925925926,0.02222222222,0,0.02222222222,0,0,0,0,"Medina, MG, Brazil",3,1,1,0
3142908,Monte Azul,0,0,0,0,0,0,0,0,0,0,0,0,77,56,12.83333333,47,35,37,37,30,0,4,3,2,0,0,0,0,148,0.2364864865,0.25,0.25,0.2027027027,0,0.02702702703,0.02027027027,0.01351351351,0,0,0,0,"Monte Azul, MG, Brazil",3,1,1,0
3143104,Monte Carmelo,0,0,1,3,10,31,36,11,7,2,0,0,15,104,2.5,104,45,70,25,4,0,0,2,0,0,0,0,0,146,0.3082191781,0.4794520548,0.1712328767,0.02739726027,0,0,0.01369863014,0,0,0,0,0,"Monte Carmelo, MG, Brazil",4,1,1,101
3143203,Monte Santo de Minas,0,0,0,0,0,0,0,0,0,0,0,0,127,135,21.16666667,45,57,47,29,18,67,1,3,2,0,0,0,0,224,0.2544642857,0.2098214286,0.1294642857,0.08035714286,0.2991071429,0.004464285714,0.01339285714,0.008928571429,0,0,0,0,"Monte Santo de Minas, MG, Brazil",2,1,1,0
3143302,Montes Claros,0,0,0,0,0,0,0,0,0,0,0,0,636,1065,106,21,407,311,365,86,256,28,12,4,0,1,0,0,1470,0.2768707483,0.2115646259,0.2482993197,0.05850340136,0.1741496599,0.01904761905,0.008163265306,0.002721088435,0,0.0006802721088,0,0,"Montes Claros, MG, Brazil",7,2,1,0
3143906,Muriaé,0,0,0,0,0,0,1,1,0,0,0,0,801,485,133.5,86,610,498,360,39,33,135,69,22,14,10,7,0,1797,0.3394546466,0.2771285476,0.2003338898,0.02170283806,0.0183639399,0.07512520868,0.03839732888,0.0122426266,0.007790762382,0.005564830273,0.003895381191,0,"Muriaé, MG, Brazil",3,2,1,2
3144300,Nanuque,0,0,0,0,0,0,0,0,0,0,0,0,44,116,7.333333333,109,19,35,25,12,40,2,1,0,1,2,0,0,137,0.1386861314,0.2554744526,0.1824817518,0.08759124088,0.2919708029,0.01459854015,0.007299270073,0,0.007299270073,0.01459854015,0,0,"Nanuque, MG, Brazil",2,1,1,0
3144607,Nepomuceno,0,0,0,0,0,0,0,0,0,0,0,0,5,60,0.8333333333,0,14,15,11,6,10,2,0,0,0,0,0,0,58,0.2413793103,0.2586206897,0.1896551724,0.1034482759,0.1724137931,0.03448275862,0,0,0,0,0,0,"Nepomuceno, MG, Brazil",1,1,1,0
3144706,Nova Era,0,0,0,0,0,0,0,0,0,0,0,0,98,92,16.33333333,90,27,39,20,7,24,7,2,0,0,0,0,0,126,0.2142857143,0.3095238095,0.1587301587,0.05555555556,0.1904761905,0.05555555556,0.01587301587,0,0,0,0,0,"Nova Era, MG, Brazil",2,1,1,0
3144805,Nova Lima,0,0,0,1,0,2,2,0,0,0,0,0,43,241,7.166666667,95,19,43,31,0,0,2,1,0,0,0,0,0,96,0.1979166667,0.4479166667,0.3229166667,0,0,0.02083333333,0.01041666667,0,0,0,0,0,"Nova Lima, MG, Brazil",3,2,1,5
3145208,Nova Serrana,0,0,0,0,0,0,0,0,0,0,0,0,325,138,54.16666667,110,91,52,54,10,52,8,1,1,4,0,0,0,273,0.3333333333,0.1904761905,0.1978021978,0.03663003663,0.1904761905,0.0293040293,0.003663003663,0.003663003663,0.01465201465,0,0,0,"Nova Serrana, MG, Brazil",3,1,1,0
3145307,Novo Cruzeiro,42,0,0,0,2,4,23,11,8,3,0,0,0,79,0,39,7,10,44,10,21,1,0,0,0,0,0,0,93,0.0752688172,0.1075268817,0.4731182796,0.1075268817,0.2258064516,0.01075268817,0,0,0,0,0,0,"Novo Cruzeiro, MG, Brazil",4,1,1,93
3145604,Oliveira,0,0,0,1,10,54,62,19,11,3,0,0,136,114,22.66666667,46,137,96,43,5,1,11,12,1,0,0,0,0,306,0.4477124183,0.3137254902,0.1405228758,0.01633986928,0.003267973856,0.03594771242,0.03921568627,0.003267973856,0,0,0,0,"Oliveira, MG, Brazil",2,1,1,160
3146107,Ouro Preto,0,0,0,0,0,0,0,0,0,0,0,0,54,114,9,48,17,34,16,0,19,1,1,0,0,0,0,0,88,0.1931818182,0.3863636364,0.1818181818,0,0.2159090909,0.01136363636,0.01136363636,0,0,0,0,0,"Ouro Preto, MG, Brazil",1,1,1,0
3147006,Paracatu,0,0,0,0,0,0,0,0,0,0,0,0,90,445,15,239,146,89,80,20,108,7,7,3,0,2,0,0,462,0.316017316,0.1926406926,0.1731601732,0.04329004329,0.2337662338,0.01515151515,0.01515151515,0.006493506494,0,0.004329004329,0,0,"Paracatu, MG, Brazil",1,2,1,0
3147105,Pará de Minas,0,0,0,0,0,0,0,0,0,0,0,0,435,399,72.5,2,209,249,143,43,89,16,12,4,2,1,0,0,768,0.2721354167,0.32421875,0.1861979167,0.05598958333,0.1158854167,0.02083333333,0.015625,0.005208333333,0.002604166667,0.001302083333,0,0,"Pará de Minas, MG, Brazil",7,1,1,0
3147907,Passos,0,0,0,0,0,0,0,0,0,0,0,0,67,391,11.16666667,81,169,31,48,20,111,1,2,0,0,0,0,0,382,0.442408377,0.08115183246,0.1256544503,0.05235602094,0.2905759162,0.002617801047,0.005235602094,0,0,0,0,0,"Passos, MG, Brazil",2,2,1,0
3148004,Patos de Minas,0,0,0,0,0,0,0,0,0,0,0,0,88,300,14.66666667,200,167,90,88,11,75,2,3,1,0,1,0,0,438,0.3812785388,0.2054794521,0.200913242,0.02511415525,0.1712328767,0.004566210046,0.006849315068,0.002283105023,0,0.002283105023,0,0,"Patos de Minas, MG, Brazil",4,2,1,0
3148103,Patrocínio,0,0,0,0,0,0,0,0,0,0,0,0,925,867,154.1666667,88,332,268,218,56,457,26,13,5,2,0,0,0,1377,0.2411038489,0.1946259985,0.1583151779,0.0406681191,0.3318809005,0.01888162672,0.009440813362,0.003631082062,0.001452432825,0,0,0,"Patrocínio, MG, Brazil",4,2,1,0
3148608,Peçanha,0,0,0,0,0,0,0,0,0,0,0,0,5,129,0.8333333333,52,5,29,24,11,18,1,0,0,0,0,0,0,88,0.05681818182,0.3295454545,0.2727272727,0.125,0.2045454545,0.01136363636,0,0,0,0,0,0,"Peçanha, MG, Brazil",8,1,1,0
3148707,Pedra Azul,0,0,0,0,0,0,0,0,0,0,0,0,11,92,1.833333333,50,0,7,6,1,32,0,2,0,1,0,0,0,49,0,0.1428571429,0.1224489796,0.02040816327,0.6530612245,0,0.04081632653,0,0.02040816327,0,0,0,"Pedra Azul, MG, Brazil",4,2,1,0
3149309,Pedro Leopoldo,0,0,0,0,0,0,0,0,0,0,0,0,9,65,1.5,65,28,29,14,0,11,2,0,0,1,1,0,0,86,0.3255813953,0.3372093023,0.1627906977,0,0.1279069767,0.02325581395,0,0,0.01162790698,0.01162790698,0,0,"Pedro Leopoldo, MG, Brazil",2,1,1,0
3149804,Perdizes,0,0,0,1,8,24,27,7,6,2,1,0,22,66,3.666666667,53,13,55,32,2,0,1,1,0,0,0,0,0,104,0.125,0.5288461538,0.3076923077,0.01923076923,0,0.009615384615,0.009615384615,0,0,0,0,0,"Perdizes, MG, Brazil",2,1,1,76
3149903,Perdões,0,0,0,0,0,0,0,0,0,0,0,0,58,90,9.666666667,0,5,19,7,9,36,2,0,0,0,0,0,0,78,0.0641025641,0.2435897436,0.08974358974,0.1153846154,0.4615384615,0.02564102564,0,0,0,0,0,0,"Perdões, MG, Brazil",2,1,1,0
3151206,Pirapora,0,0,0,0,0,20,33,18,14,0,0,0,37,260,6.166666667,86,72,98,94,16,47,3,2,1,0,1,0,0,334,0.2155688623,0.2934131737,0.2814371257,0.04790419162,0.1407185629,0.008982035928,0.005988023952,0.002994011976,0,0.002994011976,0,0,"Pirapora, MG, Brazil",3,2,1,85
3151503,Piumhi,0,0,0,0,3,18,40,9,15,10,0,1,0,208,0,106,56,46,32,21,84,0,0,0,0,6,0,0,245,0.2285714286,0.187755102,0.1306122449,0.08571428571,0.3428571429,0,0,0,0,0.02448979592,0,0,"Piumhi, MG, Brazil",3,1,1,96
3151800,Poços de Caldas,0,0,0,0,0,0,0,0,0,0,0,0,575,140,95.83333333,104,47,35,27,3,29,1,1,0,0,0,0,0,143,0.3286713287,0.2447552448,0.1888111888,0.02097902098,0.2027972028,0.006993006993,0.006993006993,0,0,0,0,0,"Poços de Caldas, MG, Brazil",1,1,1,0
3152105,Ponte Nova,0,0,0,0,0,0,0,0,0,0,0,0,321,1086,53.5,382,206,339,157,45,333,19,9,0,4,1,1,0,1114,0.1849192101,0.3043087971,0.1409335727,0.04039497307,0.2989228007,0.0170556553,0.008078994614,0,0.003590664273,0.0008976660682,0.0008976660682,0,"Ponte Nova, MG, Brazil",8,1,1,0
3152204,Porteirinha,0,0,0,0,0,0,0,0,0,0,0,0,35,92,5.833333333,80,26,20,21,26,20,1,1,1,0,0,0,0,116,0.224137931,0.1724137931,0.1810344828,0.224137931,0.1724137931,0.008620689655,0.008620689655,0.008620689655,0,0,0,0,"Porteirinha, MG, Brazil",5,1,1,0
3152501,Pouso Alegre,0,0,0,0,0,0,0,0,0,0,0,0,163,660,27.16666667,353,244,390,102,45,229,19,13,0,0,1,2,0,1045,0.233492823,0.3732057416,0.0976076555,0.04306220096,0.219138756,0.01818181818,0.01244019139,0,0,0.000956937799,0.001913875598,0,"Pouso Alegre, MG, Brazil",4,3,1,0
3152808,Prata,41,0,1,1,2,15,21,12,8,2,1,0,51,99,8.5,50,44,40,27,10,0,8,0,1,0,0,0,0,130,0.3384615385,0.3076923077,0.2076923077,0.07692307692,0,0.06153846154,0,0.007692307692,0,0,0,0,"Prata, MG, Brazil",1,1,1,104
3153400,Presidente Olegário,0,0,0,0,1,24,19,9,4,2,0,0,85,64,14.16666667,61,53,24,21,5,0,3,3,0,3,1,1,0,114,0.4649122807,0.2105263158,0.1842105263,0.04385964912,0,0.02631578947,0.02631578947,0,0.02631578947,0.008771929825,0.008771929825,0,"Presidente Olegário, MG, Brazil",3,1,1,59
3154200,Resende Costa,0,0,0,0,0,0,0,0,0,0,0,0,9,50,1.5,39,18,12,21,7,18,0,2,0,0,0,0,0,78,0.2307692308,0.1538461538,0.2692307692,0.08974358974,0.2307692308,0,0.02564102564,0,0,0,0,0,"Resende Costa, MG, Brazil",2,1,1,0
3154606,Ribeirão das Neves,0,2,5,8,21,190,391,180,161,106,41,7,7682,5438,1280.333333,1303,1964,1902,933,487,1784,165,48,10,29,12,24,0,7358,0.2669203588,0.258494156,0.1268007611,0.06618646371,0.2424571895,0.02242457189,0.006523511824,0.001359064963,0.003941288394,0.001630877956,0.003261755912,0,"Ribeirão das Neves, MG, Brazil",1,8,1,1112
3155702,Rio Piracicaba,0,0,0,0,0,0,0,0,0,0,0,0,18,56,3,0,8,24,9,3,9,1,1,0,0,0,0,0,55,0.1454545455,0.4363636364,0.1636363636,0.05454545455,0.1636363636,0.01818181818,0.01818181818,0,0,0,0,0,"Rio Piracicaba, MG, Brazil",1,1,1,0
3155801,Rio Pomba,0,0,0,0,0,0,0,0,0,0,0,0,60,196,10,130,59,43,16,9,69,3,2,0,0,0,0,0,201,0.2935323383,0.2139303483,0.07960199005,0.0447761194,0.3432835821,0.01492537313,0.009950248756,0,0,0,0,0,"Rio Pomba, MG, Brazil",3,1,1,0
3156908,Sacramento,0,0,0,0,0,0,0,0,0,0,0,0,131,122,21.83333333,45,53,64,38,27,1,20,1,0,0,0,0,0,204,0.2598039216,0.3137254902,0.1862745098,0.1323529412,0.004901960784,0.09803921569,0.004901960784,0,0,0,0,0,"Sacramento, MG, Brazil",1,1,1,0
3157005,Salinas,0,0,1,2,7,51,28,2,0,0,0,0,106,127,17.66666667,34,17,16,20,19,76,2,2,0,0,0,0,0,152,0.1118421053,0.1052631579,0.1315789474,0.125,0.5,0.01315789474,0.01315789474,0,0,0,0,0,"Salinas, MG, Brazil",6,2,1,91
3157203,Santa Bárbara,0,0,0,0,0,0,0,0,0,0,0,0,5,64,0.8333333333,0,1,10,6,4,13,0,0,0,0,0,0,0,34,0.02941176471,0.2941176471,0.1764705882,0.1176470588,0.3823529412,0,0,0,0,0,0,0,"Santa Bárbara, MG, Brazil",3,1,1,0
3157807,Santa Luzia,0,0,0,0,0,0,0,0,0,0,0,0,35,248,5.833333333,48,35,71,70,26,97,16,3,0,1,0,0,0,319,0.1097178683,0.2225705329,0.2194357367,0.08150470219,0.3040752351,0.05015673981,0.009404388715,0,0.003134796238,0,0,0,"Santa Luzia, MG, Brazil",1,2,1,0
3158201,Santa Maria do Suaçuí,0,0,0,0,0,0,0,0,0,0,0,0,9,50,1.5,0,2,7,11,11,17,1,0,0,0,0,0,0,49,0.04081632653,0.1428571429,0.2244897959,0.2244897959,0.3469387755,0.02040816327,0,0,0,0,0,0,"Santa Maria do Suaçuí, MG, Brazil",4,1,1,0
3159605,Santa Rita do Sapucaí,0,0,0,0,0,0,0,0,0,0,0,0,88,93,14.66666667,46,36,70,18,4,24,0,1,1,0,1,0,0,155,0.2322580645,0.4516129032,0.1161290323,0.02580645161,0.1548387097,0,0.006451612903,0.006451612903,0,0.006451612903,0,0,"Santa Rita do Sapucaí, MG, Brazil",2,1,1,0
3159803,Santa Vitória,0,0,0,0,0,0,0,0,0,0,0,0,15,105,2.5,77,22,28,12,5,30,1,1,0,0,0,0,0,99,0.2222222222,0.2828282828,0.1212121212,0.05050505051,0.303030303,0.0101010101,0.0101010101,0,0,0,0,0,"Santa Vitória, MG, Brazil",1,1,1,0
3160702,Santos Dumont,5,0,1,2,0,24,50,28,23,4,0,0,71,98,11.83333333,98,41,37,16,2,69,2,1,0,0,0,0,0,168,0.244047619,0.2202380952,0.09523809524,0.0119047619,0.4107142857,0.0119047619,0.005952380952,0,0,0,0,0,"Santos Dumont, MG, Brazil",5,1,1,137
3161106,São Francisco,31,0,3,2,4,21,32,11,7,1,0,0,195,56,32.5,56,71,29,34,4,0,5,3,0,0,0,0,0,146,0.4863013699,0.198630137,0.2328767123,0.02739726027,0,0.03424657534,0.02054794521,0,0,0,0,0,"São Francisco, MG, Brazil",3,1,1,112
3162401,São João da Ponte,0,0,0,0,0,0,0,0,0,0,0,0,109,68,18.16666667,66,52,25,73,26,37,0,1,0,0,0,0,0,214,0.2429906542,0.1168224299,0.3411214953,0.1214953271,0.1728971963,0,0.004672897196,0,0,0,0,0,"São João da Ponte, MG, Brazil",4,1,1,0
3162500,São João del Rei,0,0,0,0,0,0,0,0,0,0,0,0,194,876,32.33333333,120,234,350,154,35,306,26,25,1,3,2,1,0,1137,0.2058047493,0.3078276165,0.1354441513,0.03078276165,0.2691292876,0.02286719437,0.0219876869,0.0008795074758,0.002638522427,0.001759014952,0.0008795074758,0,"São João del Rei, MG, Brazil",8,3,1,0
3162807,São João Evangelista,0,2,15,20,31,45,19,3,0,0,0,0,45,86,7.5,25,40,30,22,18,0,5,0,0,0,0,0,0,115,0.347826087,0.2608695652,0.1913043478,0.1565217391,0,0.04347826087,0,0,0,0,0,0,"São João Evangelista, MG, Brazil",2,1,1,135
3163706,São Lourenço,0,0,0,0,0,0,0,0,0,0,0,0,33,281,5.5,74,136,252,87,34,49,5,6,2,1,0,0,0,572,0.2377622378,0.4405594406,0.1520979021,0.05944055944,0.08566433566,0.008741258741,0.01048951049,0.003496503497,0.001748251748,0,0,0,"São Lourenço, MG, Brazil",4,1,1,0
3164704,São Sebastião do Paraíso,0,0,0,0,0,0,0,0,0,0,0,0,29,154,4.833333333,53,98,92,21,16,69,3,2,0,2,1,0,0,304,0.3223684211,0.3026315789,0.06907894737,0.05263157895,0.2269736842,0.009868421053,0.006578947368,0,0.006578947368,0.003289473684,0,0,"São Sebastião do Paraíso, MG, Brazil",2,1,1,0
3167103,Serro,0,0,0,0,0,0,0,0,0,0,0,0,7,110,1.166666667,90,24,61,64,19,30,1,4,0,0,0,0,0,203,0.118226601,0.3004926108,0.315270936,0.09359605911,0.1477832512,0.004926108374,0.0197044335,0,0,0,0,0,"Serro, MG, Brazil",4,1,1,0
3167202,Sete Lagoas,0,0,0,0,0,0,0,0,0,0,0,0,116,407,19.33333333,304,239,252,101,5,71,25,4,0,5,0,1,5,708,0.3375706215,0.3559322034,0.1426553672,0.007062146893,0.1002824859,0.03531073446,0.005649717514,0,0.007062146893,0,0.001412429379,0.007062146893,"Sete Lagoas, MG, Brazil",8,2,1,0
3168002,Taiobeiras,0,0,0,5,2,15,11,4,4,0,0,0,309,40,51.5,30,27,31,23,12,19,1,0,1,0,0,0,0,114,0.2368421053,0.2719298246,0.201754386,0.1052631579,0.1666666667,0.008771929825,0,0.008771929825,0,0,0,0,"Taiobeiras, MG, Brazil",4,1,1,41
3168408,Tarumirim,0,0,0,0,0,0,0,0,0,0,0,0,0,105,0,62,13,21,18,6,41,2,2,0,0,0,0,0,103,0.1262135922,0.2038834951,0.1747572816,0.05825242718,0.3980582524,0.01941747573,0.01941747573,0,0,0,0,0,"Tarumirim, MG, Brazil",4,1,1,0
3168606,Teófilo Otoni,0,0,0,0,0,0,0,0,0,0,0,0,475,669,79.16666667,125,138,205,155,34,319,27,12,1,2,10,1,0,904,0.1526548673,0.2267699115,0.171460177,0.03761061947,0.3528761062,0.02986725664,0.01327433628,0.00110619469,0.002212389381,0.0110619469,0.00110619469,0,"Teófilo Otoni, MG, Brazil",7,3,1,0
3168705,Timóteo,0,0,0,0,0,0,0,0,0,0,0,0,14,166,2.333333333,68,38,75,23,2,40,1,1,0,0,0,0,0,180,0.2111111111,0.4166666667,0.1277777778,0.01111111111,0.2222222222,0.005555555556,0.005555555556,0,0,0,0,0,"Timóteo, MG, Brazil",3,1,1,0
3169307,Três Corações,0,0,0,0,0,0,0,0,0,0,0,0,406,546,67.66666667,126,298,439,188,50,249,27,12,3,3,2,0,0,1271,0.2344610543,0.3453973249,0.1479150275,0.03933910307,0.1959087333,0.02124311566,0.009441384736,0.002360346184,0.002360346184,0.001573564123,0,0,"Três Corações, MG, Brazil",3,1,1,0
3169356,Três Marias,0,1,0,2,2,7,18,6,1,2,0,0,11,26,1.833333333,26,18,9,8,6,17,0,1,1,0,0,0,0,60,0.3,0.15,0.1333333333,0.1,0.2833333333,0,0.01666666667,0.01666666667,0,0,0,0,"Três Marias, MG, Brazil",1,1,1,39
3169406,Três Pontas,0,0,0,0,0,0,0,0,0,0,0,0,10,100,1.666666667,59,19,0,31,0,0,0,0,0,0,0,0,0,50,0.38,0,0.62,0,0,0,0,0,0,0,0,0,"Três Pontas, MG, Brazil",2,1,1,0
3169604,Tupaciguara,0,2,2,6,2,14,32,12,10,3,1,0,107,104,17.83333333,104,33,39,17,16,0,1,0,0,0,0,0,0,106,0.3113207547,0.3679245283,0.1603773585,0.1509433962,0,0.009433962264,0,0,0,0,0,0,"Tupaciguara, MG, Brazil",2,1,1,84
3169703,Turmalina,0,0,0,2,1,35,22,4,1,1,0,0,11,86,1.833333333,70,19,12,23,11,29,1,1,0,1,0,0,2,99,0.1919191919,0.1212121212,0.2323232323,0.1111111111,0.2929292929,0.0101010101,0.0101010101,0,0.0101010101,0,0,0.0202020202,"Turmalina, MG, Brazil",4,1,1,66
3169901,Ubá,0,0,0,0,0,0,0,0,0,0,0,0,0,126,0,122,64,110,28,3,47,7,4,0,0,0,0,0,263,0.2433460076,0.4182509506,0.1064638783,0.01140684411,0.1787072243,0.02661596958,0.01520912548,0,0,0,0,0,"Ubá, MG, Brazil",5,1,1,0
3170107,Uberaba,0,0,0,0,0,0,0,0,0,0,0,0,1444,698,240.6666667,0,376,258,119,25,636,25,11,20,3,0,0,0,1473,0.2552613714,0.1751527495,0.08078750849,0.01697216565,0.4317718941,0.01697216565,0.007467752885,0.01357773252,0.002036659878,0,0,0,"Uberaba, MG, Brazil",5,1,1,0
3170206,Uberlândia,0,0,0,0,0,0,0,0,0,0,0,0,2086,1353,347.6666667,628,712,1053,164,50,466,3,3,10,8,0,0,0,2469,0.2883758607,0.4264884569,0.0664236533,0.02025111381,0.1887403807,0.001215066829,0.001215066829,0.004050222762,0.00324017821,0,0,0,"Uberlândia, MG, Brazil",1,2,1,0
3170404,Unaí,0,0,0,0,0,8,7,2,4,0,0,0,628,626,104.6666667,106,239,254,239,98,357,18,14,2,0,2,0,0,1223,0.1954210957,0.207686018,0.1954210957,0.08013082584,0.2919051513,0.01471790679,0.01144726083,0.001635322976,0,0.001635322976,0,0,"Unaí, MG, Brazil",2,2,1,21
3170701,Varginha,0,0,0,0,0,0,0,0,0,0,0,0,630,184,105,68,80,74,50,9,99,8,6,0,5,0,0,0,331,0.2416918429,0.2235649547,0.1510574018,0.02719033233,0.2990936556,0.02416918429,0.01812688822,0,0.01510574018,0,0,0,"Varginha, MG, Brazil",3,2,1,0
3170800,Várzea da Palma,0,0,0,0,0,0,0,0,0,0,0,0,2,35,0.3333333333,24,21,20,12,8,31,6,2,1,0,0,0,0,101,0.2079207921,0.198019802,0.1188118812,0.07920792079,0.3069306931,0.05940594059,0.0198019802,0.009900990099,0,0,0,0,"Várzea da Palma, MG, Brazil",2,1,1,0
3171204,Vespasiano,0,0,0,0,0,7,5,3,1,0,0,0,72,283,12,280,51,83,61,1,38,3,1,0,1,3,2,0,244,0.2090163934,0.3401639344,0.25,0.004098360656,0.1557377049,0.01229508197,0.004098360656,0,0.004098360656,0.01229508197,0.008196721311,0,"Vespasiano, MG, Brazil",2,2,1,16
3171303,Viçosa,0,0,0,0,0,0,0,0,0,0,0,0,13,179,2.166666667,129,69,93,35,9,55,16,8,1,1,0,0,0,287,0.2404181185,0.3240418118,0.1219512195,0.03135888502,0.1916376307,0.05574912892,0.02787456446,0.003484320557,0.003484320557,0,0,0,"Viçosa, MG, Brazil",6,2,1,0
3172004,Visconde do Rio Branco,0,1,1,7,11,24,13,3,2,0,0,0,51,226,8.5,127,54,62,42,4,170,2,1,0,0,0,0,0,335,0.1611940299,0.1850746269,0.1253731343,0.01194029851,0.5074626866,0.005970149254,0.002985074627,0,0,0,0,0,"Visconde do Rio Branco, MG, Brazil",3,2,1,62
3200607,Aracruz,0,0,0,0,1,5,10,1,0,0,0,0,481,246,80.16666667,246,113,164,103,5,0,31,2,7,3,0,0,19,447,0.2527964206,0.3668903803,0.2304250559,0.01118568233,0,0.06935123043,0.004474272931,0.01565995526,0.006711409396,0,0,0.04250559284,"Aracruz, ES, Brazil",1,1,1,17
3200904,Barra de São Francisco,0,0,2,15,85,90,25,8,17,16,7,1,97,112,16.16666667,0,66,107,74,0,0,18,0,0,0,0,0,0,265,0.2490566038,0.4037735849,0.279245283,0,0,0.0679245283,0,0,0,0,0,0,"Barra de São Francisco, ES, Brazil",1,1,1,266
3201209,Cachoeiro de Itapemirim,47,0,0,1,2,23,47,31,38,15,0,1,1079,918,179.8333333,308,405,1026,322,6,0,43,18,3,3,0,1,0,1827,0.2216748768,0.5615763547,0.1762452107,0.00328407225,0,0.02353585112,0.009852216749,0.001642036125,0.001642036125,0,0.0005473453749,0,"Cachoeiro de Itapemirim, ES, Brazil",1,7,1,205
3201308,Cariacica,16,0,0,1,0,148,240,101,81,30,7,0,569,944,94.83333333,169,239,496,202,26,16,71,29,16,8,3,3,1,1110,0.2153153153,0.4468468468,0.181981982,0.02342342342,0.01441441441,0.06396396396,0.02612612613,0.01441441441,0.007207207207,0.002702702703,0.002702702703,0.0009009009009,"Cariacica, ES, Brazil",1,5,1,624
3201506,Colatina,0,0,0,1,1,4,6,2,2,0,0,0,1220,1266,203.3333333,646,273,540,214,34,0,55,70,0,2,3,0,0,1191,0.2292191436,0.4534005038,0.1796809404,0.02854743913,0,0.04617968094,0.05877413938,0,0.001679261125,0.002518891688,0,0,"Colatina, ES, Brazil",2,6,1,16
3202405,Guarapari,0,0,0,0,0,0,0,0,0,0,0,0,888,580,148,580,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Guarapari, ES, Brazil",1,2,0,0
3203205,Linhares,0,0,0,0,0,0,0,0,0,0,0,0,735,982,122.5,336,430,496,186,0,0,97,13,4,3,0,1,0,1230,0.3495934959,0.4032520325,0.1512195122,0,0,0.07886178862,0.01056910569,0.00325203252,0.00243902439,0,0.0008130081301,0,"Linhares, ES, Brazil",2,3,1,0
3203320,Marataízes,0,0,0,0,0,0,0,0,0,0,0,0,389,236,64.83333333,236,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Marataízes, ES, Brazil",1,2,0,0
3204658,São Domingos do Norte,0,0,0,0,0,0,0,0,0,0,0,0,626,236,104.3333333,236,97,144,222,18,0,42,69,0,3,0,0,3,598,0.1622073579,0.2408026756,0.3712374582,0.03010033445,0,0.07023411371,0.1153846154,0,0.005016722408,0,0,0.005016722408,"São Domingos do Norte, ES, Brazil",1,2,1,0
3204906,São Mateus,0,0,0,0,28,144,311,338,88,31,2,0,1513,1106,252.1666667,390,106,344,195,0,0,67,32,1,0,2,1,0,748,0.1417112299,0.4598930481,0.2606951872,0,0,0.08957219251,0.04278074866,0.001336898396,0,0.002673796791,0.001336898396,0,"São Mateus, ES, Brazil",1,5,1,942
3205002,Serra,0,0,0,0,0,0,0,0,0,0,0,0,110,548,18.33333333,548,356,682,391,356,0,75,0,0,0,0,0,0,1860,0.1913978495,0.3666666667,0.2102150538,0.1913978495,0,0.04032258065,0,0,0,0,0,0,"Serra, ES, Brazil",1,2,1,0
3205101,Viana,123,0,2,1,14,269,241,63,43,44,47,17,6321,6336,1053.5,1166,1079,1363,965,53,16,201,50,32,42,8,2,1,3812,0.2830535152,0.3575550892,0.2531479538,0.01390346275,0.004197271773,0.05272822665,0.01311647429,0.008394543547,0.01101783841,0.002098635887,0.0005246589717,0.0002623294858,"Viana, ES, Brazil",1,11,1,864
3205200,Vila Velha,10,2,1,2,3,25,25,5,4,0,0,0,3001,5378,500.1666667,534,2680,2299,1444,4,86,416,106,22,46,12,8,0,7123,0.3762459638,0.3227572652,0.2027235715,0.00056156114,0.01207356451,0.05840235856,0.01488137021,0.00308858627,0.00645795311,0.00168468342,0.00112312228,0,"Vila Velha, ES, Brazil",1,12,1,77
3301009,Campos dos Goytacazes,0,0,3,20,16,19,21,24,20,1,1,0,2557,1592,426.1666667,642,27,103,59,18,0,6,5,0,0,0,0,0,218,0.123853211,0.4724770642,0.2706422018,0.08256880734,0,0.02752293578,0.02293577982,0,0,0,0,0,"Campos dos Goytacazes, RJ, Brazil",1,4,1,125
3302205,Itaperuna,0,0,0,0,0,0,0,0,0,0,0,0,1256,507,209.3333333,198,18,7,1,6,735,0,0,0,0,0,0,0,767,0.02346805737,0.009126466754,0.001303780965,0.007822685789,0.9582790091,0,0,0,0,0,0,0,"Itaperuna, RJ, Brazil",1,1,1,0
3302270,Japeri,0,0,0,0,0,0,0,0,0,0,0,0,5383,2518,897.1666667,1634,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Japeri, RJ, Brazil",1,3,0,0
3302502,Magé,0,0,0,1,0,9,8,3,7,4,1,0,1483,1786,247.1666667,1142,22,8,2,0,1105,4,0,1,0,0,0,0,1142,0.01926444834,0.00700525394,0.001751313485,0,0.9676007005,0.00350262697,0,0.0008756567426,0,0,0,0,"Magé, RJ, Brazil",1,3,1,33
3303302,Niterói,0,0,0,0,0,16,27,20,21,9,2,0,116,718,19.33333333,30,34,15,59,8,0,4,1,7,1,5,1,0,135,0.2518518519,0.1111111111,0.437037037,0.05925925926,0,0.02962962963,0.007407407407,0.05185185185,0.007407407407,0.03703703704,0.007407407407,0,"Niterói, RJ, Brazil",1,4,1,95
3304201,Resende,0,0,0,1,4,67,184,104,79,28,8,0,148,432,24.66666667,72,204,234,159,181,0,35,0,5,3,0,8,0,829,0.246079614,0.2822677925,0.1917973462,0.2183353438,0,0.04221954162,0,0.006031363088,0.003618817853,0,0.009650180941,0,"Resende, RJ, Brazil",1,1,1,475
3304557,Rio de Janeiro,1189,87,132,197,264,348,220,159,62,36,15,1,20779,31868,3463.166667,6142,357,223,257,90,2390,37,27,27,13,1,8,0,3430,0.1040816327,0.06501457726,0.0749271137,0.02623906706,0.6967930029,0.01078717201,0.007871720117,0.007871720117,0.003790087464,0.0002915451895,0.002332361516,0,"Rio de Janeiro, RJ, Brazil",1,38,1,2710
3304904,São Gonçalo,0,0,0,0,0,0,0,0,0,0,0,0,6977,1274,1162.833333,1274,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São Gonçalo, RJ, Brazil",1,2,0,0
3306305,Volta Redonda,0,0,0,0,0,0,0,0,0,0,0,0,43,339,7.166666667,339,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Volta Redonda, RJ, Brazil",1,1,0,0
3500303,Aguaí,0,1,18,51,129,223,242,15,17,4,0,0,1432,823,238.6666667,823,364,685,154,0,0,10,0,4,3,1,0,9,1230,0.2959349593,0.5569105691,0.125203252,0,0,0.008130081301,0,0.00325203252,0.00243902439,0.0008130081301,0,0.007317073171,"Aguaí, SP, Brazil",1,1,1,700
3501608,Americana,0,0,12,18,36,58,14,8,0,0,0,0,1304,640,217.3333333,640,228,398,189,0,0,12,0,10,0,0,0,0,837,0.2724014337,0.4755077658,0.2258064516,0,0,0.01433691756,0,0.0119474313,0,0,0,0,"Americana, SP, Brazil",1,1,1,146
3502101,Andradina,0,0,2,28,202,740,836,297,319,246,367,5,1134,2120,189,0,1383,992,654,0,0,98,9,13,15,0,0,0,3164,0.4371049305,0.3135271808,0.2067003793,0,0,0.03097345133,0.002844500632,0.004108723135,0.004740834387,0,0,0,"Andradina, SP, Brazil",4,2,1,3042
3502507,Aparecida,0,2,19,142,155,830,706,365,331,57,41,6,952,1907,158.6666667,0,1316,1050,402,1,0,46,0,9,2,0,0,9,2835,0.4641975309,0.3703703704,0.1417989418,0.0003527336861,0,0.01622574956,0,0.003174603175,0.0007054673721,0,0,0.003174603175,"Aparecida, SP, Brazil",2,2,1,2654
3502804,Araçatuba,12,0,0,0,0,70,67,8,2,0,0,0,65,214,10.83333333,38,14,56,29,57,0,0,3,0,0,0,0,0,159,0.08805031447,0.3522012579,0.1823899371,0.358490566,0,0,0.01886792453,0,0,0,0,0,"Araçatuba, SP, Brazil",2,1,1,159
3503208,Araraquara,0,51,3,94,29,463,569,410,131,32,3,0,1288,1916,214.6666667,496,928,1017,299,0,50,42,0,8,11,0,1,2,2358,0.3935538592,0.4312977099,0.1268023749,0,0.02120441052,0.01781170483,0,0.003392705683,0.004664970314,0,0.0004240882103,0.0008481764207,"Araraquara, SP, Brazil",6,3,1,1785
3504008,Assis,0,34,93,169,342,671,781,334,164,64,25,3,1253,1964,208.8333333,0,852,313,170,0,0,8,22,0,8,0,0,0,1373,0.6205389658,0.2279679534,0.1238164603,0,0,0.005826656956,0.01602330663,0,0.005826656956,0,0,0,"Assis, SP, Brazil",4,2,1,2680
3504107,Atibaia,0,1,7,16,27,99,21,0,0,0,0,0,169,204,28.16666667,0,34,109,28,0,0,0,0,0,0,0,0,0,171,0.1988304094,0.6374269006,0.1637426901,0,0,0,0,0,0,0,0,0,"Atibaia, SP, Brazil",2,1,1,171
3504503,Avaré,0,0,0,13,92,942,666,137,108,134,95,40,1628,2085,271.3333333,0,741,1266,202,2,0,27,6,18,9,1,1,1,2274,0.3258575198,0.5567282322,0.08883025506,0.0008795074758,0,0.01187335092,0.002638522427,0.007915567282,0.003957783641,0.0004397537379,0.0004397537379,0.0004397537379,"Avaré, SP, Brazil",2,3,1,2227
3506003,Bauru,0,11,25,59,209,1741,1746,889,611,208,45,0,4454,5384,742.3333333,844,2573,3002,544,0,0,34,4,13,4,0,2,3,6179,0.4164104224,0.4858391325,0.08804013594,0,0,0.005502508497,0.0006473539408,0.002103900307,0.0006473539408,0,0.0003236769704,0.0004855154556,"Bauru, SP, Brazil",3,4,1,5544
3506508,Birigui,0,0,0,0,16,86,107,48,34,14,7,0,211,214,35.16666667,0,117,92,90,0,7,2,1,0,0,1,1,1,312,0.375,0.2948717949,0.2884615385,0,0.02243589744,0.00641025641,0.003205128205,0,0,0.003205128205,0.003205128205,0.003205128205,"Birigui, SP, Brazil",5,1,1,312
3507001,Boituva,0,0,0,2,12,129,485,437,610,447,186,22,1115,1851,185.8333333,344,1389,637,314,362,0,144,108,0,26,0,4,1,2985,0.4653266332,0.213400335,0.1051926298,0.1212730318,0,0.04824120603,0.03618090452,0,0.008710217755,0,0.001340033501,0.0003350083752,"Boituva, SP, Brazil",2,1,1,2330
3507605,Bragança Paulista,0,1,0,0,3,109,124,5,1,0,0,0,153,259,25.5,0,50,61,62,92,0,1,3,0,0,1,0,1,271,0.184501845,0.2250922509,0.2287822878,0.3394833948,0,0.0036900369,0.0110701107,0,0,0.0036900369,0,0.0036900369,"Bragança Paulista, SP, Brazil",4,1,1,243
3509502,Campinas,0,104,194,355,551,1161,458,146,88,39,18,0,3347,3436,557.8333333,822,1832,1553,356,0,0,73,19,69,14,0,2,2,3920,0.4673469388,0.3961734694,0.09081632653,0,0,0.01862244898,0.004846938776,0.01760204082,0.003571428571,0,0.0005102040816,0.0005102040816,"Campinas, SP, Brazil",1,3,1,3114
3510500,Caraguatatuba,0,0,1,12,37,285,197,67,42,17,1,0,657,847,109.5,751,348,561,143,0,0,15,3,0,6,0,1,0,1077,0.3231197772,0.5208913649,0.1327762303,0,0,0.0139275766,0.00278551532,0,0.005571030641,0,0.0009285051068,0,"Caraguatatuba, SP, Brazil",1,1,1,659
3510807,Casa Branca,0,0,0,1,8,209,506,235,268,156,31,5,398,926,66.33333333,0,2100,1511,460,0,0,129,53,24,37,0,12,0,4326,0.4854368932,0.3492834027,0.1063337957,0,0,0.02981969487,0.01225150254,0.005547850208,0.008552935737,0,0.002773925104,0,"Casa Branca, SP, Brazil",2,1,1,1419
3511409,Cerqueira César,0,2,4,49,102,676,1896,746,485,172,93,12,1601,3305,266.8333333,847,1044,1152,292,2525,0,28,9,2,4,0,1,2,5059,0.2063648942,0.2277129868,0.05771891678,0.4991104961,0,0.00553469065,0.001779007709,0.0003953350465,0.0007906700929,0,0.0001976675232,0.0003953350465,"Cerqueira César, SP, Brazil",3,3,1,4237
3513108,Cravinhos,0,22,54,59,167,795,1338,445,389,318,51,19,1204,2561,200.6666667,0,1080,1119,446,1308,0,21,10,8,14,0,0,0,4006,0.2695956066,0.2793310035,0.1113330005,0.3265102346,0,0.005242136795,0.002496255617,0.001997004493,0.003494757863,0,0,0,"Cravinhos, SP, Brazil",2,3,1,3657
3513801,Diadema,0,0,0,0,48,97,36,16,6,0,0,0,959,613,159.8333333,613,426,237,76,0,0,31,3,6,19,0,0,0,798,0.5338345865,0.2969924812,0.09523809524,0,0,0.03884711779,0.003759398496,0.007518796992,0.02380952381,0,0,0,"Diadema, SP, Brazil",1,1,1,203
3514403,Dracena,0,0,4,7,84,420,409,94,54,13,0,0,633,844,105.5,0,411,608,29,0,0,31,0,1,4,0,1,0,1085,0.3788018433,0.5603686636,0.0267281106,0,0,0.02857142857,0,0.0009216589862,0.003686635945,0,0.0009216589862,0,"Dracena, SP, Brazil",2,1,1,1085
3516002,Flórida Paulista,0,0,0,0,26,217,404,276,232,51,2,0,288,844,48,0,518,581,109,0,0,0,0,0,0,0,0,0,1208,0.428807947,0.4809602649,0.09023178808,0,0,0,0,0,0,0,0,0,"Flórida Paulista, SP, Brazil",1,1,1,1208
3516200,Franca,0,0,65,85,539,143,21,17,1,0,0,0,704,847,117.3333333,0,425,628,126,0,0,4,0,4,0,0,0,0,1187,0.3580454928,0.5290648694,0.1061499579,0,0,0.003369839933,0,0.003369839933,0,0,0,0,"Franca, SP, Brazil",5,1,1,871
3516408,Franco da Rocha,0,80,71,173,217,2449,1726,828,266,94,18,1,4131,6489,688.5,1005,3683,2525,861,107,3,63,17,35,9,1,0,2,7306,0.5041062141,0.3456063509,0.1178483438,0.01464549685,0.0004106214071,0.008623049548,0.00232685464,0.004790583082,0.001231864221,0.0001368738024,0,0.0002737476047,"Franco da Rocha, SP, Brazil",1,7,1,5923
3516606,Gália,0,5,4,26,89,677,827,407,421,117,38,3,774,1642,129,0,925,314,286,1181,1,7,9,0,1,0,0,0,2724,0.3395741557,0.1152716593,0.1049926579,0.4335535977,0.0003671071953,0.002569750367,0.003303964758,0,0.0003671071953,0,0,0,"Gália, SP, Brazil",2,2,1,2614
3516705,Garça,0,0,2,18,30,190,387,323,412,241,50,6,1282,1916,213.6666667,821,1076,1052,537,0,0,19,0,3,0,0,0,0,2687,0.4004465947,0.3915147004,0.1998511351,0,0,0.007071082992,0,0.001116486788,0,0,0,0,"Garça, SP, Brazil",4,2,1,1659
3517000,Getulina,0,0,0,0,143,419,205,149,147,179,7,7,259,857,43.16666667,0,764,295,178,1,0,18,0,2,0,0,0,0,1258,0.6073131955,0.2344992051,0.1414944356,0.0007949125596,0,0.01430842607,0,0.001589825119,0,0,0,0,"Getulina, SP, Brazil",2,1,1,1256
3518602,Guariba,0,0,0,10,51,125,35,11,8,2,0,0,428,837,71.33333333,0,97,121,11,0,0,7,2,0,2,0,0,2,242,0.4008264463,0.5,0.04545454545,0,0,0.02892561983,0.00826446281,0,0.00826446281,0,0,0.00826446281,"Guariba, SP, Brazil",2,1,1,242
3518800,Guarulhos,0,227,65,196,468,923,1383,756,443,154,31,7,5386,4449,897.6666667,1397,1872,1041,498,2254,226,76,21,67,47,7,1,0,6110,0.3063829787,0.1703764321,0.08150572831,0.368903437,0.03698854337,0.0124386252,0.003436988543,0.01096563011,0.007692307692,0.001145662848,0.0001636661211,0,"Guarulhos, SP, Brazil",1,4,1,4653
3519071,Hortolândia,0,15,20,104,353,1327,1182,394,286,73,15,1,3917,3524,652.8333333,844,2101,2103,469,0,0,26,27,8,52,7,0,17,4810,0.4367983368,0.4372141372,0.09750519751,0,0,0.005405405405,0.005613305613,0.001663201663,0.01081081081,0.001455301455,0,0.003534303534,"Hortolândia, SP, Brazil",1,4,1,3770
3520905,Ipaussu,0,28,0,9,20,273,400,251,177,54,12,1,508,871,84.66666667,0,294,862,24,0,28,19,0,0,0,0,0,0,1227,0.239608802,0.7025264874,0.0195599022,0,0.0228198859,0.01548492258,0,0,0,0,0,0,"Ipaussu, SP, Brazil",2,1,1,1225
3521804,Itaí,0,0,1,2,41,351,502,290,240,175,79,22,862,1618,143.6666667,0,390,515,95,1139,0,0,0,14,1,0,0,0,2154,0.1810584958,0.239090065,0.04410399257,0.5287836583,0,0,0,0.006499535747,0.0004642525534,0,0,0,"Itaí, SP, Brazil",1,1,1,1703
3522208,Itapecerica da Serra,0,0,0,0,0,5,0,0,0,0,0,0,873,851,145.5,845,432,539,110,0,0,4,2,41,4,4,5,0,1141,0.3786152498,0.472392638,0.09640666082,0,0,0.003505696757,0.001752848379,0.03593339176,0.003505696757,0.003505696757,0.004382120947,0,"Itapecerica da Serra, SP, Brazil",3,1,1,5
3522307,Itapetininga,0,0,1,38,164,2039,838,109,36,0,0,0,2572,2295,428.6666667,0,947,2194,107,0,0,11,2,10,1,0,4,0,3276,0.2890720391,0.6697191697,0.03266178266,0,0,0.003357753358,0.0006105006105,0.003052503053,0.0003052503053,0,0.001221001221,0,"Itapetininga, SP, Brazil",3,3,1,3225
3523503,Itatinga,0,18,2,53,109,411,175,25,19,7,0,0,574,847,95.66666667,0,378,637,166,0,0,12,26,15,0,1,0,0,1235,0.3060728745,0.5157894737,0.1344129555,0,0,0.00971659919,0.02105263158,0.01214574899,0,0.0008097165992,0,0,"Itatinga, SP, Brazil",1,1,1,819
3523602,Itirapina,0,0,12,102,524,676,568,502,312,245,12,0,1350,1926,225,0,1389,1289,249,0,27,19,1,25,2,0,0,0,3001,0.4628457181,0.4295234922,0.08297234255,0,0.008997001,0.006331222926,0.0003332222592,0.008330556481,0.0006664445185,0,0,0,"Itirapina, SP, Brazil",1,2,1,2953
3524303,Jaboticabal,0,4,24,36,69,261,194,72,28,1,0,0,912,847,152,0,416,584,219,0,0,9,0,12,3,0,0,12,1255,0.3314741036,0.4653386454,0.174501992,0,0,0.007171314741,0,0.009561752988,0.002390438247,0,0,0.009561752988,"Jaboticabal, SP, Brazil",3,1,1,689
3525102,Jardinópolis,0,39,48,139,164,488,289,109,86,38,10,0,1196,1080,199.3333333,0,637,520,173,0,0,32,35,2,3,0,0,8,1410,0.4517730496,0.3687943262,0.1226950355,0,0,0.02269503546,0.02482269504,0.001418439716,0.002127659574,0,0,0.005673758865,"Jardinópolis, SP, Brazil",1,1,1,1410
3525300,Jaú,0,0,1,2,2,103,90,11,2,0,0,0,83,214,13.83333333,0,43,128,45,0,0,1,1,0,1,0,0,0,219,0.196347032,0.5844748858,0.2054794521,0,0,0.004566210046,0.004566210046,0,0.004566210046,0,0,0,"Jaú, SP, Brazil",4,1,1,211
3525904,Jundiaí,1271,0,0,0,0,0,0,0,0,0,0,0,1693,847,282.1666667,847,394,539,293,0,0,10,8,6,21,0,0,0,1271,0.3099921322,0.4240755311,0.230527144,0,0,0.007867820614,0.006294256491,0.004720692368,0.01652242329,0,0,0,"Jundiaí, SP, Brazil",1,1,1,1271
3526001,Junqueirópolis,357,0,0,3,6,116,257,175,265,249,80,10,388,873,64.66666667,0,1216,0,215,2,0,0,0,67,59,0,7,0,1566,0.7765006386,0,0.1372924649,0.001277139208,0,0,0,0.04278416347,0.03767560664,0,0.004469987229,0,"Junqueirópolis, SP, Brazil",1,1,1,1518
3526902,Limeira,22,1,2,12,25,546,705,154,70,18,0,0,856,1064,142.6666667,0,475,788,128,165,22,0,2,0,0,0,0,1,1581,0.3004427577,0.4984187223,0.08096141682,0.1043643264,0.01391524352,0,0.001265022138,0,0,0,0,0.0006325110689,"Limeira, SP, Brazil",2,2,1,1555
3527108,Lins,0,20,0,2,9,91,3,3,0,0,0,0,152,214,25.33333333,0,62,120,27,0,2,3,0,0,0,0,0,0,214,0.2897196262,0.5607476636,0.1261682243,0,0.009345794393,0.01401869159,0,0,0,0,0,0,"Lins, SP, Brazil",3,1,1,128
3527405,Lucélia,53,2,13,38,63,953,1130,491,404,194,75,22,1142,2394,190.3333333,0,860,1059,318,2039,50,6,101,0,6,0,0,0,4439,0.1937373282,0.2385672449,0.07163775625,0.4593376887,0.01126379815,0.001351655778,0.02275287227,0,0.001351655778,0,0,0,"Lucélia, SP, Brazil",3,2,1,3438
3528403,Mairinque,0,0,0,5,43,286,530,243,129,36,9,0,435,847,72.5,0,420,573,224,0,0,45,0,19,0,0,0,0,1281,0.3278688525,0.4473067916,0.174863388,0,0,0.03512880562,0,0.01483216237,0,0,0,0,"Mairinque, SP, Brazil",2,1,1,1281
3529005,Marília,0,0,85,83,62,713,592,122,70,10,3,0,899,1406,149.8333333,0,725,848,135,0,0,14,1,7,6,0,4,0,1740,0.4166666667,0.4873563218,0.0775862069,0,0,0.008045977011,0.0005747126437,0.004022988506,0.003448275862,0,0.002298850575,0,"Marília, SP, Brazil",3,2,1,1740
3529203,Martinópolis,0,0,0,0,0,31,42,15,7,7,5,0,393,872,65.5,0,47,21,25,1,0,6,0,7,0,0,0,0,107,0.4392523364,0.1962616822,0.2336448598,0.009345794393,0,0.05607476636,0,0.06542056075,0,0,0,0,"Martinópolis, SP, Brazil",2,1,1,107
3529401,Mauá,0,0,0,0,10,71,61,19,15,8,0,0,906,633,151,630,480,249,98,0,0,10,0,3,7,0,0,3,850,0.5647058824,0.2929411765,0.1152941176,0,0,0.01176470588,0,0.003529411765,0.008235294118,0,0,0.003529411765,"Mauá, SP, Brazil",1,1,1,184
3530102,Mirandópolis,133,7,0,26,107,1097,1403,1235,995,370,129,45,2758,6386,459.6666667,0,5966,3969,1463,4,0,486,13,1205,65,1,19,0,13191,0.4522780684,0.3008869684,0.1109089531,0.0003032370556,0,0.03684330225,0.0009855204306,0.09135016299,0.004927602153,7.58E-05,0.001440376014,0,"Mirandópolis, SP, Brazil",3,6,1,5547
3530508,Mococa,0,0,0,3,7,96,63,9,10,0,0,0,128,214,21.33333333,0,35,123,28,0,0,1,0,1,0,0,0,0,188,0.1861702128,0.6542553191,0.1489361702,0,0,0.005319148936,0,0.005319148936,0,0,0,0,"Mococa, SP, Brazil",1,1,1,188
3530607,Mogi das Cruzes,0,0,0,10,23,105,91,7,7,0,0,0,458,844,76.33333333,838,429,415,84,0,0,12,1,17,0,0,1,0,959,0.4473409802,0.43274244,0.08759124088,0,0,0.01251303441,0.001042752868,0.01772679875,0,0,0.001042752868,0,"Mogi das Cruzes, SP, Brazil",2,1,1,243
3530706,Mogi Guaçu,242,13,24,159,234,89,30,0,0,0,0,0,587,849,97.83333333,0,231,545,72,0,0,7,1,9,4,4,0,0,873,0.264604811,0.6242840779,0.0824742268,0,0,0.008018327606,0.001145475372,0.01030927835,0.004581901489,0.004581901489,0,0,"Mogi Guaçu, SP, Brazil",2,1,1,791
3530805,Moji Mirim,0,0,0,0,0,31,112,47,7,2,0,0,81,215,13.5,0,2,1,4,192,0,0,0,0,0,0,0,0,199,0.01005025126,0.005025125628,0.02010050251,0.9648241206,0,0,0,0,0,0,0,0,"Moji Mirim, SP, Brazil",3,1,1,199
3531100,Mongaguá,0,164,267,246,492,564,205,41,61,10,1,0,1529,1640,254.8333333,0,1388,398,194,2,0,30,10,10,16,0,3,0,2051,0.6767430522,0.1940516821,0.09458800585,0.0009751340809,0,0.01462701121,0.004875670405,0.004875670405,0.007801072647,0,0.001462701121,0,"Mongaguá, SP, Brazil",1,1,1,2051
3533007,Nova Granada,0,0,0,0,132,649,284,70,23,4,0,0,579,847,96.5,0,317,751,72,0,0,11,1,4,5,0,0,1,1162,0.2728055077,0.6462994836,0.06196213425,0,0,0.009466437177,0.0008605851979,0.003442340792,0.00430292599,0,0,0.0008605851979,"Nova Granada, SP, Brazil",3,1,1,1162
3534401,Osasco,0,0,10,27,60,127,92,43,27,10,0,0,1778,1666,296.3333333,1666,1228,634,293,0,0,55,4,46,16,0,0,3,2279,0.5388328214,0.2781921896,0.1285651602,0,0,0.02413339184,0.00175515577,0.02018429136,0.00702062308,0,0,0.001316366828,"Osasco, SP, Brazil",1,2,1,396
3534609,Osvaldo Cruz,0,0,3,14,57,424,319,64,27,10,2,2,857,844,142.8333333,0,305,502,37,0,0,8,37,4,8,0,3,0,904,0.3373893805,0.5553097345,0.04092920354,0,0,0.008849557522,0.04092920354,0.004424778761,0.008849557522,0,0.003318584071,0,"Osvaldo Cruz, SP, Brazil",4,1,1,922
3534708,Ourinhos,0,0,0,1,3,68,33,11,0,0,0,0,59,123,9.833333333,0,18,97,20,0,0,1,0,0,0,0,0,1,137,0.1313868613,0.7080291971,0.1459854015,0,0,0.007299270073,0,0,0,0,0,0.007299270073,"Ourinhos, SP, Brazil",3,1,1,116
3534906,Pacaembu,71,3,7,50,239,1215,1756,753,489,177,37,10,2484,4049,414,823,4428,3376,888,40,0,189,56,47,49,0,7,0,9080,0.4876651982,0.3718061674,0.09779735683,0.004405286344,0,0.02081497797,0.006167400881,0.005176211454,0.005396475771,0,0.0007709251101,0,"Pacaembu, SP, Brazil",3,5,1,4807
3535507,Paraguaçu Paulista,0,0,0,0,0,0,0,0,0,0,0,0,639,844,106.5,0,249,167,347,0,0,0,0,0,0,0,0,0,763,0.3263433814,0.2188728702,0.4547837484,0,0,0,0,0,0,0,0,0,"Paraguaçu Paulista, SP, Brazil",4,1,1,0
3536604,Paulo de Faria,0,6,4,19,94,525,735,247,191,85,25,0,1975,2535,329.1666667,823,989,2010,481,1,148,15,12,9,6,4,0,8,3683,0.2685310888,0.5457507467,0.1306000543,0.0002715177844,0.04018463209,0.004072766766,0.003258213413,0.00244366006,0.001629106706,0.001086071138,0,0.002172142275,"Paulo de Faria, SP, Brazil",3,3,1,1931
3537305,Penápolis,0,4,0,4,20,521,416,112,75,17,1,0,382,844,63.66666667,0,501,613,47,0,4,4,0,2,0,0,0,0,1171,0.4278394535,0.5234842015,0.04013663535,0,0.00341588386,0.00341588386,0,0.00170794193,0,0,0,0,"Penápolis, SP, Brazil",7,1,1,1170
3538709,Piracicaba,0,0,3,45,134,661,575,173,102,35,5,0,1737,1550,289.5,514,957,1029,261,0,0,24,2,15,2,2,5,0,2297,0.4166303875,0.4479756204,0.1136264693,0,0,0.01044841097,0.0008707009142,0.006530256857,0.0008707009142,0.0008707009142,0.002176752286,0,"Piracicaba, SP, Brazil",3,3,1,1733
3538907,Pirajuí,0,33,8,68,281,3349,3324,946,624,329,93,17,2931,6550,488.5,0,4033,4367,800,2,24,75,15,34,5,2,0,0,9357,0.431014214,0.4667094154,0.08549748851,0.0002137437213,0.002564924655,0.008015389548,0.00160307791,0.003633643262,0.0005343593032,0.0002137437213,0,0,"Pirajuí, SP, Brazil",6,7,1,9072
3540200,Pontal,0,40,0,0,42,301,266,102,76,15,0,0,807,847,134.5,0,300,619,197,0,40,5,1,2,0,0,0,2,1166,0.2572898799,0.5308747856,0.1689536878,0,0.03430531732,0.004288164666,0.0008576329331,0.001715265866,0,0,0,0.001715265866,"Pontal, SP, Brazil",1,1,1,842
3540507,Porangaba,0,0,5,26,176,669,1200,525,235,92,25,3,1399,1688,233.1666667,0,1555,1020,316,0,0,56,0,7,0,0,0,0,2954,0.5264048747,0.3452945159,0.1069735951,0,0,0.01895734597,0,0.002369668246,0,0,0,0,"Porangaba, SP, Brazil",3,2,1,2956
3540606,Porto Feliz,0,7,19,102,151,399,345,184,211,91,27,1,1382,1080,230.3333333,0,662,667,151,0,0,22,21,5,7,0,0,2,1537,0.4307091737,0.4339622642,0.09824333116,0,0,0.01431359792,0.01366297983,0.003253090436,0.00455432661,0,0,0.001301236174,"Porto Feliz, SP, Brazil",1,1,1,1537
3541000,Praia Grande,0,0,4,4,4,66,67,22,5,4,1,0,740,564,123.3333333,540,350,280,113,2,0,1,1,1,0,0,0,0,748,0.4679144385,0.3743315508,0.1510695187,0.002673796791,0,0.001336898396,0.001336898396,0.001336898396,0,0,0,0,"Praia Grande, SP, Brazil",1,1,1,177
3541208,Presidente Bernardes,2,3,24,37,78,457,686,440,358,208,37,6,973,1636,162.1666667,0,1188,919,214,1,0,11,0,3,0,0,0,0,2336,0.5085616438,0.3934075342,0.09160958904,0.0004280821918,0,0.00470890411,0,0.001284246575,0,0,0,0,"Presidente Bernardes, SP, Brazil",2,2,1,2336
3541307,Presidente Epitácio,0,0,0,17,34,509,382,61,36,7,0,0,1681,1667,280.1666667,844,536,924,207,0,0,18,0,4,8,0,1,0,1698,0.3156654888,0.5441696113,0.1219081272,0,0,0.01060070671,0,0.002355712603,0.004711425206,0,0.0005889281508,0,"Presidente Epitácio, SP, Brazil",2,2,1,1046
3541406,Presidente Prudente,72,33,6,7,19,394,328,432,216,21,3,2,898,1157,149.6666667,61,1287,521,327,81,30,69,5,8,23,2,2,5,2360,0.5453389831,0.2207627119,0.138559322,0.0343220339,0.01271186441,0.02923728814,0.002118644068,0.003389830508,0.009745762712,0.0008474576271,0.0008474576271,0.002118644068,"Presidente Prudente, SP, Brazil",5,2,1,1533
3541505,Presidente Venceslau,68,0,4,7,66,453,919,161,189,203,145,48,1549,3017,258.1666667,0,1543,776,432,358,0,16,4,108,86,1,12,7,3343,0.4615614717,0.2321268322,0.1292252468,0.1070894406,0,0.004786120251,0.001196530063,0.0323063117,0.02572539635,0.0002991325157,0.003589590188,0.00209392761,"Presidente Venceslau, SP, Brazil",2,3,1,2263
3542602,Registro,0,19,18,23,160,151,147,102,58,26,7,0,220,823,36.66666667,0,359,618,180,0,0,24,1,11,11,0,0,2,1206,0.2976782753,0.5124378109,0.1492537313,0,0,0.01990049751,0.0008291873964,0.00912106136,0.00912106136,0,0,0.001658374793,"Registro, SP, Brazil",2,1,1,711
3543402,Ribeirão Preto,0,19,31,232,365,682,296,193,108,47,13,5,1835,1964,305.8333333,586,1104,956,309,174,0,25,4,12,18,1,0,1,2604,0.4239631336,0.3671274962,0.1186635945,0.0668202765,0,0.009600614439,0.00153609831,0.004608294931,0.006912442396,0.0003840245776,0,0.0003840245776,"Ribeirão Preto, SP, Brazil",2,3,1,1991
3543907,Rio Claro,5,0,0,14,93,170,36,4,3,2,0,0,144,351,24,0,32,75,64,167,0,1,0,1,0,1,0,0,341,0.09384164223,0.219941349,0.1876832845,0.4897360704,0,0.00293255132,0,0.00293255132,0,0.00293255132,0,0,"Rio Claro, SP, Brazil",5,2,1,327
3547809,Santo André,0,1,4,14,25,118,64,7,13,1,2,0,977,534,162.8333333,534,482,154,65,0,0,9,4,6,18,0,0,0,738,0.6531165312,0.2086720867,0.08807588076,0,0,0.01219512195,0.005420054201,0.008130081301,0.0243902439,0,0,0,"Santo André, SP, Brazil",1,1,1,249
3548708,São Bernardo do Campo,0,0,5,18,31,79,23,4,2,3,0,0,754,844,125.6666667,844,426,176,67,0,0,14,4,14,3,6,0,0,710,0.6,0.2478873239,0.09436619718,0,0,0.01971830986,0.005633802817,0.01971830986,0.004225352113,0.008450704225,0,0,"São Bernardo do Campo, SP, Brazil",1,1,1,165
3549805,São José do Rio Preto,0,8,10,62,94,564,405,152,98,43,5,1,2935,2137,489.1666667,858,748,1105,456,0,0,33,24,5,5,1,2,4,2383,0.3138900546,0.463701217,0.1913554343,0,0,0.01384809064,0.01007133865,0.002098195552,0.002098195552,0.0004196391104,0.0008392782207,0.001678556441,"São José do Rio Preto, SP, Brazil",6,3,1,1442
3549904,São José dos Campos,0,1,1,5,12,119,45,5,1,0,0,0,821,735,136.8333333,546,216,311,111,0,1,14,0,6,0,0,0,0,659,0.3277693475,0.4719271624,0.1684370258,0,0.001517450683,0.02124430956,0,0.009104704097,0,0,0,0,"São José dos Campos, SP, Brazil",2,2,1,189
3550308,São Paulo,0,7,34,187,566,1808,1730,499,329,300,116,7,17637,19015,2939.5,4971,6051,3736,1357,463,0,74,15,138,32,3,2,51,11922,0.5075490689,0.3133702399,0.113823184,0.03883576581,0,0.006207012246,0.001258178158,0.01157523905,0.002684113404,0.0002516356316,0.0001677570877,0.004277805737,"São Paulo, SP, Brazil",1,13,1,5583
3551009,São Vicente,1046,43,9,63,402,900,890,441,224,42,8,0,1802,3862,300.3333333,843,2473,1980,461,0,41,50,18,57,6,1,0,7,5094,0.4854731056,0.3886925795,0.09049862583,0,0.008048684727,0.009815469179,0.003533568905,0.01118963486,0.001177856302,0.0001963093836,0,0.001374165685,"São Vicente, SP, Brazil",1,4,1,4068
3552205,Sorocaba,342,97,31,79,199,450,664,256,188,95,68,19,3261,2267,543.5,712,972,290,495,1679,4,4,80,10,7,0,1,1,3543,0.2743437765,0.08185153824,0.1397121084,0.4738921818,0.001128986734,0.001128986734,0.02257973469,0.002822466836,0.001975726785,0,0.0002822466836,0.0002822466836,"Sorocaba, SP, Brazil",2,3,1,2488
3552403,Sumaré,0,6,0,0,5,39,31,8,7,2,0,0,625,223,104.1666667,0,31,42,22,0,45,3,0,0,0,0,0,0,143,0.2167832168,0.2937062937,0.1538461538,0,0.3146853147,0.02097902098,0,0,0,0,0,0,"Sumaré, SP, Brazil",1,1,1,98
3552502,Suzano,0,1,0,6,16,167,83,29,9,5,0,0,763,844,127.1666667,835,554,201,170,0,0,5,0,17,0,0,0,0,947,0.5850052798,0.212249208,0.1795142555,0,0,0.005279831045,0,0.01795142555,0,0,0,0,"Suzano, SP, Brazil",1,1,1,316
3553807,Taquarituba,0,0,3,5,223,550,236,135,63,12,1,0,393,847,65.5,0,561,615,47,0,0,7,0,0,0,0,0,0,1230,0.456097561,0.5,0.03821138211,0,0,0.005691056911,0,0,0,0,0,0,"Taquarituba, SP, Brazil",2,1,1,1228
3554003,Tatuí,0,1,0,11,11,589,1106,429,304,114,15,1,1366,1694,227.6666667,0,842,1459,212,0,22,18,7,21,0,0,0,0,2581,0.3262301434,0.5652847733,0.08213870593,0,0.008523827974,0.006974041069,0.002712127083,0.008136381248,0,0,0,0,"Tatuí, SP, Brazil",4,2,1,2581
3554102,Taubaté,0,0,0,9,4,135,32,5,3,0,0,0,887,1128,147.8333333,844,365,512,358,37,0,38,2,3,0,0,0,0,1315,0.2775665399,0.3893536122,0.272243346,0.02813688213,0,0.0288973384,0.001520912548,0.002281368821,0,0,0,0,"Taubaté, SP, Brazil",2,2,1,188
3554805,Tremembé,2871,59,5,31,79,877,1000,541,451,250,100,12,2998,6018,499.6666667,71,3199,1445,1079,242,50,141,68,83,94,15,59,7,6482,0.4935205184,0.2229250231,0.1664609688,0.03733415612,0.007713668621,0.02175254551,0.01049058932,0.01280468991,0.01450169701,0.002314100586,0.009102128973,0.001079913607,"Tremembé, SP, Brazil",1,5,1,6276
3555109,Tupi Paulista,601,2,1,13,40,378,373,201,190,110,21,13,735,1634,122.5,357,2700,812,802,0,0,65,5,10,19,0,0,0,4413,0.611828688,0.1840018128,0.1817357806,0,0,0.01472920915,0.001133016089,0.002266032178,0.004305461138,0,0,0,"Tupi Paulista, SP, Brazil",4,2,1,1943
3556305,Valparaíso,24,6,11,20,39,324,383,272,361,322,149,16,1653,1564,275.5,0,981,632,213,3,24,44,11,43,0,0,0,0,1951,0.5028190671,0.3239364428,0.1091747822,0.001537672988,0.01230138391,0.02255253716,0.00563813429,0.0220399795,0,0,0,0,"Valparaíso, SP, Brazil",2,2,1,1927
3557006,Votorantim,141,1,1,7,18,185,131,45,89,13,6,1,458,842,76.33333333,0,212,380,43,3,2,0,0,0,0,0,0,0,640,0.33125,0.59375,0.0671875,0.0046875,0.003125,0,0,0,0,0,0,0,"Votorantim, SP, Brazil",1,1,1,638
4100509,Altônia,41,0,0,0,0,0,0,0,0,0,0,0,133,16,22.16666667,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Altônia, PR, Brazil",2,1,0,41
4100608,Alto Paraná,57,0,0,0,0,0,0,0,0,0,0,0,85,28,14.16666667,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Alto Paraná, PR, Brazil",3,1,0,57
4101101,Andirá,137,0,0,0,0,0,0,0,0,0,0,0,0,43,0,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Andirá, PR, Brazil",3,1,0,137
4101408,Apucarana,350,0,0,0,0,0,0,0,0,0,0,0,660,168,110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Apucarana, PR, Brazil",3,1,0,350
4101507,Arapongas,181,0,0,0,0,0,0,0,0,0,0,0,498,141,83,53,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Arapongas, PR, Brazil",2,1,0,181
4101606,Arapoti,37,0,0,1,0,7,8,6,5,2,1,0,57,33,9.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Arapoti, PR, Brazil",1,1,0,67
4101804,Araucária,122,0,0,0,0,0,0,0,0,0,0,0,786,24,131,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Araucária, PR, Brazil",1,1,0,122
4101903,Assaí,175,0,0,0,0,0,0,0,0,0,0,0,211,109,35.16666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Assaí, PR, Brazil",3,1,0,175
4102000,Assis Chateaubriand,89,0,0,0,0,0,0,0,0,0,0,0,401,90,66.83333333,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Assis Chateaubriand, PR, Brazil",2,1,0,89
4102109,Astorga,70,0,0,0,0,0,0,0,0,0,0,0,197,35,32.83333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Astorga, PR, Brazil",2,1,0,70
4102406,Bandeirantes,139,0,0,0,0,0,0,0,0,0,0,0,203,51,33.83333333,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Bandeirantes, PR, Brazil",2,1,0,139
4103602,Cambará,108,0,0,0,0,0,0,0,0,0,0,0,33,106,5.5,19,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cambará, PR, Brazil",1,1,0,108
4103701,Cambé,184,0,0,0,0,0,0,0,0,0,0,0,322,86,53.66666667,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cambé, PR, Brazil",1,1,0,184
4103909,Campina da Lagoa,51,0,0,0,0,0,0,0,0,0,0,0,135,18,22.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Campina da Lagoa, PR, Brazil",3,1,0,51
4104204,Campo Largo,122,0,0,0,0,0,0,0,0,0,0,0,502,35,83.66666667,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Campo Largo, PR, Brazil",2,1,0,122
4104303,Campo Mourão,537,0,0,0,3,14,17,6,6,0,0,0,468,468,78,28,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Campo Mourão, PR, Brazil",4,2,0,583
4104501,Capanema,106,0,0,0,0,0,0,0,0,0,0,0,128,85,21.33333333,85,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Capanema, PR, Brazil",4,1,0,106
4104709,Carlópolis,47,0,0,0,0,0,0,0,0,0,0,0,52,57,8.666666667,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Carlópolis, PR, Brazil",3,1,0,47
4104808,Cascavel,4875,0,0,0,0,18,138,77,87,48,11,0,1620,1336,270,16,397,814,554,103,7,132,23,47,27,1,4,4,2113,0.1878845244,0.3852342641,0.2621864647,0.04874585897,0.003312825367,0.0624704212,0.01088499763,0.02224325603,0.0127780407,0.0004732607667,0.001893043067,0.001893043067,"Cascavel, PR, Brazil",1,4,1,5254
4104907,Castro,120,0,0,0,0,0,0,0,0,0,0,0,335,60,55.83333333,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Castro, PR, Brazil",2,1,0,120
4105003,Catanduvas,0,0,0,7,8,16,28,26,22,34,5,2,32,208,5.333333333,0,185,143,98,0,0,125,15,42,50,0,9,0,667,0.2773613193,0.2143928036,0.1469265367,0,0,0.1874062969,0.02248875562,0.06296851574,0.07496251874,0,0.01349325337,0,"Catanduvas, PR, Brazil",3,1,1,148
4105508,Cianorte,179,0,0,0,0,0,0,0,0,0,0,0,519,130,86.5,130,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cianorte, PR, Brazil",6,1,0,179
4105607,Cidade Gaúcha,11,0,0,0,1,3,13,3,6,1,0,0,155,21,25.83333333,21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cidade Gaúcha, PR, Brazil",5,1,0,38
4105805,Colombo,0,0,0,0,0,0,4,0,1,2,0,0,1243,54,207.1666667,54,8,5,4,4,0,0,2,0,0,0,0,0,23,0.347826087,0.2173913043,0.1739130435,0.1739130435,0,0,0.08695652174,0,0,0,0,0,"Colombo, PR, Brazil",1,1,1,7
4105904,Colorado,109,0,0,0,0,0,0,0,0,0,0,0,155,80,25.83333333,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Colorado, PR, Brazil",4,1,0,109
4106308,Corbélia,79,0,0,0,0,0,0,0,0,0,0,0,100,23,16.66666667,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Corbélia, PR, Brazil",5,1,0,79
4106407,Cornélio Procópio,376,0,0,0,0,0,0,0,0,0,0,0,374,287,62.33333333,275,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cornélio Procópio, PR, Brazil",3,1,0,376
4106605,Cruzeiro do Oeste,151,0,4,9,13,204,291,174,158,186,45,3,180,960,30,0,985,1068,352,37,0,68,122,62,24,1,9,1,2729,0.3609380726,0.3913521436,0.1289849762,0.01355807988,0,0.02491755222,0.04470502015,0.02271894467,0.008794430194,0.0003664345914,0.003297911323,0.0003664345914,"Cruzeiro do Oeste, PR, Brazil",4,1,1,1238
4106902,Curitiba,26439,4,0,2,3,17,257,132,147,126,49,9,8365,19944,1394.166667,768,267,88,185,861,0,39,16,11,12,0,2,0,1481,0.1802835922,0.05941931128,0.1249155976,0.5813639433,0,0.02633355841,0.01080351114,0.00742741391,0.008102633356,0,0.001350438893,0,"Curitiba, PR, Brazil",1,4,1,27185
4107207,Dois Vizinhos,174,0,0,0,0,0,0,0,0,0,0,0,215,174,35.83333333,94,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Dois Vizinhos, PR, Brazil",5,1,0,174
4107504,Engenheiro Beltrão,0,1,0,1,2,4,11,6,0,2,0,0,89,18,14.83333333,18,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Engenheiro Beltrão, PR, Brazil",3,1,0,27
4107603,Faxinal,124,0,0,0,0,0,0,0,0,0,0,0,199,66,33.16666667,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Faxinal, PR, Brazil",3,1,0,124
4108304,Foz do Iguaçu,634,7,3,10,48,346,1010,577,590,325,71,2,2334,3352,389,435,1240,755,496,2,14,188,99,18,22,0,30,1,2865,0.4328097731,0.2635253054,0.1731239092,0.0006980802792,0.004886561955,0.06561954625,0.03455497382,0.006282722513,0.007678883072,0,0.01047120419,0.0003490401396,"Foz do Iguaçu, PR, Brazil",2,6,1,3623
4108403,Francisco Beltrão,2398,0,0,0,0,0,0,0,0,0,0,0,944,1218,157.3333333,219,1000,945,408,89,262,58,49,49,15,0,2,1,2878,0.3474635163,0.3283530229,0.1417651147,0.03092425295,0.09103544128,0.02015288395,0.0170257123,0.0170257123,0.005211952745,0,0.0006949270327,0.0003474635163,"Francisco Beltrão, PR, Brazil",3,3,1,2398
4108601,Goioerê,0,0,0,0,0,23,35,10,6,0,0,0,181,43,30.16666667,0,28,40,5,3,0,0,0,0,0,0,0,0,76,0.3684210526,0.5263157895,0.06578947368,0.03947368421,0,0,0,0,0,0,0,0,"Goioerê, PR, Brazil",4,1,1,74
4108809,Guaíra,910,0,0,0,0,0,0,0,0,0,0,0,691,770,115.1666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Guaíra, PR, Brazil",1,1,0,910
4109401,Guarapuava,926,51,98,81,75,71,128,85,120,88,30,4,1135,835,189.1666667,180,1079,911,599,272,1,153,160,15,21,4,5,3,3223,0.3347812597,0.2826559106,0.185851691,0.08439342228,0.0003102699348,0.04747130003,0.04964318957,0.004654049023,0.006515668632,0.001241079739,0.001551349674,0.0009308098045,"Guarapuava, PR, Brazil",5,4,1,1757
4109609,Guaratuba,80,0,0,0,0,0,0,0,0,0,0,0,679,66,113.1666667,66,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Guaratuba, PR, Brazil",1,1,0,80
4109708,Ibaiti,0,0,0,0,0,0,0,0,0,0,0,0,26,57,4.333333333,57,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ibaiti, PR, Brazil",3,1,0,0
4109807,Ibiporã,170,0,0,0,0,0,0,0,0,0,0,0,198,60,33,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ibiporã, PR, Brazil",1,1,0,170
4110607,Iporã,41,0,0,0,0,0,0,0,0,0,0,0,213,47,35.5,10,15,0,6,0,0,0,0,0,0,0,0,0,21,0.7142857143,0,0.2857142857,0,0,0,0,0,0,0,0,0,"Iporã, PR, Brazil",3,1,1,41
4110706,Irati,118,0,0,0,0,0,0,0,0,0,0,0,539,60,89.83333333,60,21,15,72,47,0,5,0,0,2,0,0,1,163,0.1288343558,0.09202453988,0.4417177914,0.2883435583,0,0.03067484663,0,0,0.01226993865,0,0,0.006134969325,"Irati, PR, Brazil",2,1,1,118
4111506,Ivaiporã,146,0,0,0,0,0,0,0,0,0,0,0,110,107,18.33333333,107,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ivaiporã, PR, Brazil",5,1,0,146
4111803,Jacarezinho,116,0,0,0,0,0,0,0,0,0,0,0,88,118,14.66666667,118,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Jacarezinho, PR, Brazil",1,1,0,116
4112009,Jaguariaíva,132,0,0,0,0,0,0,0,0,0,0,0,323,125,53.83333333,125,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Jaguariaíva, PR, Brazil",1,1,0,132
4112108,Jandaia do Sul,15,0,0,0,0,5,20,8,10,3,0,0,178,25,29.66666667,25,2,0,3,56,0,0,0,0,0,0,0,0,61,0.03278688525,0,0.04918032787,0.9180327869,0,0,0,0,0,0,0,0,"Jandaia do Sul, PR, Brazil",5,1,1,61
4113304,Laranjeiras do Sul,129,0,0,0,0,0,0,0,0,0,0,0,272,86,45.33333333,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Laranjeiras do Sul, PR, Brazil",5,1,0,129
4113502,Loanda,127,0,0,0,0,0,0,0,0,0,0,0,281,96,46.83333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Loanda, PR, Brazil",5,1,0,127
4113700,Londrina,6186,0,0,0,2,28,195,182,147,119,39,8,5651,3228,941.8333333,450,2303,1412,611,140,168,154,55,23,44,0,7,0,4917,0.4683750254,0.2871669717,0.1242627618,0.02847264592,0.03416717511,0.03131991051,0.01118568233,0.004677648973,0.008948545861,0,0.001423632296,0,"Londrina, PR, Brazil",2,8,1,6906
4114104,Mandaguaçu,55,0,0,0,0,0,0,0,0,0,0,0,0,28,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Mandaguaçu, PR, Brazil",3,1,0,55
4114203,Mandaguari,9,0,0,0,0,0,47,29,12,0,0,0,119,45,19.83333333,9,0,0,0,97,97,0,0,0,0,0,0,0,194,0,0,0,0.5,0.5,0,0,0,0,0,0,0,"Mandaguari, PR, Brazil",1,1,1,97
4114500,Manoel Ribas,73,0,0,0,0,0,0,0,0,0,0,0,177,50,29.5,50,12,9,38,14,0,0,0,0,0,0,0,0,73,0.1643835616,0.1232876712,0.5205479452,0.1917808219,0,0,0,0,0,0,0,0,"Manoel Ribas, PR, Brazil",2,1,1,73
4114609,Marechal Cândido Rondon,85,0,0,0,0,0,0,0,0,0,0,0,279,53,46.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Marechal Cândido Rondon, PR, Brazil",6,1,0,85
4114807,Marialva,34,0,0,0,0,10,23,12,8,1,0,0,43,40,7.166666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Marialva, PR, Brazil",2,1,0,88
4115200,Maringá,531,0,2,2,17,298,512,427,317,102,23,7,3063,1897,510.5,1059,1997,1041,605,93,0,240,69,3,24,0,21,1,4094,0.4877870054,0.2542745481,0.147777235,0.02271617,0,0.05862237421,0.01685393258,0.0007327796776,0.005862237421,0,0.005129457743,0.0002442598925,"Maringá, PR, Brazil",5,4,1,2238
4115804,Medianeira,112,0,0,0,0,0,0,0,0,0,0,0,617,90,102.8333333,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Medianeira, PR, Brazil",3,1,0,112
4116901,Nova Esperança,59,0,0,0,0,0,0,0,0,0,0,0,123,30,20.5,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Nova Esperança, PR, Brazil",5,1,0,59
4117107,Nova Londrina,68,0,0,0,0,0,0,0,0,0,0,0,161,43,26.83333333,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Nova Londrina, PR, Brazil",4,1,0,68
4117305,Ortigueira,57,0,0,0,0,0,0,0,0,0,0,0,157,63,26.16666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ortigueira, PR, Brazil",1,1,0,57
4117602,Palmas,161,0,0,0,0,0,0,0,0,0,0,0,410,79,68.33333333,79,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Palmas, PR, Brazil",2,1,0,161
4117909,Palotina,7,0,0,0,0,3,13,6,6,6,6,0,36,40,6,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Palotina, PR, Brazil",2,1,0,47
4118105,Paranacity,107,0,0,0,0,0,0,0,0,0,0,0,162,24,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Paranacity, PR, Brazil",5,1,0,107
4118204,Paranaguá,120,0,0,0,0,0,0,0,0,0,0,0,834,56,139,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Paranaguá, PR, Brazil",1,1,0,120
4118402,Paranavaí,273,0,0,0,0,0,0,0,0,0,0,0,575,116,95.83333333,116,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Paranavaí, PR, Brazil",4,1,0,273
4118501,Pato Branco,481,0,0,0,0,0,0,0,0,0,0,0,889,479,148.1666667,303,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Pato Branco, PR, Brazil",4,1,0,481
4119152,Pinhais,646,0,0,0,0,0,0,0,0,0,0,0,741,532,123.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Pinhais, PR, Brazil",1,1,0,646
4119301,Pinhão,47,0,0,0,0,0,0,0,0,0,0,0,174,30,29,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Pinhão, PR, Brazil",2,1,0,47
4119509,Piraquara,6014,0,1,15,80,252,279,201,94,57,25,6,5279,6462,879.8333333,1528,831,906,299,46,0,305,67,40,12,1,11,0,2518,0.3300238284,0.3598093725,0.1187450357,0.01826846704,0,0.1211278793,0.02660841938,0.01588562351,0.004765687053,0.0003971405878,0.004368546465,0,"Piraquara, PR, Brazil",1,8,1,7024
4119608,Pitanga,61,0,0,0,0,0,0,0,0,0,0,0,238,61,39.66666667,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Pitanga, PR, Brazil",4,1,0,61
4119905,Ponta Grossa,964,0,0,25,120,343,444,290,62,9,3,0,1767,1848,294.5,355,405,247,293,85,0,106,91,18,6,2,4,1,1258,0.3219395866,0.1963434022,0.23290938,0.06756756757,0,0.08426073132,0.07233704293,0.01430842607,0.004769475358,0.001589825119,0.003179650238,0.0007949125596,"Ponta Grossa, PR, Brazil",1,4,1,2260
4120002,Porecatu,121,0,0,0,0,0,0,0,0,0,0,0,208,57,34.66666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Porecatu, PR, Brazil",4,1,0,121
4120606,Prudentópolis,93,0,0,0,0,0,0,0,0,0,0,0,15,50,2.5,24,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Prudentópolis, PR, Brazil",1,1,0,93
4120903,Quedas do Iguaçu,74,0,0,0,0,0,0,0,0,0,0,0,184,22,30.66666667,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Quedas do Iguaçu, PR, Brazil",2,1,0,74
4121703,Reserva,82,0,0,0,0,0,0,0,0,0,0,0,102,43,17,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Reserva, PR, Brazil",1,1,0,82
4122305,Rio Negro,65,0,0,0,0,0,0,0,0,0,0,0,210,54,35,54,1,1,0,63,0,0,0,0,0,0,0,0,65,0.01538461538,0.01538461538,0,0.9692307692,0,0,0,0,0,0,0,0,"Rio Negro, PR, Brazil",4,1,1,65
4122404,Rolândia,193,0,0,0,0,0,0,0,0,0,0,0,451,111,75.16666667,111,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Rolândia, PR, Brazil",2,1,0,193
4124103,Santo Antônio da Platina,0,0,0,0,0,0,0,0,0,0,0,0,10,68,1.666666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Santo Antônio da Platina, PR, Brazil",1,1,0,0
4124400,Santo Antônio do Sudoeste,0,2,1,2,4,15,18,4,4,2,1,0,261,26,43.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Santo Antônio do Sudoeste, PR, Brazil",3,1,0,53
4125506,São José dos Pinhais,1037,0,0,0,0,0,0,0,0,0,0,0,283,876,47.16666667,676,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São José dos Pinhais, PR, Brazil",2,1,0,1037
4125605,São Mateus do Sul,208,0,0,0,0,0,0,0,0,0,0,0,8,44,1.333333333,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São Mateus do Sul, PR, Brazil",2,1,0,208
4126256,Sarandi,206,0,0,0,0,0,0,0,0,0,0,0,244,68,40.66666667,68,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Sarandi, PR, Brazil",1,1,0,206
4126306,Sengés,0,0,0,0,0,0,0,0,0,0,0,0,64,63,10.66666667,60,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Sengés, PR, Brazil",1,1,0,0
4127106,Telêmaco Borba,240,0,0,0,0,0,0,0,0,0,0,0,524,123,87.33333333,123,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Telêmaco Borba, PR, Brazil",2,1,0,240
4127700,Toledo,369,0,0,0,0,0,0,0,0,0,0,0,34,130,5.666666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Toledo, PR, Brazil",3,2,0,369
4128104,Umuarama,140,0,0,0,0,0,0,0,0,0,0,0,775,100,129.1666667,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Umuarama, PR, Brazil",4,1,0,140
4128203,União da Vitória,168,0,0,0,0,0,0,0,0,0,0,0,451,36,75.16666667,10,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"União da Vitória, PR, Brazil",6,1,0,168
4128500,Wenceslau Braz,66,0,0,0,0,0,0,0,0,0,0,0,296,61,49.33333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Wenceslau Braz, PR, Brazil",3,1,0,66
4201406,Araranguá,0,1,1,9,25,70,75,36,34,10,0,0,339,244,56.5,42,144,111,81,34,0,3,0,0,0,0,0,0,373,0.3860589812,0.2975871314,0.2171581769,0.09115281501,0,0.008042895442,0,0,0,0,0,0,"Araranguá, SC, Brazil",3,1,1,261
4202107,Barra Velha,0,0,0,0,0,0,0,0,0,0,0,0,138,107,23,72,59,78,33,29,0,11,14,5,3,0,0,0,232,0.2543103448,0.3362068966,0.1422413793,0.125,0,0.0474137931,0.06034482759,0.02155172414,0.01293103448,0,0,0,"Barra Velha, SC, Brazil",2,1,1,0
4202305,Biguaçu,0,0,0,0,0,2,42,18,21,5,5,2,17,83,2.833333333,83,7,3,13,90,0,0,6,0,0,0,0,0,119,0.05882352941,0.02521008403,0.1092436975,0.756302521,0,0,0.05042016807,0,0,0,0,0,"Biguaçu, SC, Brazil",3,1,1,95
4202404,Blumenau,0,0,0,0,0,0,0,0,0,0,0,0,1032,1625,172,295,829,1070,403,225,0,147,143,36,22,2,4,0,2881,0.28774731,0.3713988199,0.1398819854,0.07809788268,0,0.05102395002,0.04963554321,0.01249566123,0.007636237418,0.0006942034016,0.001388406803,0,"Blumenau, SC, Brazil",1,2,1,0
4202909,Brusque,0,0,0,1,9,23,19,1,5,0,0,0,268,126,44.66666667,60,43,21,8,6,0,2,2,0,3,0,0,0,85,0.5058823529,0.2470588235,0.09411764706,0.07058823529,0,0.02352941176,0.02352941176,0,0.03529411765,0,0,0,"Brusque, SC, Brazil",3,1,1,58
4203006,Caçador,0,0,0,0,0,0,5,0,3,0,0,0,285,190,47.5,104,52,78,46,16,0,4,4,0,0,2,0,2,204,0.2549019608,0.3823529412,0.2254901961,0.07843137255,0,0.01960784314,0.01960784314,0,0,0.009803921569,0,0.009803921569,"Caçador, SC, Brazil",4,1,1,8
4203600,Campos Novos,0,1,0,1,1,20,27,17,9,6,0,0,95,102,15.83333333,102,77,52,35,25,0,6,9,3,0,0,2,1,210,0.3666666667,0.2476190476,0.1666666667,0.119047619,0,0.02857142857,0.04285714286,0.01428571429,0,0,0.009523809524,0.004761904762,"Campos Novos, SC, Brazil",4,1,1,82
4203808,Canoinhas,0,0,0,1,1,19,47,25,26,7,1,0,113,109,18.83333333,109,59,88,41,16,0,10,10,1,0,0,0,0,225,0.2622222222,0.3911111111,0.1822222222,0.07111111111,0,0.04444444444,0.04444444444,0.004444444444,0,0,0,0,"Canoinhas, SC, Brazil",4,1,1,127
4204202,Chapecó,0,0,0,0,0,0,0,0,0,0,0,0,838,2729,139.6666667,389,2046,1449,1138,414,0,494,248,35,36,384,144,2,6390,0.3201877934,0.2267605634,0.1780907668,0.06478873239,0,0.07730829421,0.03881064163,0.005477308294,0.005633802817,0.06009389671,0.02253521127,0.0003129890454,"Chapecó, SC, Brazil",6,4,1,0
4204301,Concórdia,0,0,2,0,6,62,71,17,20,2,1,1,209,169,34.83333333,83,100,118,75,31,0,16,16,1,2,0,0,0,359,0.278551532,0.3286908078,0.208913649,0.08635097493,0,0.04456824513,0.04456824513,0.00278551532,0.005571030641,0,0,0,"Concórdia, SC, Brazil",5,1,1,182
4204608,Criciúma,0,10,3,12,36,254,649,395,429,109,14,1,1076,1750,179.3333333,696,1640,1217,571,296,0,207,268,30,36,2,12,3,4282,0.3829985988,0.2842129846,0.1333489024,0.06912657637,0,0.04834189631,0.0625875759,0.007006071929,0.008407286315,0.0004670714619,0.002802428772,0.0007006071929,"Criciúma, SC, Brazil",4,3,1,1912
4204806,Curitibanos,27,1,2,8,9,84,328,200,231,126,27,3,1255,1926,209.1666667,6,941,593,647,882,0,248,107,23,29,5,4,4,3483,0.2701693942,0.1702555268,0.1857594028,0.2532299742,0,0.07120298593,0.03072064312,0.006603502728,0.008326155613,0.001435544071,0.001148435257,0.001148435257,"Curitibanos, SC, Brazil",4,3,1,1046
4205407,Florianópolis,0,1,25,9,11,29,82,22,16,7,2,0,4621,6838,770.1666667,663,1708,2244,813,207,1604,365,195,49,52,6,5,4,7252,0.2355212355,0.3094318809,0.112107005,0.02854384997,0.221180364,0.05033094319,0.02688913403,0.006756756757,0.007170435742,0.0008273579702,0.0006894649752,0.0005515719801,"Florianópolis, SC, Brazil",1,5,1,204
4207304,Imbituba,0,0,0,0,0,0,0,0,0,0,0,0,228,185,38,109,64,136,34,14,0,12,7,0,1,3,0,0,271,0.2361623616,0.5018450185,0.1254612546,0.05166051661,0,0.0442804428,0.0258302583,0,0.0036900369,0.0110701107,0,0,"Imbituba, SC, Brazil",1,1,1,0
4207502,Indaial,0,13,0,1,3,25,62,17,12,4,0,0,159,138,26.5,42,58,56,38,37,0,2,13,3,2,1,1,1,212,0.2735849057,0.2641509434,0.179245283,0.1745283019,0,0.009433962264,0.06132075472,0.0141509434,0.009433962264,0.004716981132,0.004716981132,0.004716981132,"Indaial, SC, Brazil",1,1,1,137
4208203,Itajaí,0,1,1,15,35,194,263,75,62,27,5,0,2083,2136,347.1666667,976,1840,1429,696,250,0,271,212,70,65,10,13,1,4857,0.3788346716,0.2942145357,0.1432983323,0.05147210212,0,0.0557957587,0.0436483426,0.01441218859,0.01338274655,0.002058884085,0.00267654931,0.0002058884085,"Itajaí, SC, Brazil",1,3,1,678
4208302,Itapema,0,0,0,0,0,0,0,0,0,0,0,0,113,174,18.83333333,56,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Itapema, SC, Brazil",1,1,0,0
4208500,Ituporanga,0,0,0,0,0,0,0,0,0,0,0,0,82,62,13.66666667,62,17,79,10,3,0,1,1,2,0,0,0,0,113,0.1504424779,0.6991150442,0.08849557522,0.02654867257,0,0.008849557522,0.008849557522,0.01769911504,0,0,0,0,"Ituporanga, SC, Brazil",7,1,1,0
4208906,Jaraguá do Sul,0,0,0,0,0,0,0,0,0,0,0,0,264,401,44,283,135,329,102,121,0,21,36,4,2,0,1,1,752,0.1795212766,0.4375,0.1356382979,0.1609042553,0,0.02792553191,0.04787234043,0.005319148936,0.002659574468,0,0.001329787234,0.001329787234,"Jaraguá do Sul, SC, Brazil",2,1,1,0
4209003,Joaçaba,0,0,0,0,0,0,0,0,0,0,0,0,49,211,8.166666667,82,98,100,43,19,0,10,19,1,2,0,1,1,294,0.3333333333,0.3401360544,0.1462585034,0.06462585034,0,0.03401360544,0.06462585034,0.003401360544,0.006802721088,0,0.003401360544,0.003401360544,"Joaçaba, SC, Brazil",5,1,1,0
4209102,Joinville,3,0,0,6,10,110,177,123,225,195,93,16,836,1795,139.3333333,1125,2110,1362,807,294,3,337,285,93,55,109,5,5,5465,0.3860933211,0.2492223239,0.1476669716,0.0537968893,0.00054894785,0.06166514181,0.05215004575,0.01701738335,0.01006404392,0.01994510522,0.0009149130833,0.0009149130833,"Joinville, SC, Brazil",1,3,1,958
4209300,Lages,0,0,0,0,0,0,0,0,0,0,0,0,738,549,123,344,264,303,186,52,0,40,36,5,3,1,2,1,893,0.2956326988,0.3393057111,0.2082866741,0.05823068309,0,0.04479283315,0.04031354983,0.005599104143,0.003359462486,0.001119820829,0.002239641657,0.001119820829,"Lages, SC, Brazil",4,2,1,0
4209409,Laguna,0,0,0,0,0,0,0,0,0,0,0,0,94,102,15.66666667,102,59,83,43,7,0,9,20,0,0,1,0,0,222,0.2657657658,0.3738738739,0.1936936937,0.03153153153,0,0.04054054054,0.09009009009,0,0,0.004504504505,0,0,"Laguna, SC, Brazil",2,1,1,0
4210100,Mafra,0,0,0,0,0,0,0,0,0,0,0,0,0,194,0,194,181,154,554,46,0,16,30,1,4,2,0,0,988,0.1831983806,0.1558704453,0.5607287449,0.04655870445,0,0.01619433198,0.03036437247,0.001012145749,0.004048582996,0.002024291498,0,0,"Mafra, SC, Brazil",1,1,1,0
4210506,Maravilha,0,0,0,1,3,11,32,18,9,1,1,0,103,125,17.16666667,68,27,39,27,31,0,4,5,0,0,0,0,0,133,0.2030075188,0.2932330827,0.2030075188,0.2330827068,0,0.03007518797,0.03759398496,0,0,0,0,0,"Maravilha, SC, Brazil",5,1,1,76
4211900,Palhoça,0,0,0,0,0,0,0,0,0,0,0,0,4516,682,752.6666667,0,439,399,110,58,0,115,49,10,14,0,2,2,1198,0.3664440735,0.3330550918,0.0918196995,0.04841402337,0,0.0959933222,0.0409015025,0.008347245409,0.01168614357,0,0.001669449082,0.001669449082,"Palhoça, SC, Brazil",1,1,1,0
4213609,Porto União,0,0,2,3,5,23,62,22,21,7,0,0,106,166,17.66666667,114,134,78,76,26,0,11,20,0,1,1,0,1,348,0.3850574713,0.224137931,0.2183908046,0.07471264368,0,0.0316091954,0.05747126437,0,0.002873563218,0.002873563218,0,0.002873563218,"Porto União, SC, Brazil",3,1,1,145
4214805,Rio do Sul,0,0,0,0,0,0,0,0,0,0,0,0,724,225,120.6666667,225,142,232,103,80,0,21,35,2,2,1,0,1,619,0.2294022617,0.3747980614,0.1663974152,0.1292407108,0,0.03392568659,0.05654281099,0.003231017771,0.003231017771,0.001615508885,0,0.001615508885,"Rio do Sul, SC, Brazil",5,1,1,0
4216206,São Francisco do Sul,0,0,0,0,0,0,0,0,0,0,0,0,1405,177,234.1666667,100,97,58,77,34,0,0,0,0,0,0,0,0,266,0.3646616541,0.2180451128,0.2894736842,0.1278195489,0,0,0,0,0,0,0,0,"São Francisco do Sul, SC, Brazil",1,1,1,0
4216602,São José,0,0,0,0,3,58,225,217,316,328,129,21,203,1312,33.83333333,0,1269,861,488,92,0,353,201,47,50,0,6,2,3369,0.3766696349,0.2555654497,0.1448501039,0.02730780647,0,0.1047788661,0.05966162066,0.01395072722,0.01484119917,0,0.0017809439,0.0005936479668,"São José, SC, Brazil",2,1,1,1297
4216701,São José do Cedro,0,0,1,5,5,26,54,23,17,5,1,0,95,127,15.83333333,85,67,46,69,36,0,15,13,2,7,0,4,0,259,0.2586872587,0.1776061776,0.2664092664,0.138996139,0,0.05791505792,0.05019305019,0.007722007722,0.02702702703,0,0.01544401544,0,"São José do Cedro, SC, Brazil",3,1,1,137
4217204,São Miguel do Oeste,0,1,5,7,3,26,60,31,20,8,0,0,257,176,42.83333333,85,74,94,66,57,0,15,23,4,1,0,0,0,334,0.2215568862,0.2814371257,0.1976047904,0.1706586826,0,0.04491017964,0.06886227545,0.0119760479,0.002994011976,0,0,0,"São Miguel do Oeste, SC, Brazil",5,1,1,161
4218004,Tijucas,0,0,0,0,0,0,0,0,0,0,0,0,300,155,50,87,152,168,53,36,0,11,14,5,1,1,0,0,441,0.3446712018,0.380952381,0.1201814059,0.08163265306,0,0.02494331066,0.03174603175,0.01133786848,0.002267573696,0.002267573696,0,0,"Tijucas, SC, Brazil",2,1,1,0
4218707,Tubarão,0,0,0,0,2,65,116,70,63,17,6,1,1032,1166,172,253,538,714,208,108,0,61,80,10,4,2,0,1,1726,0.3117033604,0.4136732329,0.1205098494,0.06257242178,0,0.03534183082,0.04634994206,0.005793742758,0.002317497103,0.001158748552,0,0.0005793742758,"Tubarão, SC, Brazil",2,2,1,340
4219309,Videira,0,0,0,0,0,0,0,0,0,0,0,0,170,102,28.33333333,102,120,92,53,16,0,13,7,3,0,1,0,0,305,0.393442623,0.3016393443,0.1737704918,0.05245901639,0,0.04262295082,0.02295081967,0.009836065574,0,0.003278688525,0,0,"Videira, SC, Brazil",4,1,1,0
4219507,Xanxerê,0,0,0,0,0,0,0,0,0,0,0,0,629,156,104.8333333,74,110,101,140,30,0,23,26,0,1,0,0,0,431,0.2552204176,0.2343387471,0.3248259861,0.06960556845,0,0.05336426914,0.06032482599,0,0.002320185615,0,0,0,"Xanxerê, SC, Brazil",3,1,1,0
4300109,Agudo,0,73,0,0,0,1,5,2,3,0,0,0,64,59,10.66666667,0,13,37,5,16,9,1,2,0,1,0,0,0,84,0.1547619048,0.4404761905,0.05952380952,0.1904761905,0.1071428571,0.0119047619,0.02380952381,0,0.0119047619,0,0,0,"Agudo, RS, Brazil",2,1,1,84
4300406,Alegrete,0,81,0,0,1,7,16,10,14,9,0,0,348,37,58,0,22,27,23,13,46,4,3,0,0,0,0,0,138,0.1594202899,0.1956521739,0.1666666667,0.09420289855,0.3333333333,0.02898550725,0.02173913043,0,0,0,0,0,"Alegrete, RS, Brazil",1,1,1,138
4301008,Arroio do Meio,0,55,0,0,0,2,4,2,1,2,1,0,40,26,6.666666667,0,21,18,4,49,0,3,2,0,0,0,0,0,97,0.2164948454,0.1855670103,0.0412371134,0.5051546392,0,0.03092783505,0.0206185567,0,0,0,0,0,"Arroio do Meio, RS, Brazil",6,1,1,67
4301602,Bagé,0,385,0,5,5,43,124,59,65,27,9,1,665,444,110.8333333,0,398,393,88,94,0,101,5,16,1,3,1,0,1100,0.3618181818,0.3572727273,0.08,0.08545454545,0,0.09181818182,0.004545454545,0.01454545455,0.0009090909091,0.002727272727,0.0009090909091,0,"Bagé, RS, Brazil",4,2,1,723
4302105,Bento Gonçalves,0,414,0,0,0,7,12,5,13,9,1,1,534,420,89,0,186,193,54,71,0,25,27,38,0,0,0,0,594,0.3131313131,0.3249158249,0.09090909091,0.1195286195,0,0.04208754209,0.04545454545,0.06397306397,0,0,0,0,"Bento Gonçalves, RS, Brazil",4,1,1,462
4302808,Caçapava do Sul,0,69,0,1,1,5,7,3,4,2,0,0,73,31,12.16666667,0,29,33,9,14,0,4,2,6,0,0,0,0,97,0.2989690722,0.3402061856,0.09278350515,0.1443298969,0,0.0412371134,0.0206185567,0.0618556701,0,0,0,0,"Caçapava do Sul, RS, Brazil",2,1,1,92
4302907,Cacequi,0,22,0,0,0,1,4,3,1,1,0,0,52,36,8.666666667,0,7,4,1,10,0,1,2,0,0,0,0,0,25,0.28,0.16,0.04,0.4,0,0.04,0.08,0,0,0,0,0,"Cacequi, RS, Brazil",1,1,1,32
4303004,Cachoeira do Sul,0,117,0,0,0,10,15,9,3,8,2,0,243,116,40.5,0,93,87,16,15,0,13,4,2,1,3,0,0,234,0.3974358974,0.3717948718,0.06837606838,0.0641025641,0,0.05555555556,0.01709401709,0.008547008547,0.004273504274,0.01282051282,0,0,"Cachoeira do Sul, RS, Brazil",3,1,1,164
4303509,Camaquã,0,376,0,0,0,5,4,4,9,11,1,0,262,150,43.66666667,0,168,140,50,73,0,43,0,2,2,2,0,0,480,0.35,0.2916666667,0.1041666667,0.1520833333,0,0.08958333333,0,0.004166666667,0.004166666667,0.004166666667,0,0,"Camaquã, RS, Brazil",4,1,1,410
4304200,Candelária,0,36,0,0,1,2,11,4,17,1,1,0,93,91,15.5,29,44,14,7,22,0,7,0,0,0,0,0,0,94,0.4680851064,0.1489361702,0.07446808511,0.2340425532,0,0.07446808511,0,0,0,0,0,0,"Candelária, RS, Brazil",1,1,1,73
4304408,Canela,0,169,0,0,0,4,2,6,6,6,2,0,277,80,46.16666667,0,98,119,14,26,0,12,7,8,2,0,0,0,286,0.3426573427,0.4160839161,0.04895104895,0.09090909091,0,0.04195804196,0.02447552448,0.02797202797,0.006993006993,0,0,0,"Canela, RS, Brazil",1,1,1,195
4304507,Canguçu,0,81,0,0,1,0,1,0,0,1,1,0,68,38,11.33333333,0,21,25,11,36,0,7,2,8,0,0,0,0,110,0.1909090909,0.2272727273,0.1,0.3272727273,0,0.06363636364,0.01818181818,0.07272727273,0,0,0,0,"Canguçu, RS, Brazil",1,1,1,85
4304606,Canoas,0,2650,2,1,2,43,85,66,91,73,33,6,2326,3085,387.6666667,0,1488,933,349,827,0,247,178,166,11,14,3,0,4216,0.3529411765,0.2212998102,0.08277988615,0.1961574953,0,0.05858633776,0.04222011385,0.03937381404,0.002609108159,0.003320683112,0.0007115749526,0,"Canoas, RS, Brazil",2,3,1,3052
4304705,Carazinho,0,227,1,0,1,10,18,9,19,11,4,0,366,132,61,0,208,120,37,34,0,36,23,28,0,2,0,0,488,0.4262295082,0.2459016393,0.07581967213,0.06967213115,0,0.0737704918,0.04713114754,0.05737704918,0,0.004098360656,0,0,"Carazinho, RS, Brazil",5,1,1,300
4305108,Caxias do Sul,0,2187,1,1,5,64,149,98,160,97,45,5,1570,2130,261.6666667,0,1559,1244,259,326,0,200,181,150,17,6,9,8,3959,0.3937863097,0.3142207628,0.06542056075,0.08234402627,0,0.05051780753,0.04571861581,0.03788835565,0.00429401364,0.001515534226,0.002273301339,0.002020712301,"Caxias do Sul, RS, Brazil",1,3,1,2812
4305207,Cerro Largo,0,58,0,2,0,3,16,11,1,2,0,0,88,48,14.66666667,0,35,29,13,34,0,6,12,4,0,0,0,0,133,0.2631578947,0.2180451128,0.0977443609,0.2556390977,0,0.04511278195,0.09022556391,0.03007518797,0,0,0,0,"Cerro Largo, RS, Brazil",5,1,1,93
4305355,Charqueadas,0,4589,0,5,20,188,367,234,380,337,196,32,5597,5243,932.8333333,287,4598,2837,556,669,0,1156,525,448,42,2,28,4,10865,0.4231937414,0.2611136677,0.05117349287,0.06157386102,0,0.1063966866,0.04832029452,0.04123331799,0.003865623562,0.0001840773125,0.002577082375,0.0003681546249,"Charqueadas, RS, Brazil",1,9,1,6348
4306106,Cruz Alta,0,142,0,0,1,12,15,16,5,14,4,0,308,148,51.33333333,0,90,114,30,49,0,18,8,2,0,1,0,0,312,0.2884615385,0.3653846154,0.09615384615,0.1570512821,0,0.05769230769,0.02564102564,0.00641025641,0,0.003205128205,0,0,"Cruz Alta, RS, Brazil",5,1,1,209
4306601,Dom Pedrito,0,142,0,0,3,4,11,6,16,6,2,0,181,167,30.16666667,0,55,57,16,22,6,13,16,4,0,3,0,0,192,0.2864583333,0.296875,0.08333333333,0.1145833333,0.03125,0.06770833333,0.08333333333,0.02083333333,0,0.015625,0,0,"Dom Pedrito, RS, Brazil",1,1,1,190
4306809,Encantado,0,65,0,0,0,4,4,5,8,3,0,0,87,81,14.5,0,28,30,8,37,0,3,1,0,5,1,1,0,114,0.2456140351,0.2631578947,0.0701754386,0.3245614035,0,0.02631578947,0.008771929825,0,0.04385964912,0.008771929825,0.008771929825,0,"Encantado, RS, Brazil",7,1,1,89
4306908,Encruzilhada do Sul,0,47,0,0,0,1,5,6,10,3,1,0,54,38,9,0,36,13,8,25,0,6,1,0,1,1,0,0,91,0.3956043956,0.1428571429,0.08791208791,0.2747252747,0,0.06593406593,0.01098901099,0,0.01098901099,0.01098901099,0,0,"Encruzilhada do Sul, RS, Brazil",3,1,1,73
4307005,Erechim,0,235,1,0,3,31,61,59,70,42,5,0,450,239,75,0,287,207,85,90,0,57,33,16,0,2,2,0,779,0.3684210526,0.2657252888,0.109114249,0.1155327343,0,0.07317073171,0.04236200257,0.02053915276,0,0.002567394095,0.002567394095,0,"Erechim, RS, Brazil",14,1,1,507
4307500,Espumoso,0,126,0,0,0,3,4,2,9,1,0,0,168,80,28,0,53,49,26,22,0,4,7,10,1,0,2,0,174,0.3045977011,0.2816091954,0.1494252874,0.1264367816,0,0.02298850575,0.04022988506,0.05747126437,0.005747126437,0,0.01149425287,0,"Espumoso, RS, Brazil",3,1,1,145
4308508,Frederico Westphalen,0,124,0,0,0,6,8,17,16,9,1,0,180,146,30,0,70,69,36,33,0,5,16,8,1,0,0,0,238,0.2941176471,0.2899159664,0.1512605042,0.1386554622,0,0.02100840336,0.06722689076,0.03361344538,0.004201680672,0,0,0,"Frederico Westphalen, RS, Brazil",7,1,1,181
4308904,Getúlio Vargas,0,125,0,0,1,4,9,11,11,6,1,0,156,56,26,0,49,46,18,75,0,7,5,2,2,2,0,0,206,0.2378640777,0.2233009709,0.08737864078,0.3640776699,0,0.03398058252,0.02427184466,0.009708737864,0.009708737864,0.009708737864,0,0,"Getúlio Vargas, RS, Brazil",6,1,1,168
4309209,Gravataí,0,31,0,0,0,3,6,6,7,5,3,0,104,60,17.33333333,0,59,31,5,3,0,9,6,2,1,0,1,0,117,0.5042735043,0.264957265,0.04273504274,0.02564102564,0,0.07692307692,0.05128205128,0.01709401709,0.008547008547,0,0.008547008547,0,"Gravataí, RS, Brazil",2,1,1,61
4309308,Guaíba,0,373,0,0,0,5,7,9,5,2,3,0,401,392,66.83333333,0,129,254,38,46,0,16,29,28,0,1,2,1,544,0.2371323529,0.4669117647,0.06985294118,0.08455882353,0,0.02941176471,0.05330882353,0.05147058824,0,0.001838235294,0.003676470588,0.001838235294,"Guaíba, RS, Brazil",1,1,1,404
4309407,Guaporé,0,108,0,0,1,1,11,3,5,5,2,0,146,60,24.33333333,0,47,70,14,25,0,5,16,22,0,0,0,0,199,0.2361809045,0.351758794,0.07035175879,0.1256281407,0,0.02512562814,0.08040201005,0.1105527638,0,0,0,0,"Guaporé, RS, Brazil",7,1,1,136
4310207,Ijuí,0,327,1,2,6,81,146,56,103,88,31,0,556,518,92.66666667,0,570,437,154,107,0,117,54,45,3,0,0,0,1487,0.3833221251,0.2938802959,0.1035642233,0.07195696032,0,0.07868190989,0.03631472764,0.03026227303,0.002017484869,0,0,0,"Ijuí, RS, Brazil",5,2,1,841
4310504,Iraí,0,50,0,0,1,3,14,3,9,3,0,0,50,90,8.333333333,0,29,23,18,17,0,5,12,2,1,1,1,0,109,0.2660550459,0.2110091743,0.1651376147,0.1559633028,0,0.04587155963,0.1100917431,0.01834862385,0.009174311927,0.009174311927,0.009174311927,0,"Iraí, RS, Brazil",1,1,1,83
4310603,Itaqui,0,135,0,0,0,1,3,0,2,1,0,0,104,50,17.33333333,0,36,24,9,11,43,7,5,8,0,0,0,0,143,0.2517482517,0.1678321678,0.06293706294,0.07692307692,0.3006993007,0.04895104895,0.03496503497,0.05594405594,0,0,0,0,"Itaqui, RS, Brazil",2,1,1,142
4311007,Jaguarão,0,90,0,0,0,2,0,0,1,0,0,0,110,38,18.33333333,0,21,41,3,27,0,6,0,0,0,0,0,0,98,0.2142857143,0.4183673469,0.0306122449,0.2755102041,0,0.0612244898,0,0,0,0,0,0,"Jaguarão, RS, Brazil",1,1,1,93
4311106,Jaguari,0,55,0,1,1,2,5,2,5,1,1,0,81,66,13.5,0,42,0,0,0,31,0,0,0,0,0,0,0,73,0.5753424658,0,0,0,0.4246575342,0,0,0,0,0,0,0,"Jaguari, RS, Brazil",2,1,1,73
4311205,Júlio de Castilhos,0,62,0,0,1,1,6,4,5,2,1,0,67,80,11.16666667,0,31,21,6,32,0,2,2,0,0,2,0,0,96,0.3229166667,0.21875,0.0625,0.3333333333,0,0.02083333333,0.02083333333,0,0,0.02083333333,0,0,"Júlio de Castilhos, RS, Brazil",2,1,1,82
4311304,Lagoa Vermelha,0,132,0,0,1,9,17,21,35,12,3,0,204,94,34,0,157,117,35,32,0,17,21,10,1,0,1,0,391,0.4015345269,0.2992327366,0.0895140665,0.08184143223,0,0.04347826087,0.0537084399,0.02557544757,0.002557544757,0,0.002557544757,0,"Lagoa Vermelha, RS, Brazil",5,1,1,230
4311403,Lajeado,0,321,0,0,0,4,14,8,7,5,2,0,562,318,93.66666667,0,131,189,43,58,0,32,14,22,1,2,2,0,494,0.2651821862,0.3825910931,0.08704453441,0.1174089069,0,0.06477732794,0.02834008097,0.04453441296,0.002024291498,0.004048582996,0.004048582996,0,"Lajeado, RS, Brazil",8,2,1,361
4311502,Lavras do Sul,0,33,0,0,0,1,11,4,7,1,1,0,35,65,5.833333333,0,5,4,7,46,0,1,5,0,0,0,0,0,68,0.07352941176,0.05882352941,0.1029411765,0.6764705882,0,0.01470588235,0.07352941176,0,0,0,0,0,"Lavras do Sul, RS, Brazil",1,1,1,58
4312401,Montenegro,0,1405,0,0,5,62,130,79,94,87,25,2,700,1006,116.6666667,0,1434,697,172,179,0,194,111,90,4,6,7,0,2894,0.4955079475,0.2408431237,0.0594333103,0.06185210781,0,0.06703524534,0.03835521769,0.03109882516,0.001382170007,0.00207325501,0.002418797512,0,"Montenegro, RS, Brazil",7,2,1,1889
4313300,Nova Prata,0,111,0,0,2,8,5,1,5,2,2,0,161,58,26.83333333,0,51,82,14,18,0,9,5,6,0,0,0,0,185,0.2756756757,0.4432432432,0.07567567568,0.0972972973,0,0.04864864865,0.02702702703,0.03243243243,0,0,0,0,"Nova Prata, RS, Brazil",7,1,1,136
4313409,Novo Hamburgo,0,941,0,0,5,29,63,45,47,36,7,0,1164,1181,194,0,531,549,59,45,0,103,264,44,9,2,3,0,1609,0.3300186451,0.3412057178,0.03666873835,0.02796768179,0,0.0640149161,0.1640770665,0.02734617775,0.005593536358,0.00124300808,0.001864512119,0,"Novo Hamburgo, RS, Brazil",1,2,1,1173
4313508,Osório,0,1329,0,0,4,30,58,45,55,32,17,0,1231,685,205.1666667,0,778,901,117,179,0,185,95,62,5,0,1,0,2323,0.334911752,0.3878605252,0.05036590616,0.07705553164,0,0.07963839862,0.04089539389,0.02668962548,0.002152389152,0,0.0004304778304,0,"Osório, RS, Brazil",3,1,1,1570
4313706,Palmeira das Missões,0,151,0,0,0,5,12,5,10,4,0,0,140,156,23.33333333,0,58,83,16,14,0,10,7,2,0,0,0,0,190,0.3052631579,0.4368421053,0.08421052632,0.07368421053,0,0.05263157895,0.03684210526,0.01052631579,0,0,0,0,"Palmeira das Missões, RS, Brazil",7,1,1,187
4314100,Passo Fundo,0,865,2,3,5,58,174,89,114,64,14,0,879,1246,146.5,0,837,552,143,157,0,135,119,90,4,5,9,0,2051,0.4080936129,0.2691370063,0.06972208679,0.07654802535,0,0.06582155046,0.05802047782,0.04388103364,0.001950268162,0.002437835202,0.004388103364,0,"Passo Fundo, RS, Brazil",5,3,1,1388
4314407,Pelotas,0,1433,0,0,2,8,22,23,28,26,12,2,1381,1141,230.1666667,0,512,399,84,116,0,86,4,34,2,11,7,0,1255,0.4079681275,0.3179282869,0.06693227092,0.09243027888,0,0.06852589641,0.003187250996,0.02709163347,0.001593625498,0.008764940239,0.005577689243,0,"Pelotas, RS, Brazil",5,2,1,1556
4314902,Porto Alegre,0,4338,14,19,27,150,308,219,296,180,73,6,12795,6171,2132.5,709,3011,2450,395,608,0,681,379,316,32,16,18,1,7907,0.3808018212,0.3098520298,0.04995573542,0.07689389149,0,0.08612621728,0.04793221196,0.03996458834,0.00404704692,0.00202352346,0.002276463893,0.0001264702163,"Porto Alegre, RS, Brazil",1,10,1,5630
4315305,Quaraí,0,36,0,0,1,8,9,2,2,0,0,0,75,32,12.5,0,19,10,7,10,6,3,1,0,0,0,0,0,56,0.3392857143,0.1785714286,0.125,0.1785714286,0.1071428571,0.05357142857,0.01785714286,0,0,0,0,0,"Quaraí, RS, Brazil",1,1,1,58
4315602,Rio Grande,0,812,0,0,0,5,16,20,45,23,2,0,258,448,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Rio Grande, RS, Brazil",1,1,0,923
4315701,Rio Pardo,0,18,0,0,2,5,18,4,3,3,0,0,66,74,11,0,17,29,7,9,0,3,9,2,0,0,0,0,76,0.2236842105,0.3815789474,0.09210526316,0.1184210526,0,0.03947368421,0.1184210526,0.02631578947,0,0,0,0,"Rio Pardo, RS, Brazil",2,1,1,53
4316402,Rosário do Sul,0,126,0,0,0,6,6,4,5,0,0,0,157,88,26.16666667,0,41,72,14,10,0,7,7,0,0,0,0,0,151,0.2715231788,0.4768211921,0.09271523179,0.06622516556,0,0.04635761589,0.04635761589,0,0,0,0,0,"Rosário do Sul, RS, Brazil",1,1,1,147
4316808,Santa Cruz do Sul,0,571,0,6,12,100,175,95,74,38,1,0,1054,1047,175.6666667,0,528,397,82,249,0,94,86,30,4,8,3,0,1481,0.3565158677,0.2680621202,0.0553679946,0.1681296421,0,0.06347062795,0.05806887238,0.02025658339,0.002700877785,0.005401755571,0.002025658339,0,"Santa Cruz do Sul, RS, Brazil",5,2,1,1072
4316907,Santa Maria,0,968,3,6,15,179,293,149,181,89,15,0,1811,1680,301.8333333,0,1268,833,169,275,0,203,120,66,8,0,11,3,2956,0.4289580514,0.2817997294,0.05717185386,0.09303112314,0,0.06867388363,0.04059539919,0.02232746955,0.002706359946,0,0.003721244926,0.00101488498,"Santa Maria, RS, Brazil",4,4,1,1898
4317103,Santana do Livramento,0,653,0,1,4,23,46,16,25,12,0,0,706,752,117.6666667,0,159,179,49,56,0,29,31,16,3,0,5,0,527,0.3017077799,0.339658444,0.09297912713,0.1062618596,0,0.055028463,0.05882352941,0.03036053131,0.00569259962,0,0.009487666034,0,"Santana do Livramento, RS, Brazil",1,2,1,780
4317202,Santa Rosa,0,390,0,1,3,22,17,15,21,14,4,0,494,288,82.33333333,0,192,245,51,74,0,17,7,0,2,8,4,0,600,0.32,0.4083333333,0.085,0.1233333333,0,0.02833333333,0.01166666667,0,0.003333333333,0.01333333333,0.006666666667,0,"Santa Rosa, RS, Brazil",3,1,1,487
4317301,Santa Vitória do Palmar,0,83,0,0,0,3,2,2,0,0,1,0,63,48,10.5,0,34,41,11,6,0,15,7,2,0,1,0,0,117,0.2905982906,0.3504273504,0.09401709402,0.05128205128,0,0.1282051282,0.05982905983,0.01709401709,0,0.008547008547,0,0,"Santa Vitória do Palmar, RS, Brazil",2,1,1,91
4317400,Santiago,0,43,1,2,2,25,62,24,25,8,1,0,191,207,31.83333333,0,96,147,19,47,0,13,19,0,0,0,0,0,341,0.2815249267,0.431085044,0.05571847507,0.137829912,0,0.03812316716,0.05571847507,0,0,0,0,0,"Santiago, RS, Brazil",4,1,1,193
4317509,Santo Ângelo,0,439,1,1,11,62,135,65,76,42,2,0,738,715,123,0,413,234,56,53,0,47,29,32,1,4,0,0,869,0.4752589183,0.2692750288,0.06444188723,0.06098964327,0,0.05408515535,0.0333716916,0.03682393556,0.001150747986,0.004602991945,0,0,"Santo Ângelo, RS, Brazil",5,3,1,834
4317905,Santo Cristo,0,36,0,0,0,3,18,7,14,3,0,0,74,60,12.33333333,0,25,28,11,47,0,5,1,0,0,0,1,0,118,0.2118644068,0.2372881356,0.09322033898,0.3983050847,0,0.04237288136,0.008474576271,0,0,0,0.008474576271,0,"Santo Cristo, RS, Brazil",4,1,1,81
4318002,São Borja,0,122,1,0,7,34,69,30,39,18,2,0,242,234,40.33333333,0,213,117,36,84,0,38,32,4,0,0,0,0,524,0.4064885496,0.2232824427,0.06870229008,0.1603053435,0,0.07251908397,0.06106870229,0.007633587786,0,0,0,0,"São Borja, RS, Brazil",1,1,1,322
4318101,São Francisco de Assis,0,46,0,0,2,3,26,5,7,1,1,0,65,78,10.83333333,0,15,33,7,34,0,1,10,0,0,2,0,0,102,0.1470588235,0.3235294118,0.06862745098,0.3333333333,0,0.009803921569,0.09803921569,0,0,0.01960784314,0,0,"São Francisco de Assis, RS, Brazil",2,1,1,91
4318200,São Francisco de Paula,0,72,0,2,0,3,3,0,1,2,0,0,79,70,13.16666667,0,33,25,13,25,0,7,8,12,3,1,0,0,127,0.2598425197,0.1968503937,0.1023622047,0.1968503937,0,0.05511811024,0.06299212598,0.09448818898,0.02362204724,0.007874015748,0,0,"São Francisco de Paula, RS, Brazil",2,1,1,83
4318309,São Gabriel,0,292,0,0,0,8,8,10,5,7,1,0,200,142,33.33333333,0,128,168,22,39,0,12,24,4,0,7,0,0,404,0.3168316832,0.4158415842,0.05445544554,0.09653465347,0,0.0297029703,0.05940594059,0.009900990099,0,0.01732673267,0,0,"São Gabriel, RS, Brazil",2,1,1,331
4318408,São Jerônimo,0,620,0,1,2,17,46,20,29,22,10,1,408,672,68,0,482,275,70,96,0,122,0,58,3,0,0,0,1106,0.4358047016,0.2486437613,0.06329113924,0.08679927667,0,0.1103074141,0,0.05244122966,0.002712477396,0,0,0,"São Jerônimo, RS, Brazil",3,1,1,768
4318705,São Leopoldo,0,105,0,1,0,3,16,12,15,12,6,0,169,166,28.16666667,0,156,58,11,8,0,20,13,6,1,0,1,0,274,0.5693430657,0.2116788321,0.0401459854,0.02919708029,0,0.07299270073,0.04744525547,0.02189781022,0.003649635036,0,0.003649635036,0,"São Leopoldo, RS, Brazil",1,1,1,170
4318903,São Luiz Gonzaga,0,210,0,2,1,26,55,20,29,10,4,0,332,138,55.33333333,0,135,192,36,66,0,26,16,18,1,0,0,1,491,0.2749490835,0.3910386965,0.0733197556,0.1344195519,0,0.05295315682,0.03258655804,0.0366598778,0.002036659878,0,0,0.002036659878,"São Luiz Gonzaga, RS, Brazil",8,1,1,357
4319604,São Sepé,0,56,0,0,0,8,6,3,8,1,1,0,88,64,14.66666667,0,48,25,7,20,0,4,4,0,0,2,0,0,110,0.4363636364,0.2272727273,0.06363636364,0.1818181818,0,0.03636363636,0.03636363636,0,0,0.01818181818,0,0,"São Sepé, RS, Brazil",3,1,1,83
4319802,São Vicente do Sul,0,43,0,0,0,2,6,2,3,2,1,0,76,45,12.66666667,0,15,8,6,32,0,1,1,0,0,4,2,0,69,0.2173913043,0.115942029,0.08695652174,0.4637681159,0,0.01449275362,0.01449275362,0,0,0.05797101449,0.02898550725,0,"São Vicente do Sul, RS, Brazil",1,1,1,59
4320008,Sapucaia do Sul,0,480,1,0,3,16,28,15,27,11,2,0,426,600,71,0,389,281,44,38,0,79,43,48,1,0,0,0,923,0.4214517876,0.3044420368,0.04767063922,0.04117009751,0,0.08559046587,0.0465872156,0.05200433369,0.001083423619,0,0,0,"Sapucaia do Sul, RS, Brazil",1,1,1,583
4320107,Sarandi,0,160,0,0,2,11,34,19,24,15,2,0,204,178,34,0,146,119,40,43,0,22,32,8,3,5,0,0,418,0.3492822967,0.2846889952,0.0956937799,0.1028708134,0,0.05263157895,0.07655502392,0.01913875598,0.007177033493,0.01196172249,0,0,"Sarandi, RS, Brazil",3,1,1,267
4320701,Sobradinho,0,63,0,0,2,13,19,8,15,9,1,0,151,130,25.16666667,0,57,24,19,62,0,6,2,2,0,0,0,0,172,0.3313953488,0.1395348837,0.1104651163,0.3604651163,0,0.03488372093,0.01162790698,0.01162790698,0,0,0,0,"Sobradinho, RS, Brazil",6,1,1,130
4320800,Soledade,0,243,0,0,2,4,13,8,11,7,2,1,334,182,55.66666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Soledade, RS, Brazil",6,1,0,291
4321204,Taquara,0,101,0,0,0,4,12,9,9,4,0,0,189,138,31.5,0,108,47,11,5,0,12,10,2,0,0,2,0,197,0.5482233503,0.2385786802,0.05583756345,0.02538071066,0,0.06091370558,0.05076142132,0.01015228426,0,0,0.01015228426,0,"Taquara, RS, Brazil",3,1,1,139
4321501,Torres,0,77,0,0,0,1,0,1,3,0,0,0,130,100,21.66666667,0,21,85,3,1,0,3,7,2,0,0,0,0,122,0.1721311475,0.6967213115,0.02459016393,0.008196721311,0,0.02459016393,0.05737704918,0.01639344262,0,0,0,0,"Torres, RS, Brazil",7,1,1,82
4321907,Três Passos,0,228,0,0,0,13,55,30,41,14,3,0,396,162,66,0,184,169,55,67,0,11,16,8,1,0,4,0,515,0.3572815534,0.3281553398,0.1067961165,0.1300970874,0,0.0213592233,0.03106796117,0.01553398058,0.001941747573,0,0.007766990291,0,"Três Passos, RS, Brazil",4,1,1,384
4322400,Uruguaiana,0,398,1,0,9,50,77,38,69,35,9,0,588,512,98,0,287,277,77,92,0,45,30,12,2,2,1,1,826,0.3474576271,0.3353510896,0.09322033898,0.1113801453,0,0.05447941889,0.03631961259,0.01452784504,0.002421307506,0.002421307506,0.001210653753,0.001210653753,"Uruguaiana, RS, Brazil",2,2,1,686
4322509,Vacaria,0,331,0,0,1,4,5,4,18,2,3,0,378,96,63,0,92,189,63,46,0,24,21,6,0,0,0,0,441,0.20861678,0.4285714286,0.1428571429,0.10430839,0,0.05442176871,0.04761904762,0.01360544218,0,0,0,0,"Vacaria, RS, Brazil",6,1,1,368
4322608,Venâncio Aires,0,260,2,0,1,26,70,76,114,61,17,3,262,529,43.66666667,0,693,346,60,15,0,108,63,48,4,3,5,1,1346,0.514858841,0.2570579495,0.04457652303,0.01114413076,0,0.08023774146,0.04680534918,0.03566121842,0.002971768202,0.002228826152,0.003714710253,0.0007429420505,"Venâncio Aires, RS, Brazil",3,1,1,630
5000609,Amambaí,0,0,0,0,18,96,29,12,12,1,0,0,242,184,40.33333333,0,84,132,17,3,0,0,0,0,0,0,0,0,236,0.3559322034,0.5593220339,0.07203389831,0.01271186441,0,0,0,0,0,0,0,0,"Amambaí, MS, Brazil",1,2,1,168
5001102,Aquidauana,0,13,11,21,33,90,70,33,29,9,1,0,329,108,54.83333333,0,137,156,76,30,0,20,21,3,1,0,2,2,448,0.3058035714,0.3482142857,0.1696428571,0.06696428571,0,0.04464285714,0.046875,0.006696428571,0.002232142857,0,0.004464285714,0.004464285714,"Aquidauana, MS, Brazil",1,3,1,310
5001904,Bataguassu,0,0,0,2,1,14,48,13,10,5,1,0,96,80,16,0,5,14,18,101,0,0,0,0,0,0,0,0,138,0.03623188406,0.1014492754,0.1304347826,0.731884058,0,0,0,0,0,0,0,0,"Bataguassu, MS, Brazil",1,1,1,94
5002407,Caarapó,0,0,0,2,10,0,5,8,0,0,0,0,164,40,27.33333333,0,9,29,28,18,0,3,0,0,0,0,0,0,87,0.1034482759,0.3333333333,0.3218390805,0.2068965517,0,0.03448275862,0,0,0,0,0,0,"Caarapó, MS, Brazil",2,1,1,25
5002704,Campo Grande,280,3,2,70,142,1056,1163,765,427,202,113,26,7148,7353,1191.333333,0,3422,3840,2289,439,1132,504,145,145,82,10,65,6,12079,0.2833015978,0.3179071115,0.1895024423,0.03634406822,0.09371636725,0.04172530839,0.01200430499,0.01200430499,0.006788641444,0.0008278831029,0.005381240169,0.0004967298617,"Campo Grande, MS, Brazil",1,12,1,4249
5002902,Cassilândia,0,6,4,5,11,44,40,26,15,6,0,0,83,80,13.83333333,0,10,22,20,3,0,1,4,1,0,0,0,0,61,0.1639344262,0.3606557377,0.3278688525,0.04918032787,0,0.01639344262,0.06557377049,0.01639344262,0,0,0,0,"Cassilândia, MS, Brazil",1,2,1,157
5003207,Corumbá,0,4,4,5,22,81,178,85,80,35,3,0,332,472,55.33333333,0,228,310,135,73,0,1,1,1,2,0,0,0,751,0.3035952064,0.4127829561,0.1797603196,0.09720372836,0,0.001331557923,0.001331557923,0.001331557923,0.002663115846,0,0,0,"Corumbá, MS, Brazil",2,2,1,497
5003306,Coxim,0,4,1,2,8,45,71,24,18,4,2,0,331,130,55.16666667,0,64,79,52,55,0,0,0,0,0,0,0,0,250,0.256,0.316,0.208,0.22,0,0,0,0,0,0,0,0,"Coxim, MS, Brazil",2,2,1,179
5003488,Dois Irmãos do Buriti,0,0,0,0,0,0,0,0,0,0,0,0,447,238,74.5,0,175,233,125,149,0,0,0,0,0,0,2,0,684,0.2558479532,0.3406432749,0.182748538,0.2178362573,0,0,0,0,0,0,0.002923976608,0,"Dois Irmãos do Buriti, MS, Brazil",1,1,1,0
5003702,Dourados,852,16,21,69,57,497,770,322,391,216,49,8,1558,1243,259.6666667,160,959,810,666,157,0,6,4,5,17,2,7,0,2633,0.3642233194,0.3076338777,0.2529434106,0.05962780099,0,0.002278769464,0.001519179643,0.001898974554,0.006456513483,0.0007595898215,0.002658564375,0,"Dourados, MS, Brazil",1,3,1,3268
5003801,Fátima do Sul,0,0,0,0,0,22,26,9,6,0,0,0,79,48,13.16666667,0,19,92,24,4,0,1,0,0,0,0,1,0,141,0.134751773,0.6524822695,0.170212766,0.02836879433,0,0.007092198582,0,0,0,0,0.007092198582,0,"Fátima do Sul, MS, Brazil",3,1,1,63
5004304,Iguatemi,0,0,0,0,0,0,0,0,0,0,0,0,64,6,10.66666667,6,0,4,0,0,0,0,0,0,0,0,0,0,4,0,1,0,0,0,0,0,0,0,0,0,0,"Iguatemi, MS, Brazil",2,1,1,0
5004700,Ivinhema,0,0,0,0,0,0,0,0,0,0,0,0,98,46,16.33333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ivinhema, MS, Brazil",2,1,0,0
5005004,Jardim,0,0,0,0,53,45,55,28,16,7,0,0,274,114,45.66666667,0,74,87,42,72,0,0,0,0,0,0,0,46,321,0.230529595,0.2710280374,0.1308411215,0.2242990654,0,0,0,0,0,0,0,0.1433021807,"Jardim, MS, Brazil",2,1,1,204
5005707,Naviraí,21,10,11,28,26,102,187,99,100,58,16,3,323,254,53.83333333,0,394,320,256,170,0,42,53,8,10,0,11,4,1268,0.3107255521,0.2523659306,0.2018927445,0.1340694006,0,0.03312302839,0.04179810726,0.006309148265,0.007886435331,0,0.008675078864,0.003154574132,"Naviraí, MS, Brazil",1,2,1,661
5006200,Nova Andradina,0,9,9,6,12,5,0,0,0,0,0,0,168,50,28,0,29,58,24,6,0,0,0,0,0,0,0,0,117,0.2478632479,0.4957264957,0.2051282051,0.05128205128,0,0,0,0,0,0,0,0,"Nova Andradina, MS, Brazil",1,1,1,41
5006309,Paranaíba,178,8,6,11,25,57,61,35,34,16,1,0,254,116,42.33333333,0,111,164,84,54,0,11,5,0,0,1,0,2,432,0.2569444444,0.3796296296,0.1944444444,0.125,0,0.02546296296,0.01157407407,0,0,0.002314814815,0,0.00462962963,"Paranaíba, MS, Brazil",1,2,1,432
5006606,Ponta Porã,0,3,1,6,34,179,143,49,28,12,1,0,320,374,53.33333333,0,108,518,60,3,0,10,2,3,2,0,0,0,706,0.1529745042,0.7337110482,0.08498583569,0.004249291785,0,0.01416430595,0.00283286119,0.004249291785,0.00283286119,0,0,0,"Ponta Porã, MS, Brazil",4,3,1,456
5007208,Rio Brilhante,0,0,0,0,0,0,0,0,0,0,0,0,252,253,42,0,79,219,81,32,0,2,0,0,0,0,0,0,413,0.191283293,0.5302663438,0.196125908,0.07748184019,0,0.004842615012,0,0,0,0,0,0,"Rio Brilhante, MS, Brazil",1,2,1,0
5007695,São Gabriel do Oeste,25,5,0,8,10,25,17,7,6,0,0,0,59,50,9.833333333,0,14,60,18,12,0,0,1,0,0,0,1,0,106,0.1320754717,0.5660377358,0.1698113208,0.1132075472,0,0,0.009433962264,0,0,0,0.009433962264,0,"São Gabriel do Oeste, MS, Brazil",1,3,1,103
5008305,Três Lagoas,0,2,6,14,21,73,65,24,20,6,1,0,825,562,137.5,19,274,485,181,5,0,7,13,1,1,1,3,1,972,0.2818930041,0.4989711934,0.1862139918,0.005144032922,0,0.007201646091,0.0133744856,0.001028806584,0.001028806584,0.001028806584,0.003086419753,0.001028806584,"Três Lagoas, MS, Brazil",2,4,1,232
5100201,Água Boa,0,0,0,0,0,0,0,0,0,0,0,0,354,389,59,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Água Boa, MT, Brazil",3,1,0,0
5100250,Alta Floresta,0,0,0,0,0,0,0,0,0,0,0,0,0,165,0,165,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Alta Floresta, MT, Brazil",2,1,0,0
5100300,Alto Araguaia,0,0,0,0,0,0,0,0,0,0,0,0,17,80,2.833333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Alto Araguaia, MT, Brazil",4,1,0,0
5101258,Araputanga,0,0,0,0,0,3,15,8,10,3,0,0,88,152,14.66666667,152,22,38,12,36,0,0,0,0,0,0,0,0,108,0.2037037037,0.3518518519,0.1111111111,0.3333333333,0,0,0,0,0,0,0,0,"Araputanga, MT, Brazil",3,1,1,39
5101308,Arenápolis,0,0,0,0,0,0,6,4,2,1,0,0,44,87,7.333333333,87,8,30,14,15,0,0,0,0,0,0,0,0,67,0.1194029851,0.447761194,0.2089552239,0.223880597,0,0,0,0,0,0,0,0,"Arenápolis, MT, Brazil",3,1,1,13
5101704,Barra do Bugres,0,0,0,0,2,7,19,5,11,2,1,0,158,95,26.33333333,42,20,53,39,25,0,3,1,0,1,0,0,0,142,0.1408450704,0.3732394366,0.2746478873,0.176056338,0,0.02112676056,0.007042253521,0,0.007042253521,0,0,0,"Barra do Bugres, MT, Brazil",4,1,1,47
5101803,Barra do Garças,0,0,0,0,0,0,0,0,0,0,0,0,281,116,46.83333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Barra do Garças, MT, Brazil",5,1,0,0
5102504,Cáceres,29,0,0,0,12,25,58,14,41,18,2,0,553,465,92.16666667,308,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cáceres, MT, Brazil",2,2,0,199
5102637,Campo Novo do Parecis,0,0,0,0,2,6,18,8,8,9,1,3,88,207,14.66666667,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Campo Novo do Parecis, MT, Brazil",1,1,0,55
5103007,Chapada dos Guimarães,0,0,0,0,0,3,3,5,3,1,1,1,16,100,2.666666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Chapada dos Guimarães, MT, Brazil",3,1,0,17
5103205,Colíder,0,0,0,0,1,6,5,3,2,2,0,0,65,50,10.83333333,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Colíder, MT, Brazil",2,1,0,19
5103254,Colniza,0,0,0,0,0,0,0,0,0,0,0,0,101,114,16.83333333,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Colniza, MT, Brazil",1,1,0,0
5103304,Comodoro,0,0,0,0,0,0,0,0,0,0,0,0,75,112,12.5,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Comodoro, MT, Brazil",3,1,0,0
5103403,Cuiabá,0,0,0,0,45,211,351,210,282,217,83,14,1736,11928,289.3333333,1673,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cuiabá, MT, Brazil",2,3,0,1413
5103502,Diamantino,0,0,0,0,5,6,17,3,4,0,0,0,84,70,14,38,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Diamantino, MT, Brazil",2,1,0,35
5104807,Jaciara,0,0,0,1,9,10,25,9,13,14,1,0,84,115,14,115,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Jaciara, MT, Brazil",2,1,0,82
5105101,Juara,0,0,0,0,1,4,13,6,8,4,1,0,52,201,8.666666667,201,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Juara, MT, Brazil",1,1,0,37
5105150,Juína,0,0,0,0,6,9,20,18,23,18,2,0,148,184,24.66666667,184,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Juína, MT, Brazil",3,1,0,96
5105259,Lucas do Rio Verde,0,0,0,0,4,7,36,19,11,9,1,0,101,464,16.83333333,144,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Lucas do Rio Verde, MT, Brazil",1,1,0,87
5105622,Mirassol d'Oeste,0,0,0,0,0,0,0,0,0,0,0,0,103,208,17.16666667,208,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Mirassol d'Oeste, MT, Brazil",2,1,0,0
5105903,Nobres,0,0,0,0,0,0,0,0,0,0,0,0,170,126,28.33333333,126,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Nobres, MT, Brazil",1,1,0,0
5106000,Nortelândia,0,0,0,0,0,8,12,6,6,4,0,0,0,60,0,60,9,150,12,6,0,0,0,0,0,0,0,0,177,0.05084745763,0.8474576271,0.06779661017,0.03389830508,0,0,0,0,0,0,0,0,"Nortelândia, MT, Brazil",1,1,1,36
5106257,Nova Xavantina,0,0,0,0,1,3,3,2,0,0,0,0,38,54,6.333333333,54,4,17,12,6,0,0,1,0,0,1,0,0,41,0.09756097561,0.4146341463,0.2926829268,0.1463414634,0,0,0.0243902439,0,0,0.0243902439,0,0,"Nova Xavantina, MT, Brazil",1,1,1,9
5106307,Paranatinga,0,0,0,0,7,12,8,6,0,0,0,0,1,80,0.1666666667,80,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Paranatinga, MT, Brazil",2,1,0,33
5106422,Peixoto de Azevedo,0,0,0,0,2,19,48,16,22,7,1,0,0,280,0,256,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Peixoto de Azevedo, MT, Brazil",1,1,0,115
5106752,Pontes e Lacerda,0,0,0,0,2,8,23,19,22,15,4,0,230,484,38.33333333,196,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Pontes e Lacerda, MT, Brazil",3,1,0,93
5106778,Porto Alegre do Norte,0,0,0,0,3,6,13,6,3,1,0,0,1815,77,302.5,45,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Porto Alegre do Norte, MT, Brazil",3,1,0,32
5106802,Porto dos Gaúchos,0,0,0,0,0,0,12,7,9,5,1,0,25,86,4.166666667,86,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Porto dos Gaúchos, MT, Brazil",2,1,0,34
5107040,Primavera do Leste,0,0,0,0,0,0,0,0,0,0,0,0,310,261,51.66666667,146,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Primavera do Leste, MT, Brazil",2,1,0,0
5107602,Rondonópolis,933,0,0,0,13,93,205,120,131,113,45,0,931,1691,155.1666667,329,16,116,11,2,0,2,0,3,0,0,0,0,150,0.1066666667,0.7733333333,0.07333333333,0.01333333333,0,0.01333333333,0,0.02,0,0,0,0,"Rondonópolis, MT, Brazil",2,2,1,1653
5107800,Santo Antônio do Leverger,0,0,0,0,0,0,5,1,3,1,0,0,9,80,1.5,0,4,3,6,14,0,0,0,0,0,0,0,0,27,0.1481481481,0.1111111111,0.2222222222,0.5185185185,0,0,0,0,0,0,0,0,"Santo Antônio do Leverger, MT, Brazil",2,1,1,10
5107859,São Félix do Araguaia,37,0,0,0,6,4,8,2,4,1,0,0,0,64,0,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São Félix do Araguaia, MT, Brazil",4,1,0,62
5107909,Sinop,0,0,0,0,0,0,0,0,0,0,0,0,482,866,80.33333333,314,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Sinop, MT, Brazil",2,1,0,0
5107925,Sorriso,0,0,0,0,0,6,26,16,16,4,3,0,313,340,52.16666667,266,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Sorriso, MT, Brazil",1,1,0,71
5107958,Tangará da Serra,0,0,0,0,66,81,125,58,52,19,10,0,282,433,47,433,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Tangará da Serra, MT, Brazil",1,1,0,411
5108402,Várzea Grande,0,0,0,0,19,62,182,91,131,71,14,0,1340,1334,223.3333333,129,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Várzea Grande, MT, Brazil",2,2,0,570
5108600,Vila Rica,62,0,0,4,3,2,12,5,1,2,2,0,12,120,2,120,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Vila Rica, MT, Brazil",4,1,0,93
5200134,Acreúna,0,0,0,0,0,0,0,0,0,0,0,0,41,66,6.833333333,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Acreúna, GO, Brazil",2,1,0,0
5200258,Águas Lindas de Goiás,0,0,0,0,0,0,0,0,0,0,0,0,733,533,122.1666667,233,47,24,116,20,0,0,0,0,0,0,0,0,207,0.2270531401,0.115942029,0.5603864734,0.09661835749,0,0,0,0,0,0,0,0,"Águas Lindas de Goiás, GO, Brazil",1,2,1,0
5200308,Alexânia,0,0,0,0,1,9,14,5,13,5,0,0,9,74,1.5,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Alexânia, GO, Brazil",1,1,0,47
5200605,Alto Paraíso de Goiás,0,0,0,0,0,0,0,0,0,0,0,0,12,48,2,0,22,0,28,23,10,0,0,0,0,0,0,0,83,0.265060241,0,0.3373493976,0.2771084337,0.1204819277,0,0,0,0,0,0,0,"Alto Paraíso de Goiás, GO, Brazil",2,1,1,0
5200803,Alvorada do Norte,0,0,0,0,0,1,1,1,10,5,1,0,71,38,11.83333333,0,15,6,27,10,0,1,0,0,0,0,1,0,60,0.25,0.1,0.45,0.1666666667,0,0.01666666667,0,0,0,0,0.01666666667,0,"Alvorada do Norte, GO, Brazil",6,1,1,19
5201108,Anápolis,0,0,0,0,0,0,0,0,0,0,0,0,1032,585,172,85,263,125,321,76,0,0,4,1,6,0,0,0,796,0.3304020101,0.1570351759,0.4032663317,0.09547738693,0,0,0.005025125628,0.001256281407,0.007537688442,0,0,0,"Anápolis, GO, Brazil",3,2,1,0
5201306,Anicuns,0,0,0,0,0,0,0,0,0,0,0,0,46,90,7.666666667,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Anicuns, GO, Brazil",3,1,0,0
5201405,Aparecida de Goiânia,4,0,0,0,3,19,34,13,17,9,1,0,6980,2153,1163.333333,1140,658,46,556,124,0,0,1,21,17,7,1,0,1431,0.4598183089,0.0321453529,0.3885394829,0.08665269043,0,0,0.0006988120196,0.01467505241,0.01187980433,0.004891684137,0.0006988120196,0,"Aparecida de Goiânia, GO, Brazil",1,6,1,100
5201702,Aragarças,0,0,0,0,0,0,0,0,0,0,0,0,98,43,16.33333333,43,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Aragarças, GO, Brazil",3,1,0,0
5203203,Barro Alto,0,0,0,0,0,0,0,0,0,0,0,0,99,102,16.5,35,20,23,22,2,0,0,0,0,0,0,0,0,67,0.2985074627,0.3432835821,0.328358209,0.02985074627,0,0,0,0,0,0,0,0,"Barro Alto, GO, Brazil",1,1,1,0
5203302,Bela Vista de Goiás,0,0,0,0,0,0,0,0,0,0,0,0,181,123,30.16666667,0,52,16,48,23,2,4,0,0,0,0,0,0,145,0.3586206897,0.1103448276,0.3310344828,0.1586206897,0.01379310345,0.0275862069,0,0,0,0,0,0,"Bela Vista de Goiás, GO, Brazil",1,1,1,0
5204102,Cachoeira Alta,0,0,0,0,0,0,0,0,0,0,0,0,190,100,31.66666667,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cachoeira Alta, GO, Brazil",1,1,0,0
5204409,Caiapônia,0,0,0,1,3,64,6,4,3,3,2,0,119,51,19.83333333,24,32,24,24,13,0,0,6,0,0,0,0,0,99,0.3232323232,0.2424242424,0.2424242424,0.1313131313,0,0,0.06060606061,0,0,0,0,0,"Caiapônia, GO, Brazil",3,1,1,86
5204508,Caldas Novas,0,0,0,0,0,0,0,0,0,0,0,0,465,184,77.5,0,105,79,100,44,0,3,1,0,0,0,0,2,334,0.3143712575,0.2365269461,0.2994011976,0.1317365269,0,0.008982035928,0.002994011976,0,0,0,0,0.005988023952,"Caldas Novas, GO, Brazil",3,1,1,0
5204904,Campos Belos,0,0,0,0,0,0,0,0,0,0,0,0,38,53,6.333333333,53,10,11,16,7,0,3,0,0,0,0,0,2,49,0.2040816327,0.2244897959,0.3265306122,0.1428571429,0,0.0612244898,0,0,0,0,0,0.04081632653,"Campos Belos, GO, Brazil",2,1,1,0
5205109,Catalão,0,0,0,0,0,0,0,0,0,0,0,0,389,179,64.83333333,179,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Catalão, GO, Brazil",4,1,0,0
5205406,Ceres,0,0,0,0,0,0,0,0,0,0,0,0,194,124,32.33333333,70,43,0,35,14,0,0,0,0,0,0,0,0,92,0.4673913043,0,0.3804347826,0.152173913,0,0,0,0,0,0,0,0,"Ceres, GO, Brazil",3,1,1,0
5205497,Cidade Ocidental,0,0,0,0,0,0,0,0,0,0,0,0,717,125,119.5,73,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cidade Ocidental, GO, Brazil",1,1,0,0
5206206,Cristalina,0,0,0,0,0,0,0,0,0,0,0,0,152,35,25.33333333,35,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Cristalina, GO, Brazil",1,1,0,0
5207402,Edéia,0,0,0,0,0,0,0,0,0,0,0,0,81,42,13.5,0,18,29,29,1,0,1,1,0,0,0,0,0,79,0.2278481013,0.3670886076,0.3670886076,0.01265822785,0,0.01265822785,0.01265822785,0,0,0,0,0,"Edéia, GO, Brazil",2,1,1,0
5208004,Formosa,0,0,0,0,0,9,11,1,2,0,0,0,316,407,52.66666667,86,81,118,148,25,111,15,1,0,0,0,0,0,499,0.1623246493,0.2364729459,0.2965931864,0.0501002004,0.2224448898,0.03006012024,0.002004008016,0,0,0,0,0,"Formosa, GO, Brazil",2,3,1,23
5208400,Goianápolis,0,0,0,0,0,3,5,8,6,5,0,0,53,50,8.833333333,0,23,1,4,4,0,0,0,0,0,0,0,0,32,0.71875,0.03125,0.125,0.125,0,0,0,0,0,0,0,0,"Goianápolis, GO, Brazil",2,1,1,27
5208608,Goianésia,0,4,0,0,8,26,41,25,12,5,0,0,275,220,45.83333333,0,109,61,91,36,0,9,2,0,1,0,0,0,309,0.3527508091,0.1974110032,0.2944983819,0.1165048544,0,0.02912621359,0.006472491909,0,0.003236245955,0,0,0,"Goianésia, GO, Brazil",3,1,1,121
5208707,Goiânia,0,0,0,0,0,0,0,0,0,0,0,0,694,8500,115.6666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Goiânia, GO, Brazil",1,2,0,0
5208905,Goiás,81,0,0,0,0,13,0,7,10,0,0,0,81,61,13.5,0,35,10,23,21,0,1,1,0,0,0,0,0,91,0.3846153846,0.1098901099,0.2527472527,0.2307692308,0,0.01098901099,0.01098901099,0,0,0,0,0,"Goiás, GO, Brazil",2,1,1,111
5210000,Inhumas,0,0,0,0,0,0,0,0,0,0,0,0,97,89,16.16666667,44,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Inhumas, GO, Brazil",3,1,0,0
5210109,Ipameri,0,0,0,0,0,0,0,0,0,0,0,0,101,43,16.83333333,22,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Ipameri, GO, Brazil",1,1,0,0
5210208,Iporá,0,0,0,0,0,0,0,0,0,0,0,0,129,98,21.5,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Iporá, GO, Brazil",3,1,0,0
5210307,Israelândia,0,0,0,0,0,0,0,0,0,0,0,0,80,51,13.33333333,12,8,14,12,0,0,0,0,0,0,0,0,0,34,0.2352941176,0.4117647059,0.3529411765,0,0,0,0,0,0,0,0,0,"Israelândia, GO, Brazil",2,1,1,0
5210406,Itaberaí,0,0,0,0,0,0,0,0,0,0,0,0,102,90,17,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Itaberaí, GO, Brazil",2,1,0,0
5211206,Itapuranga,0,0,0,2,2,1,20,10,7,5,0,0,102,60,17,60,21,38,31,19,0,3,0,0,0,0,0,0,112,0.1875,0.3392857143,0.2767857143,0.1696428571,0,0.02678571429,0,0,0,0,0,0,"Itapuranga, GO, Brazil",2,1,1,47
5211404,Itauçu,28,0,0,0,1,20,10,15,4,1,0,0,155,202,25.83333333,32,25,16,37,24,0,8,11,3,0,0,1,0,125,0.2,0.128,0.296,0.192,0,0.064,0.088,0.024,0,0,0.008,0,"Itauçu, GO, Brazil",1,1,1,79
5211503,Itumbiara,0,0,0,0,0,0,0,0,0,0,0,0,540,302,90,302,129,153,75,53,0,4,0,3,2,1,0,0,420,0.3071428571,0.3642857143,0.1785714286,0.1261904762,0,0.009523809524,0,0.007142857143,0.004761904762,0.002380952381,0,0,"Itumbiara, GO, Brazil",1,1,1,0
5211701,Jandaia,0,0,3,18,43,33,50,20,9,3,0,0,84,44,14,20,79,68,67,7,0,8,13,4,0,0,0,0,246,0.3211382114,0.2764227642,0.2723577236,0.02845528455,0,0.0325203252,0.05284552846,0.0162601626,0,0,0,0,"Jandaia, GO, Brazil",2,1,1,179
5211800,Jaraguá,0,0,0,0,0,0,0,0,0,0,0,0,121,68,20.16666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Jaraguá, GO, Brazil",3,1,0,0
5211909,Jataí,0,0,0,0,35,42,52,6,0,0,0,0,21,347,3.5,226,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Jataí, GO, Brazil",3,1,0,135
5212204,Jussara,0,0,0,0,0,0,0,0,0,0,0,0,77,49,12.83333333,49,28,26,20,6,0,4,0,0,0,0,0,0,84,0.3333333333,0.3095238095,0.2380952381,0.07142857143,0,0.04761904762,0,0,0,0,0,0,"Jussara, GO, Brazil",3,1,1,0
5212501,Luziânia,0,0,0,0,0,0,0,0,0,0,0,0,674,468,112.3333333,156,30,22,29,4,0,0,0,0,0,0,0,0,85,0.3529411765,0.2588235294,0.3411764706,0.04705882353,0,0,0,0,0,0,0,0,"Luziânia, GO, Brazil",1,4,1,0
5213087,Minaçu,0,0,0,0,0,0,0,0,0,0,0,0,124,173,20.66666667,60,38,12,34,12,0,0,0,0,0,0,0,0,96,0.3958333333,0.125,0.3541666667,0.125,0,0,0,0,0,0,0,0,"Minaçu, GO, Brazil",2,1,1,0
5213103,Mineiros,0,0,0,0,0,0,0,0,0,0,0,0,0,122,0,122,145,126,131,46,38,8,2,1,0,0,0,0,497,0.291750503,0.2535211268,0.2635814889,0.09255533199,0.07645875252,0.01609657948,0.004024144869,0.002012072435,0,0,0,0,"Mineiros, GO, Brazil",3,1,1,0
5213806,Morrinhos,0,0,0,0,0,0,0,0,0,0,0,0,165,285,27.5,62,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Morrinhos, GO, Brazil",1,1,0,0
5214002,Mozarlândia,0,0,0,0,0,0,0,0,0,0,0,0,11,145,1.833333333,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Mozarlândia, GO, Brazil",2,1,0,0
5214606,Niquelândia,0,0,0,0,0,0,0,0,0,0,0,0,84,137,14,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Niquelândia, GO, Brazil",2,1,0,0
5214838,Nova Crixás,0,0,0,0,0,0,0,0,0,0,0,0,0,134,0,58,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Nova Crixás, GO, Brazil",1,1,0,0
5215231,Novo Gama,0,0,0,0,0,0,0,0,0,0,0,0,296,62,49.33333333,0,45,26,83,15,36,5,0,0,0,0,0,0,210,0.2142857143,0.1238095238,0.3952380952,0.07142857143,0.1714285714,0.02380952381,0,0,0,0,0,0,"Novo Gama, GO, Brazil",1,1,1,0
5215306,Orizona,0,0,0,0,0,0,0,0,0,0,0,0,73,111,12.16666667,70,32,42,33,2,0,0,2,0,0,0,0,0,111,0.2882882883,0.3783783784,0.2972972973,0.01801801802,0,0,0.01801801802,0,0,0,0,0,"Orizona, GO, Brazil",1,1,1,0
5215603,Padre Bernardo,0,0,0,0,0,0,0,0,0,0,0,0,84,98,14,98,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Padre Bernardo, GO, Brazil",2,1,0,0
5215702,Palmeiras de Goiás,0,0,0,0,0,0,0,0,0,0,0,0,117,100,19.5,26,0,0,0,0,50,0,0,0,0,0,0,0,50,0,0,0,0,1,0,0,0,0,0,0,0,"Palmeiras de Goiás, GO, Brazil",2,1,1,0
5217401,Pires do Rio,0,0,0,0,10,20,15,10,4,1,0,0,16,71,2.666666667,30,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Pires do Rio, GO, Brazil",1,1,0,60
5217609,Planaltina,0,0,0,0,0,0,0,0,0,0,0,0,257,624,42.83333333,82,55,57,30,2,4,9,0,7,6,0,2,0,172,0.3197674419,0.3313953488,0.1744186047,0.01162790698,0.02325581395,0.0523255814,0,0.04069767442,0.03488372093,0,0.01162790698,0,"Planaltina, GO, Brazil",2,2,1,0
5217708,Pontalina,0,0,0,0,0,0,0,0,0,0,0,0,54,92,9,51,26,56,25,25,0,0,0,0,0,0,0,0,132,0.196969697,0.4242424242,0.1893939394,0.1893939394,0,0,0,0,0,0,0,0,"Pontalina, GO, Brazil",2,1,1,0
5218003,Porangatu,0,0,0,0,0,0,0,0,0,0,0,0,230,220,38.33333333,100,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Porangatu, GO, Brazil",3,1,0,0
5218300,Posse,0,0,0,0,0,0,0,0,0,0,0,0,155,74,25.83333333,74,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Posse, GO, Brazil",3,1,0,0
5218508,Quirinópolis,0,0,0,0,0,0,0,0,0,0,0,0,176,188,29.33333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Quirinópolis, GO, Brazil",2,1,0,0
5218805,Rio Verde,0,0,0,0,0,0,0,0,0,0,0,0,60,454,10,307,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Rio Verde, GO, Brazil",3,3,0,0
5218904,Rubiataba,0,0,0,0,0,0,0,0,0,0,0,0,113,80,18.83333333,40,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Rubiataba, GO, Brazil",3,1,0,0
5219001,Sanclerlândia,0,0,0,0,0,0,0,0,0,0,0,0,115,115,19.16666667,51,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Sanclerlândia, GO, Brazil",3,1,0,0
5219308,Santa Helena de Goiás,0,0,0,0,0,0,0,0,0,0,0,0,0,168,0,42,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Santa Helena de Goiás, GO, Brazil",1,1,0,0
5219753,Santo Antônio do Descoberto,0,0,0,0,0,0,0,0,0,0,0,0,169,110,28.16666667,110,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Santo Antônio do Descoberto, GO, Brazil",1,1,0,0
5220108,São Luís de Montes Belos,0,0,0,0,0,5,32,21,15,2,0,0,182,97,30.33333333,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São Luís de Montes Belos, GO, Brazil",1,1,0,75
5220207,São Miguel do Araguaia,0,0,0,0,0,0,0,0,0,0,0,0,67,92,11.16666667,34,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"São Miguel do Araguaia, GO, Brazil",2,1,0,0
5220454,Senador Canedo,0,0,0,0,0,0,0,0,0,0,0,0,411,82,68.5,82,55,42,41,40,0,4,0,1,0,0,0,0,183,0.3005464481,0.2295081967,0.2240437158,0.218579235,0,0.0218579235,0,0.005464480874,0,0,0,0,"Senador Canedo, GO, Brazil",2,1,1,0
5220504,Serranópolis,0,0,0,1,0,3,11,12,19,0,0,0,246,94,41,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Serranópolis, GO, Brazil",1,1,0,46
5220603,Silvânia,0,0,0,0,0,0,0,0,0,0,0,0,56,56,9.333333333,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Silvânia, GO, Brazil",2,1,0,0
5221403,Trindade,0,0,0,0,0,0,0,0,0,0,0,0,330,165,55,125,0,0,60,0,0,0,0,0,0,0,0,0,60,0,0,1,0,0,0,0,0,0,0,0,0,"Trindade, GO, Brazil",2,1,1,0
5221601,Uruaçu,0,0,0,0,0,0,0,0,0,0,0,0,250,180,41.66666667,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Uruaçu, GO, Brazil",1,1,0,0
5221700,Uruana,0,0,0,0,0,0,0,0,0,0,0,0,145,125,24.16666667,50,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Uruana, GO, Brazil",1,1,0,0
5221858,Valparaíso de Goiás,0,0,0,0,0,0,0,0,0,0,0,0,272,168,45.33333333,130,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"Valparaíso de Goiás, GO, Brazil",1,1,0,0
5300108,Brasília,4017,0,0,1,600,2809,3644,1847,2119,1513,639,126,17531,14881,2921.833333,1766,12464,5908,5355,1033,7,1176,1120,421,229,54,28,8,27803,0.4482969464,0.2124950545,0.1926051146,0.03715426393,0.0002517713916,0.04229759378,0.04028342265,0.01514225084,0.008236521239,0.001942236449,0.001007085566,0.0002877387332,"Brasília, DF, Brazil",1,10,1,17315
,,0,0,0,0,0,0,0,0,0,0,0,0,1618,0,269.6666667,0,,,,,,,,,,,,,,0,0,0,0,0,0,0,0,0,0,0,0,,,0,0,0
"""

@st.cache_data(show_spinner="Loading embedded comarca dataset…")
def load_embedded_comarca_data() -> pd.DataFrame:
    df = pd.read_csv(StringIO(EMBEDDED_COMARCA_CSV.strip()))
    df.columns = df.columns.str.strip()
    return df

_comarca_df = load_embedded_comarca_data()
_comarca_names = get_comarca_names(_comarca_df)

# Compute national defaults from the dataset
_national_arrests_per_month = (
    round(_comarca_df["arrests_per_month"].sum(), 0)
    if (not _comarca_df.empty and "arrests_per_month" in _comarca_df.columns)
    else 196.0
)
_national_total_cap = (
    int(_comarca_df["total_capacity_threshold"].sum())
    if (not _comarca_df.empty and "total_capacity_threshold" in _comarca_df.columns)
    else 354
)
_national_pretrial_cap = (
    int(_comarca_df["pre_trial_capacity_threshold"].sum())
    if (not _comarca_df.empty and "pre_trial_capacity_threshold" in _comarca_df.columns)
    else 105
)

# ---------------------------------------------------------------------------
# HELPER: SENTENCE PROBABILITY BUILDER
# ---------------------------------------------------------------------------

def build_sentence_probs(counts: dict):
    """
    Converts sentence-bracket count dictionaries into the (support, probs)
    tuple expected by the simulation's ``detention_distribution`` field.

    This mirrors the exact normalisation logic used in Scenario 2 of the
    simulation script, distributing counts uniformly within each bracket
    and then normalising across the full support array.

    Parameters
    ----------
    counts : dict
        Keys match ``DEFAULT_SENTENCE_COUNTS``; values are integer counts.

    Returns
    -------
    sentence_support : np.ndarray
        Integer years from 0 to 101 (101 = sentinel for >100 years).
    sentence_probs : np.ndarray
        Normalised probability mass for each year bucket.
    """
    sentence_support = np.arange(0, 102)
    sentence_probs = np.zeros_like(sentence_support, dtype=float)

    sentence_probs[0]      += counts.get("0_6mo",    0)
    sentence_probs[1]      += counts.get("7_12mo",   0)
    sentence_probs[1:3]    += counts.get("13mo_2yr", 0) / 2
    sentence_probs[3:5]    += counts.get("3_4yr",    0) / 2
    sentence_probs[5:9]    += counts.get("5_8yr",    0) / 4
    sentence_probs[9:16]   += counts.get("9_15yr",   0) / 7
    sentence_probs[16:21]  += counts.get("16_20yr",  0) / 5
    sentence_probs[21:31]  += counts.get("21_30yr",  0) / 10
    sentence_probs[31:51]  += counts.get("31_50yr",  0) / 20
    sentence_probs[51:101] += counts.get("51_100yr", 0) / 50
    sentence_probs[101]    += counts.get("gt100",    0)

    total = sentence_probs.sum()
    if total == 0:
        # Uniform fallback to avoid division by zero
        sentence_probs = np.ones_like(sentence_probs, dtype=float)
        total = sentence_probs.sum()

    sentence_probs /= total
    return sentence_support, sentence_probs


# ---------------------------------------------------------------------------
# HELPER: TRUNCNORM DISTRIBUTION BUILDER
# ---------------------------------------------------------------------------

def make_truncnorm(mean: float, std: float, lower: float, upper: float):
    """
    Constructs a ``scipy.stats.truncnorm`` frozen distribution from plain
    mean / std / lower / upper parameters.

    Parameters
    ----------
    mean : float
        Distribution mean.
    std : float
        Standard deviation (must be > 0).
    lower : float
        Lower clip bound (must be < upper).
    upper : float
        Upper clip bound.

    Returns
    -------
    scipy.stats.truncnorm
        Frozen truncated-normal distribution.

    Raises
    ------
    ValueError
        When ``std <= 0`` or ``lower >= upper``.
    """
    if std <= 0:
        raise ValueError("Standard deviation must be greater than zero.")
    if lower >= upper:
        raise ValueError("Lower bound must be strictly less than upper bound.")
    a = (lower - mean) / std
    b = (upper - mean) / std
    return truncnorm(a=a, b=b, loc=mean, scale=std)


# ---------------------------------------------------------------------------
# HELPER: ARREST-RATE DISTRIBUTION BUILDER
# ---------------------------------------------------------------------------

def build_arrest_rate_dist(arrests_per_month: float):
    """
    Converts a monthly arrest rate into an exponential inter-arrival
    distribution, mirroring the approach used throughout the simulation.

    Parameters
    ----------
    arrests_per_month : float
        Average number of arrests per month.

    Returns
    -------
    scipy.stats.expon
        Frozen exponential distribution.
    """
    scale = 1.0 / max(arrests_per_month, 1e-6)   # guard against divide-by-zero
    return expon(scale=scale)


# ---------------------------------------------------------------------------
# HELPER: CRIME-PROFILE BUILDER
# ---------------------------------------------------------------------------

def build_crime_profiles(crime_params: dict, sentence_support, sentence_probs):
    """
    Assembles a crime-profile dictionary in the exact format expected by the
    simulation's ``Arrests`` class, using the user-supplied parameter values
    and the pre-computed sentence-length distribution.

    Parameters
    ----------
    crime_params : dict
        App-side crime parameter dictionary keyed by crime-group internal names.
        Each entry must include ``arrival_probability``, ``conviction_probability``,
        ``service_time``, ``public_decision``, and ``private_decision`` sub-dicts.
    sentence_support : np.ndarray
        Integer year support array (0–101).
    sentence_probs : np.ndarray
        Normalised probability mass over the support.

    Returns
    -------
    profiles : dict
        Crime profiles dictionary ready to be passed into ``Arrests()``.

    Raises
    ------
    ValueError
        When any distribution parameters are invalid.
    """
    # Normalise arrival probabilities so they sum to 1
    arrival_total = sum(
        p["arrival_probability"] for p in crime_params.values()
    )
    if arrival_total <= 0:
        raise ValueError(
            "Arrival probabilities sum to zero. "
            "Please assign positive values to at least one crime group."
        )

    profiles = {}
    for key, params in crime_params.items():
        norm_arrival = params["arrival_probability"] / arrival_total

        svc     = params["service_time"]
        pub_dec = params["public_decision"]
        prv_dec = params["private_decision"]

        profiles[key] = {
            "arrival_probability":      norm_arrival,
            "conviction_probability":   params["conviction_probability"],
            "detention_distribution":   (sentence_support, sentence_probs),
            "service_time_dist":        make_truncnorm(**svc),
            "public_decision_wait_dist":  make_truncnorm(**pub_dec),
            "private_decision_wait_dist": make_truncnorm(**prv_dec),
        }

    return profiles


# ---------------------------------------------------------------------------
# HELPER: MATPLOTLIB FIGURE RENDERER
# ---------------------------------------------------------------------------

def render_mpl_figure(fig):
    """
    Renders a Matplotlib figure inside a Streamlit container. The function
    captures the current figure, displays it, and ensures ``plt.close`` is
    called to free memory.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to render.
    """
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def capture_and_render_current_figure():
    """
    Captures whatever figure the simulation plotting function just created
    (via ``plt.show()``) and renders it in Streamlit.

    Because the simulation's plotting functions call ``plt.show()`` internally,
    we patch ``plt.show`` before calling them so the figure is not discarded,
    then render and close it afterward.

    Returns
    -------
    fig : matplotlib.figure.Figure or None
        The captured figure, or None if nothing was plotted.
    """
    fig = plt.gcf()
    if fig and fig.get_axes():
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        return fig
    return None


# ---------------------------------------------------------------------------
# HELPER: THEMED PRETRIAL RATIO PLOT  (replaces the un-styled original)
# ---------------------------------------------------------------------------

def plot_pretrial_ratio_themed(results: dict, title: str, ax_ref=None):
    """
    Plots the arrival-rate-vs-pre-trial-detention ratio with the app theme
    applied. This wraps ``plot_arrival_rate_vs_pretrial_ratio`` from the
    simulation module but adds the dark theme so it matches the other charts.

    Parameters
    ----------
    results : dict
        Output from ``analyze_arrival_rate_vs_pretrial_ratio()``.
    title : str
        Chart title.
    ax_ref : matplotlib.axes.Axes or None
        If provided, plot onto this existing Axes; otherwise create a new figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    arrest_rates = list(results.keys())
    mean_ratios  = [results[r]["mean_ratio"] for r in arrest_rates]
    ci95_vals    = [results[r]["ci95"]        for r in arrest_rates]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.errorbar(
        arrest_rates, mean_ratios, yerr=ci95_vals,
        fmt="o-", capsize=4, color=C_PRETRIAL_POP,
        ecolor=AX, label="Pre-Trial Ratio (95% CI)",
    )

    ax.set_title(title, color=FG)
    ax.set_xlabel("Arrests per Month", color=FG)
    ax.set_ylabel("Pre-Trial / Total Incarceration", color=FG)
    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)
    ax.grid(True, color=GRID, alpha=0.3)

    leg = ax.legend(frameon=False)
    if leg:
        for text in leg.get_texts():
            text.set_color(FG)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HELPER: UTILIZATION PLOT (theoretical analysis)
# ---------------------------------------------------------------------------

def plot_utilization(lambda_total: float, s_weighted: float, num_queues_range: tuple):
    """
    Computes and plots the theoretical utilisation (ρ = λ·S / Nq) across
    a range of litigation-queue counts.

    Parameters
    ----------
    lambda_total : float
        Total arrest rate (arrests per month).
    s_weighted : float
        Weighted average trial service time (months).
    num_queues_range : tuple of (int, int)
        (min_queues, max_queues) inclusive.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    min_q, max_q = num_queues_range
    nq_array  = np.arange(max(1, min_q), max_q + 1)
    rho_array = (lambda_total / nq_array) * s_weighted

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    ax.plot(nq_array, rho_array, marker="o", color=C_PRETRIAL_CAP, label="Utilisation (ρ)")
    ax.axhline(1.0, color=C_TOTAL_CAP, linestyle="--", lw=2, label="Stability Threshold (ρ = 1)")

    ax.set_title("Utilisation vs. Number of Litigation Queues", color=FG)
    ax.set_xlabel("Number of Litigation Queues (Nq)", color=FG)
    ax.set_ylabel("Utilisation (ρ)", color=FG)
    ax.tick_params(axis="both", colors=FG)
    for spine in ax.spines.values():
        spine.set_color(AX)
    ax.grid(True, color=GRID, alpha=0.3)

    leg = ax.legend(frameon=False)
    if leg:
        for text in leg.get_texts():
            text.set_color(FG)

    # Annotate stable / unstable regions
    y_min, y_max = ax.get_ylim()
    ax.fill_between(nq_array, 0, np.minimum(rho_array, 1.0),
                    color=C_PRETRIAL_POP, alpha=0.12, label="_stable zone")
    ax.fill_between(nq_array, 1.0, np.maximum(rho_array, 1.0),
                    color=C_TOTAL_CAP, alpha=0.12, label="_unstable zone")

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# HELPER: COMPUTE WEIGHTED MEAN SERVICE TIME FROM CRIME PARAMS
# ---------------------------------------------------------------------------

def compute_weighted_service_time(crime_params: dict) -> float:
    """
    Computes the weighted average trial service time (in months) across all
    crime groups, using arrival probabilities as weights.

    Parameters
    ----------
    crime_params : dict
        App-side crime parameter dictionary.

    Returns
    -------
    float
        Weighted average service time in months.
    """
    total_weight = sum(p["arrival_probability"] for p in crime_params.values())
    if total_weight == 0:
        return 1.0   # safe fallback

    s_weighted = sum(
        (p["arrival_probability"] / total_weight) * p["service_time"]["mean"]
        for p in crime_params.values()
    )
    return s_weighted


# ---------------------------------------------------------------------------
# SECTION TITLE HELPER
# ---------------------------------------------------------------------------

def section_title(text: str):
    """Renders a themed section heading using the Oswald font."""
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def note(text: str):
    """Renders small, muted interpretive text."""
    st.markdown(f'<div class="note-text">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# COMARCA DATASET  (loaded once, shared across the session)
# ---------------------------------------------------------------------------

# Start from the embedded dataset so the app always has comarca options,
# then optionally override it with an external CSV when one is available.
# This prevents the dropdown from going blank when a local file path is missing.
_COMARCA_CANDIDATE_PATHS = [
    "comarca_parameters__3_.csv",
    os.path.join(os.path.dirname(__file__), "comarca_parameters__3_.csv"),
    "/mnt/user-data/uploads/comarca_parameters__3_.csv",
]

_comarca_df = load_embedded_comarca_data().copy()
_comarca_source = "embedded"
for _path in _COMARCA_CANDIDATE_PATHS:
    if os.path.exists(_path):
        _loaded_df = load_comarca_csv(_path)
        if not _loaded_df.empty:
            _comarca_df = _loaded_df
            _comarca_source = _path
            break

_comarca_names = get_comarca_names(_comarca_df)

# Compute national defaults from the active dataset (embedded fallback or external CSV)
_national_arrests_per_month = (
    round(_comarca_df["arrests_per_month"].sum(), 0)
    if (not _comarca_df.empty and "arrests_per_month" in _comarca_df.columns)
    else 196.0   # fallback: avg_arrests_per_prison/6 from Scenario 2
)
_national_total_cap = (
    int(_comarca_df["total_capacity_threshold"].sum())
    if (not _comarca_df.empty and "total_capacity_threshold" in _comarca_df.columns)
    else 354
)
_national_pretrial_cap = (
    int(_comarca_df["pre_trial_capacity_threshold"].sum())
    if (not _comarca_df.empty and "pre_trial_capacity_threshold" in _comarca_df.columns)
    else 105
)


# ---------------------------------------------------------------------------
# SESSION-STATE INITIALISATION
# ---------------------------------------------------------------------------

def _init_session_state():
    """Initialises all Streamlit session-state keys used across the app."""
    defaults = {
        # Simulation results
        "court_system_list":          None,
        "last_sim_params_hash":       None,
        # Experimental results
        "exp_wt_queues_result":       None,
        "exp_inc_queues_result":      None,
        "exp_wt_stations_result":     None,
        "exp_inc_stations_result":    None,
        "exp_ratio_result":           None,
        "exp_ratio_zoomed_result":    None,
        # Visualisation toggles stored in state
        "viz_incarceration_multi":    True,
        "viz_incarceration_crime":    True,
        "viz_incarceration_ma12":     True,
        "viz_waiting_hist":           True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_session_state()


# ===========================================================================
# ██╗   ██╗██╗
# ██║   ██║██║
# ██║   ██║██║
# ██║   ██║██║
# ╚██████╔╝██║
#  ╚═════╝ ╚═╝
# APP LAYOUT STARTS HERE
# ===========================================================================

# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.title("⚖️ Stacked Behind Bars Simulation")
st.markdown(
    """
    This tool simulates how bottlenecks in Brazil's judicial processing pipeline
    can drive incarceration levels — and how system capacity, staffing, and
    case-flow assumptions interact. You can test different scenarios and observe
    how the prison population might respond under varying conditions.

    Results are **directional scenario explorations**, not literal predictions of
    real-world outcomes. Use them to reason about systemic pressure, not to forecast
    exact headcounts.

    This simulation is part of the broader capstone project **Stacked Behind Bars**,
    whose interactive visual story is available at
    [lookerstudio.google.com/s/pIu3L0099UQ](https://lookerstudio.google.com/s/pIu3L0099UQ).
    """
)

with st.expander("📖 How to use this app", expanded=False):
    st.markdown(
        """
        **Step 1 — Choose a starting point.**
        In the sidebar, select a comarca (judicial district) to pre-load its
        real-world parameters, or keep the **Default scenario** which uses
        national-level aggregate figures.

        **Step 2 — Adjust parameters.**
        All controls live in the right sidebar. You can modify arrest rates,
        queue and station counts, capacity thresholds, sentence-length distributions,
        and detailed crime-group profiles.

        **Step 3 — Run the simulation.**
        Click **Run Simulation** in the *Simulation Results* tab. Results will
        appear below the button. The *Experimental Analysis* tab lets you vary
        one system lever at a time. The *Theoretical Analysis* tab shows the
        mathematical utilisation curve.

        **Step 4 — Interpret the charts.**
        - A rising pre-trial population suggests a judicial processing backlog.
        - A total population crossing the red dashed line indicates overcrowding.
        - Crime-group charts reveal which categories dominate the simulated mix.
        - The waiting-time histogram illustrates the typical delay experienced
          before sentencing.

        Results are stochastic — run multiple trials to average out randomness.
        """
    )

st.divider()

def apply_comarca_defaults():
    use_default = st.session_state.get("use_default_scenario", True)
    selected = st.session_state.get("selected_comarca", _comarca_names[0] if _comarca_names else "")

    if use_default:
        comarca_row = pd.Series(dtype=float)
        is_default = True
    else:
        comarca_row = get_comarca_row(_comarca_df, selected)
        is_default = False

    # Core parameters
    st.session_state["arrests_per_month"] = (
        float(_national_arrests_per_month)
        if is_default else safe_float(comarca_row.get("arrests_per_month"), _national_arrests_per_month)
    )

    st.session_state["pre_trial_capacity"] = (
        int(_national_pretrial_cap)
        if is_default else int(safe_float(comarca_row.get("pre_trial_capacity_threshold"), _national_pretrial_cap))
    )

    st.session_state["total_capacity"] = (
        int(_national_total_cap)
        if is_default else int(safe_float(comarca_row.get("total_capacity_threshold"), _national_total_cap))
    )

    # Sentence distribution
    use_default_sentence = True
    if not is_default and not comarca_row.empty:
        total_sentence_count = safe_float(comarca_row.get("total count per sentence time"), 0.0)
        use_default_sentence = (total_sentence_count == 0.0)

    for sent_key in SENTENCE_LABELS:
        if use_default_sentence:
            default_val = DEFAULT_SENTENCE_COUNTS[sent_key]
        else:
            matched_col = next(
                (c for c, k in SENTENCE_CSV_COL_MAP.items() if k == sent_key and c in comarca_row.index),
                None,
            )
            raw = comarca_row.get(matched_col, DEFAULT_SENTENCE_COUNTS[sent_key])
            default_val = int(safe_float(raw, DEFAULT_SENTENCE_COUNTS[sent_key]))

        st.session_state[f"sentence_{sent_key}"] = int(default_val)

    # Crime arrival probabilities
    use_default_arrivals = True
    if not is_default and not comarca_row.empty:
        sum_arrival = safe_float(comarca_row.get("sum arrival probabilities"), 0.0)
        use_default_arrivals = (sum_arrival == 0.0)

    for crime_key, defaults in DEFAULT_CRIME_PARAMS.items():
        if use_default_arrivals:
            arr_default = defaults["arrival_probability"]
        else:
            csv_arr_col = next((c for c, k in ARRIVAL_CSV_COL_MAP.items() if k == crime_key), None)
            raw_arr = comarca_row.get(csv_arr_col, defaults["arrival_probability"]) if csv_arr_col else defaults["arrival_probability"]
            arr_default = safe_float(raw_arr, defaults["arrival_probability"])

        st.session_state[f"arr_{crime_key}"] = float(arr_default)

if "use_default_scenario" not in st.session_state:
    st.session_state["use_default_scenario"] = True
if "selected_comarca" not in st.session_state:
    st.session_state["selected_comarca"] = _comarca_names[0] if _comarca_names else ""
apply_comarca_defaults()

# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="section-title">⚙️ Simulation Controls</div>',
        unsafe_allow_html=True,
    )

    # -----------------------------------------------------------------------
    # COMARCA SELECTOR
    # -----------------------------------------------------------------------
    section_title("Select a Comarca or Use the Default Scenario")
    note(
        "Choose a comarca (judicial district) to pre-load its parameters, "
        "or keep the default to use national-level aggregate figures. "
        "You can edit any value after loading."
    )

    use_default_scenario = st.checkbox(
        "Use Default Scenario",
        key="use_default_scenario",
        on_change=apply_comarca_defaults,
        help=(
            "When checked, the app uses national-level aggregate defaults. "
            "When unchecked, you can pre-load a comarca while still editing any field manually."
        ),
    )

    selected_comarca = st.selectbox(
        "Comarca",
        options=_comarca_names if _comarca_names else ["No comarca data available"],
        key="selected_comarca",
        on_change=apply_comarca_defaults,
        disabled=use_default_scenario or not bool(_comarca_names),
        help=(
            "Selecting a comarca pre-loads arrest rates, capacities, and "
            "crime-group proportions from the dataset. All values remain editable."
        ),
    )

    if use_default_scenario:
        st.caption("Using national aggregate defaults.")
    elif _comarca_names:
        st.caption(f"Pre-loading values for: {selected_comarca}")

    is_default = use_default_scenario
    comarca_row = (
        pd.Series(dtype=float)
        if is_default
        else get_comarca_row(_comarca_df, selected_comarca)
    )

    st.caption(f"Comarca dataset source: {_comarca_source}")

    # -----------------------------------------------------------------------
    # SECTION I — CORE SIMULATION PARAMETERS
    # -----------------------------------------------------------------------
    st.divider()
    section_title("I. Core Simulation Parameters")
    note(
        "These settings control the fundamental dynamics of the simulation: "
        "how many people enter the system, how many processing pathways exist, "
        "and how long the experiment runs."
    )

    # -- Arrests per month --
    arrests_per_month = st.number_input(
        "Arrests per Month",
        min_value=1.0,
        max_value=100000.0,
        step=1.0,
        format="%.1f",
        key="arrests_per_month",
        help=(
            "Average number of people entering the system each month. "
            "Higher values increase system pressure and can accelerate overcrowding."
        ),
    )

    num_queues = st.number_input(
        "Number of Litigation Queues",
        min_value=1,
        max_value=100,
        step=1,
        key="num_queues",
        help=(
            "A queue here represents a separate processing stream — think of it "
            "as a separate courtroom track. More queues spread the caseload "
            "and can reduce individual waiting times."
        ),
    )

    num_service_stations = st.number_input(
        "Number of Litigation Stations (Judges per Queue)",
        min_value=1,
        max_value=100,
        step=1,
        key="num_service_stations",
        help=(
            "The number of judges (parallel trial servers) operating within "
            "each queue. Adding stations increases throughput inside a single queue."
        ),
    )

    run_until = st.number_input(
        "Simulation Duration (Months)",
        min_value=1,
        max_value=600,
        step=1,
        key="run_until",
        help="Total simulated time in months. Longer runs reveal steady-state behaviour.",
    )

    prob_private_defense = st.number_input(
        "Probability of Private Defense",
        min_value=0.0,
        max_value=1.0,
        step=0.01,
        format="%.2f",
        key="prob_private_defense",
        help=(
            "Fraction of defendants represented by private (as opposed to public) "
            "defenders. Private defense is associated with shorter decision-waiting "
            "times in the model."
        ),
    )

    num_trials = st.slider(
        "Number of Trials (Stochastic Repetitions)",
        min_value=1,
        max_value=500,
        step=1,
        key="num_trials",
        help=(
            "How many independent simulation runs to average over. "
            "More trials produce smoother, more reliable confidence intervals "
            "but increase computation time proportionally."
        ),
    )
    note(f"ℹ️ Currently set to **{num_trials}** trials. Values above ~50 may take a minute or two.")

    # -----------------------------------------------------------------------
    # SECTION II — CAPACITY THRESHOLDS
    # -----------------------------------------------------------------------
    st.divider()
    section_title("II. Capacity Thresholds")
    note(
        "Thresholds appear as dashed lines on the incarceration charts. "
        "Crossing a threshold signals overcrowding in the model."
    )

    pre_trial_capacity = st.number_input(
        "Pre-Trial Capacity Threshold",
        min_value=0,
        max_value=500000,
        step=1,
        key="pre_trial_capacity",
        help=(
            "Maximum number of people allowed in pre-trial detention (waiting for "
            "trial + currently on trial + awaiting a verdict). Exceeding this signals "
            "provisional overcrowding."
        ),
    )

    total_capacity = st.number_input(
        "Total Capacity Threshold",
        min_value=0,
        max_value=1000000,
        step=1,
        key="total_capacity",
        help=(
            "Maximum total prison population, including both pre-trial detainees "
            "and convicted individuals serving sentences."
        ),
    )

    if pre_trial_capacity >= total_capacity:
        st.warning(
            "Pre-trial capacity should normally be lower than total capacity. "
            "Please review your threshold values."
        )

    # -----------------------------------------------------------------------
    # SECTION III — SENTENCE TIME DISTRIBUTION
    # -----------------------------------------------------------------------
    st.divider()
    section_title("III. Sentence Length Distribution")
    note(
        "These counts represent how many convicted individuals fall into each "
        "sentence-length bracket. The simulation normalises them into probabilities "
        "used when sampling conviction outcomes. Relative proportions matter more "
        "than absolute values."
    )

    # Determine whether to pre-load comarca sentence counts
    _use_default_sentence = True
    if not is_default and not comarca_row.empty:
        _total_sentence_count = safe_float(
            comarca_row.get("total count per sentence time"), 0.0
        )
        _use_default_sentence = (_total_sentence_count == 0.0)

    sentence_counts = {}
    for key, label in SENTENCE_LABELS.items():
        if _use_default_sentence:
            _default_val = DEFAULT_SENTENCE_COUNTS[key]
        else:
            # Map CSV column name to internal key
            _csv_col = f"count per sentence time - {key}"
            # Also try the exact mapped name
            _matched_col = next(
                (c for c, k in SENTENCE_CSV_COL_MAP.items() if k == key and c in comarca_row.index),
                None,
            )
            _raw = comarca_row.get(_matched_col or _csv_col, DEFAULT_SENTENCE_COUNTS[key])
            _default_val = int(safe_float(_raw, DEFAULT_SENTENCE_COUNTS[key]))

        sentence_counts[key] = st.number_input(
            label,
            min_value=0,
            value=int(_default_val),
            step=1,
            key=f"sentence_{key}",
        )

    _sentence_total = sum(sentence_counts.values())
    st.caption(f"Total sentence count currently in use: **{_sentence_total:,}**")

    if _sentence_total == 0:
        st.warning(
            "⚠️ All sentence counts are zero. The default national counts will be "
            "used as a fallback to keep the simulation running."
        )
        _active_sentence_counts = DEFAULT_SENTENCE_COUNTS.copy()
    else:
        _active_sentence_counts = sentence_counts.copy()

    # Build sentence distribution (used by crime-profile builder below)
    sentence_support, sentence_probs = build_sentence_probs(_active_sentence_counts)

    # -----------------------------------------------------------------------
    # SECTION IV — CRIME PROFILES
    # -----------------------------------------------------------------------
    st.divider()
    section_title("IV. Crime Profiles")
    note(
        "Each crime group has an arrival probability (how common it is in the "
        "system) and a conviction probability (how often it leads to a sentence). "
        "Arrival probabilities are normalised automatically before sampling, so only "
        "their relative magnitudes matter."
    )

    # Determine whether comarca arrival probabilities should override defaults
    _use_default_arrivals = True
    if not is_default and not comarca_row.empty:
        _sum_arrival = safe_float(comarca_row.get("sum arrival probabilities"), 0.0)
        _use_default_arrivals = (_sum_arrival == 0.0)

    # Build the editable crime-params dictionary
    crime_params_ui = {}
    for key, defaults in DEFAULT_CRIME_PARAMS.items():
        display_name = defaults["display_name"]

        # Arrival probability: use comarca value if available
        if _use_default_arrivals:
            _arr_default = defaults["arrival_probability"]
        else:
            _csv_arr_col = next(
                (c for c, k in ARRIVAL_CSV_COL_MAP.items() if k == key),
                None,
            )
            _raw_arr = comarca_row.get(_csv_arr_col, defaults["arrival_probability"]) \
                if _csv_arr_col else defaults["arrival_probability"]
            _arr_default = safe_float(_raw_arr, defaults["arrival_probability"])

        with st.expander(f"🔹 {display_name}", expanded=False):
            arr_prob = st.number_input(
                "Arrival Probability (relative weight)",
                min_value=0.0,
                max_value=1.0,
                value=float(_arr_default),
                step=0.001,
                format="%.5f",
                key=f"arr_{key}",
                help=(
                    "Relative frequency of this crime type among arrested individuals. "
                    "All values are normalised before use, so only ratios matter."
                ),
            )
            conv_prob = st.number_input(
                "Conviction Probability",
                min_value=0.0,
                max_value=1.0,
                value=float(defaults["conviction_probability"]),
                step=0.01,
                format="%.2f",
                key=f"conv_{key}",
                help="Probability that an individual with this crime type is convicted.",
            )

            # Advanced distribution parameters (hidden by default)
            show_advanced = st.toggle(
                "Show advanced distribution parameters",
                value=False,
                key=f"adv_{key}",
            )

            svc_params = copy.deepcopy(defaults["service_time"])
            pub_params = copy.deepcopy(defaults["public_decision"])
            prv_params = copy.deepcopy(defaults["private_decision"])

            if show_advanced:
                st.markdown("**Trial Service Time (months)**")
                _c1, _c2, _c3, _c4 = st.columns(4)
                svc_params["mean"]  = _c1.number_input("Mean",  value=svc_params["mean"],  step=0.1, key=f"svc_mean_{key}")
                svc_params["std"]   = _c2.number_input("Std",   value=svc_params["std"],   step=0.05, min_value=0.01, key=f"svc_std_{key}")
                svc_params["lower"] = _c3.number_input("Lower", value=svc_params["lower"], step=0.05, key=f"svc_lo_{key}")
                svc_params["upper"] = _c4.number_input("Upper", value=svc_params["upper"], step=0.1, key=f"svc_hi_{key}")

                st.markdown("**Public Defense Decision Wait (months)**")
                _c1, _c2, _c3, _c4 = st.columns(4)
                pub_params["mean"]  = _c1.number_input("Mean",  value=pub_params["mean"],  step=0.5, key=f"pub_mean_{key}")
                pub_params["std"]   = _c2.number_input("Std",   value=pub_params["std"],   step=0.1, min_value=0.01, key=f"pub_std_{key}")
                pub_params["lower"] = _c3.number_input("Lower", value=pub_params["lower"], step=0.1, key=f"pub_lo_{key}")
                pub_params["upper"] = _c4.number_input("Upper", value=pub_params["upper"], step=0.5, key=f"pub_hi_{key}")

                st.markdown("**Private Defense Decision Wait (months)**")
                _c1, _c2, _c3, _c4 = st.columns(4)
                prv_params["mean"]  = _c1.number_input("Mean",  value=prv_params["mean"],  step=0.5, key=f"prv_mean_{key}")
                prv_params["std"]   = _c2.number_input("Std",   value=prv_params["std"],   step=0.1, min_value=0.01, key=f"prv_std_{key}")
                prv_params["lower"] = _c3.number_input("Lower", value=prv_params["lower"], step=0.1, key=f"prv_lo_{key}")
                prv_params["upper"] = _c4.number_input("Upper", value=prv_params["upper"], step=0.5, key=f"prv_hi_{key}")

                # Validate advanced parameters
                for label_adv, params_adv in [
                    ("Service Time", svc_params),
                    ("Public Decision Wait", pub_params),
                    ("Private Decision Wait", prv_params),
                ]:
                    if params_adv["std"] <= 0:
                        st.error(f"{display_name} — {label_adv}: Std must be > 0.")
                    if params_adv["lower"] >= params_adv["upper"]:
                        st.error(f"{display_name} — {label_adv}: Lower must be < Upper.")

        crime_params_ui[key] = {
            "arrival_probability":    arr_prob,
            "conviction_probability": conv_prob,
            "service_time":   svc_params,
            "public_decision":  pub_params,
            "private_decision": prv_params,
        }

    # Validate arrival probability sum
    _arr_total = sum(p["arrival_probability"] for p in crime_params_ui.values())
    if _arr_total == 0.0:
        st.error(
            "All arrival probabilities are zero. "
            "Please set at least one crime group to a positive arrival probability."
        )

    # -----------------------------------------------------------------------
    # SECTION V — VISUALISATION CONTROLS
    # -----------------------------------------------------------------------
    st.divider()
    section_title("V. Visualisation Controls")
    note("Toggle which charts are displayed after running the simulation.")

    viz_incarceration_multi  = st.checkbox("Incarceration levels over time (multiple trials)", value=True)
    viz_incarceration_crime  = st.checkbox("Incarceration by crime group", value=True)
    viz_incarceration_ma12   = st.checkbox("Incarceration over time – MA12 only (by crime type)", value=True)
    viz_waiting_hist         = st.checkbox("Waiting-time histogram", value=True)

    # Experimental analysis toggles
    st.divider()
    section_title("Experimental Visualisation Controls")
    exp_viz_wt_queues   = st.checkbox("Avg. waiting time vs. queues",       value=True)
    exp_viz_inc_queues  = st.checkbox("Incarceration vs. queues",           value=True)
    exp_viz_wt_stations = st.checkbox("Avg. waiting time vs. stations",     value=True)
    exp_viz_inc_stations = st.checkbox("Incarceration vs. stations",        value=True)
    exp_viz_ratio       = st.checkbox("Arrival rate vs. pre-trial ratio",   value=True)
    exp_viz_ratio_zoom  = st.checkbox("Zoomed arrival rate vs. pre-trial ratio", value=True)


# ---------------------------------------------------------------------------
# VALIDATE & ASSEMBLE SIMULATION PARAMETERS
# ---------------------------------------------------------------------------

def _validate_and_build_sim_params():
    """
    Validates all sidebar inputs and builds the ``sim_params`` dictionary
    ready to be passed into the simulation functions.

    Returns
    -------
    sim_params : dict or None
        Assembled parameter dict, or None if validation fails.
    error_msg : str or None
        Human-readable error description if validation fails.
    """
    errors = []

    # Validate arrest rate
    if arrests_per_month <= 0:
        errors.append("Arrests per month must be greater than zero.")

    # Validate arrival total
    if _arr_total <= 0:
        errors.append("At least one crime group must have a positive arrival probability.")

    # Validate truncnorm parameters for each crime group
    for key, params in crime_params_ui.items():
        display = DEFAULT_CRIME_PARAMS[key]["display_name"]
        for dist_label, dist_key in [
            ("Service Time", "service_time"),
            ("Public Decision Wait", "public_decision"),
            ("Private Decision Wait", "private_decision"),
        ]:
            d = params[dist_key]
            if d["std"] <= 0:
                errors.append(f"{display} — {dist_label}: Std must be > 0.")
            if d["lower"] >= d["upper"]:
                errors.append(f"{display} — {dist_label}: Lower must be < Upper.")

    if errors:
        return None, "\n".join(f"• {e}" for e in errors)

    try:
        # Build sentence distribution
        _sc, _sp = build_sentence_probs(_active_sentence_counts)

        # Build crime profiles
        profiles = build_crime_profiles(crime_params_ui, _sc, _sp)

        # Build arrests object and arrest rate distribution
        arrests_obj       = sim.Arrests(profiles)
        arrest_rate_dist  = build_arrest_rate_dist(arrests_per_month)

        sim_params = {
            "arrest_rate_dist":           arrest_rate_dist,
            "arrests":                    arrests_obj,
            "num_queues":                 int(num_queues),
            "num_service_stations":       int(num_service_stations),
            "capacity_threshold":         int(total_capacity),
            "pre_trial_capacity_threshold": int(pre_trial_capacity),
            "prob_private_defense":       float(prob_private_defense),
            "run_until":                  int(run_until),
            "is_print":                   False,
            "progress_bar":               False,
        }
        return sim_params, None

    except Exception as exc:
        return None, f"Parameter construction error: {exc}"


# ===========================================================================
# MAIN TABS
# ===========================================================================

tab_sim, tab_exp, tab_theory = st.tabs(
    ["📊 Simulation Results", "🔬 Experimental Analysis", "📐 Theoretical Analysis"]
)


# ===========================================================================
# TAB 1 — SIMULATION RESULTS
# ===========================================================================

with tab_sim:
    st.markdown(
        """
        ### Run Your Scenario
        Configure parameters in the sidebar, then click **Run Simulation** below.
        The charts that appear are based on the parameter set you have chosen.
        Because this is a stochastic model, each run introduces some randomness;
        increasing the number of trials in the sidebar will smooth out that noise.

        > **Interpretive note:** Charts with a moving-average (MA) overlay help you
        > see the underlying trend past short-term fluctuations.
        """
    )

    run_button = st.button("▶ Run Simulation", type="primary", use_container_width=True)

    if run_button:
        sim_params, error_msg = _validate_and_build_sim_params()

        if error_msg:
            st.error(f"Cannot run simulation — please fix the following:\n\n{error_msg}")
        else:
            with st.spinner(
                f"Running {num_trials} simulation trial(s) for {run_until} months each… "
                "This may take a moment."
            ):
                try:
                    _stdout_buf = io.StringIO()
                    _old = sys.stdout
                    sys.stdout = _stdout_buf   # Suppress simulation print output

                    court_system_list = sim.run_multiple_simulations(
                        num_trials=num_trials,
                        run_simulation_func=sim.run_simulation,
                        sim_params=sim_params,
                    )

                    sys.stdout = _old

                    st.session_state["court_system_list"] = court_system_list
                    st.success(
                        f"✅ Simulation complete — {num_trials} trial(s) finished."
                    )

                except Exception as exc:
                    sys.stdout = _old
                    st.error(
                        f"The simulation encountered an unexpected error:\n\n`{exc}`\n\n"
                        "Common causes: very short run duration with high arrest rates, "
                        "or extreme distribution parameters. Try reducing arrests per month "
                        "or increasing the simulation duration."
                    )

    # -----------------------------------------------------------------------
    # DISPLAY RESULTS (persisted in session state)
    # -----------------------------------------------------------------------
    court_system_list = st.session_state.get("court_system_list")

    if court_system_list:
        st.divider()

        # -- Chart 1: Incarceration over time (multiple trials) --
        if viz_incarceration_multi:
            st.markdown(
                """
                #### Incarceration Levels Over Time
                Each dot represents the mean population across all trials at that
                time point, with 95% confidence interval bars.
                A **rising pre-trial population** (blue) suggests a growing processing
                backlog. When total population (red) crosses the dashed threshold line,
                the system is simulated to be overcrowded.
                """
            )
            try:
                # Patch plt.show so the figure is not discarded
                _orig_show = plt.show
                plt.show = lambda *a, **kw: None

                sim.plot_incarceration_multiple(
                    court_system_list=court_system_list,
                    breakdowns_to_plot=["total", "pre-trial", "convicted"],
                    capacity_threshold=int(total_capacity),
                    pre_trial_capacity_threshold=int(pre_trial_capacity),
                    summary_start=0,
                    summary_end=int(run_until),
                    title="Incarceration Over Time (Multiple Trials)",
                )
                plt.show = _orig_show
                capture_and_render_current_figure()

            except Exception as exc:
                plt.show = _orig_show
                st.warning(f"Could not render incarceration chart: {exc}")

        # -- Chart 2: Incarceration by crime group --
        if viz_incarceration_crime:
            st.markdown(
                """
                #### Incarceration by Crime Group
                This chart breaks the simulated prison population down by crime
                category over time. It reveals which groups contribute most to
                overall incarceration and whether their relative shares shift as
                the system reaches capacity.
                """
            )
            try:
                _orig_show = plt.show
                plt.show = lambda *a, **kw: None

                sim.plot_incarceration_by_crime_multiple(
                    court_system_list=court_system_list,
                    summary_start=0,
                    summary_end=int(run_until),
                    title="Incarceration by Crime Type (Multiple Trials)",
                )

                plt.show = _orig_show
                capture_and_render_current_figure()

            except Exception as exc:
                plt.show = _orig_show
                st.warning(f"Could not render crime-group chart: {exc}")

        # -- Chart 3: Incarceration by crime type — MA12 only --
        if viz_incarceration_ma12:
            st.markdown(
                """
                #### Incarceration Over Time by Crime Type — 12-Month Moving Average
                A 12-month moving average applied to each crime category smooths
                out trial-to-trial noise and makes medium-term trends much easier
                to read. Use this chart to compare how different crime groups grow
                or stabilise over the simulation window.
                """
            )
            try:
                # Use the first trial only (single-run plot with MA overlay)
                _orig_show = plt.show
                plt.show = lambda *a, **kw: None

                sim.plot_incarceration(
                    court_system=court_system_list[0],
                    breakdowns_to_plot=["by_crime_type"],
                    capacity_threshold=int(total_capacity),
                    pre_trial_capacity_threshold=int(pre_trial_capacity),
                    title="Incarceration by Crime Type",
                    moving_average=True,
                    window=12,
                )

                plt.show = _orig_show
                capture_and_render_current_figure()

            except Exception as exc:
                plt.show = _orig_show
                st.warning(f"Could not render MA12 crime-type chart: {exc}")

        # -- Chart 4: Waiting-time histogram --
        if viz_waiting_hist:
            st.markdown(
                """
                #### Time in System Before Sentencing
                This histogram shows the total time each simulated individual
                spent in the system before receiving a verdict — from arrest
                through queue waiting, trial, and post-trial decision waiting.
                Mean and median lines are marked. A long tail indicates that a
                subset of people experience disproportionately long delays.
                """
            )
            try:
                _orig_show = plt.show
                plt.show = lambda *a, **kw: None

                sim.plot_time_before_sentence(
                    court_system=court_system_list[0],
                    title="Total Time in System Before Sentencing",
                )

                plt.show = _orig_show
                capture_and_render_current_figure()

            except Exception as exc:
                plt.show = _orig_show
                st.warning(f"Could not render waiting-time histogram: {exc}")

    elif not run_button:
        st.info(
            "Configure the parameters in the sidebar and click **▶ Run Simulation** "
            "to generate charts."
        )


# ===========================================================================
# TAB 2 — EXPERIMENTAL ANALYSIS
# ===========================================================================

with tab_exp:
    st.markdown(
        """
        ### Experimental Analysis
        Each chart in this section varies **one system lever at a time** while
        holding all other parameters constant. This allows you to isolate the
        effect of adding more queues or more judge stations, or to see how
        changes in arrest volume shift the pre-trial detention ratio.

        > **Tip:** When the population appears to have stabilised in the main
        > simulation tab, try returning here and adding more queues or stations
        > to see how throughput capacity changes the outcome. Even a single
        > additional queue can significantly reduce waiting times in congested
        > scenarios.

        Click the individual **Run** buttons below to execute each experiment.
        Experiments are cached within your session so you don't need to re-run
        them every time you switch tabs.
        """
    )

    # Assemble a base sim_params for experiments (without running a full trial set)
    _exp_params, _exp_error = _validate_and_build_sim_params()

    if _exp_error:
        st.error(
            f"Fix parameter errors in the sidebar before running experiments:\n\n{_exp_error}"
        )
    else:
        # ----------------------------------------------------------------
        # QUEUE EXPERIMENT CONTROLS
        # ----------------------------------------------------------------
        st.divider()
        section_title("Queue-Varying Experiments")
        note(
            "These experiments vary the number of simultaneous litigation queues "
            "from a minimum to a maximum value, running multiple trials for each "
            "queue count to produce stable estimates."
        )

        _qcol1, _qcol2, _qcol3 = st.columns(3)
        q_min = _qcol1.number_input(
            "Min Queues", min_value=1, max_value=50, value=1, step=1,
            help="Smallest number of queues to test.", key="q_min"
        )
        q_max = _qcol2.number_input(
            "Max Queues", min_value=1, max_value=50, value=10, step=1,
            help="Largest number of queues to test.", key="q_max"
        )
        q_exp_trials = _qcol3.number_input(
            "Trials per Config.", min_value=1, max_value=100, value=5, step=1,
            help="Simulation trials per queue count. Keep low (≤10) for speed.",
            key="q_trials"
        )

        if q_min >= q_max:
            st.warning("Min Queues must be less than Max Queues.")
        else:
            num_queues_list_exp = list(range(int(q_min), int(q_max) + 1))
            _exp_summary_end    = int(run_until)

            run_queue_exp = st.button(
                "▶ Run Queue Experiments", key="run_queue_exp", use_container_width=True
            )

            if run_queue_exp:
                with st.spinner("Running queue-varying experiments…"):
                    try:
                        _orig_show = plt.show
                        plt.show = lambda *a, **kw: None

                        _wt_q = sim.analyze_waiting_time_vs_queues(
                            num_queues_list=num_queues_list_exp,
                            num_trials=int(q_exp_trials),
                            run_simulation_func=sim.run_simulation,
                            sim_params_base=_exp_params,
                            summary_start=0,
                            summary_end=_exp_summary_end,
                            progress_bar=False,
                        )
                        _inc_q = sim.analyze_queues_vs_incarceration(
                            num_queues_list=num_queues_list_exp,
                            num_trials=int(q_exp_trials),
                            run_simulation_func=sim.run_simulation,
                            sim_params_base=_exp_params,
                            summary_start=0,
                            summary_end=_exp_summary_end,
                            progress_bar=False,
                        )

                        plt.show = _orig_show
                        st.session_state["exp_wt_queues_result"]  = _wt_q
                        st.session_state["exp_inc_queues_result"] = _inc_q
                        st.success("Queue experiments complete.")

                    except Exception as exc:
                        plt.show = _orig_show
                        st.error(f"Queue experiment error: {exc}")

            # Display queue experiment results
            _wt_q_res  = st.session_state.get("exp_wt_queues_result")
            _inc_q_res = st.session_state.get("exp_inc_queues_result")

            if _wt_q_res and exp_viz_wt_queues:
                st.markdown(
                    """
                    #### Impact of Simultaneous Litigation Queues on Average Waiting Time
                    As more queues (processing streams) are added, cases are distributed
                    more broadly and individuals generally wait less before their trial begins.
                    The error bars represent 95% confidence intervals across trials.
                    """
                )
                try:
                    _orig_show = plt.show
                    plt.show = lambda *a, **kw: None
                    sim.plot_waiting_times_vs_queues(results_dict=_wt_q_res)
                    plt.show = _orig_show
                    capture_and_render_current_figure()
                except Exception as exc:
                    plt.show = _orig_show
                    st.warning(f"Chart error: {exc}")

            if _inc_q_res and exp_viz_inc_queues:
                st.markdown(
                    """
                    #### Impact of Litigation Queue Count on Incarceration Population
                    This chart shows how the total, pre-trial, and convicted populations
                    change as more queues are added. In congested scenarios, adding queues
                    can meaningfully reduce pre-trial population by accelerating case flow.
                    """
                )
                try:
                    _orig_show = plt.show
                    plt.show = lambda *a, **kw: None
                    sim.plot_incarceration_vs_queues(
                        results_dict=_inc_q_res,
                        capacity_threshold=int(total_capacity),
                        pre_trial_capacity_threshold=int(pre_trial_capacity),
                    )
                    plt.show = _orig_show
                    capture_and_render_current_figure()
                except Exception as exc:
                    plt.show = _orig_show
                    st.warning(f"Chart error: {exc}")

        # ----------------------------------------------------------------
        # STATION EXPERIMENT CONTROLS
        # ----------------------------------------------------------------
        st.divider()
        section_title("Station-Varying Experiments")
        note(
            "These experiments hold the number of queues fixed at the value you "
            "set in the sidebar and instead vary the number of judge stations "
            "(parallel servers) inside each queue. This isolates the effect of "
            "adding processing capacity within an existing court track."
        )

        _scol1, _scol2, _scol3 = st.columns(3)
        s_min = _scol1.number_input(
            "Min Stations", min_value=1, max_value=50, value=1, step=1,
            help="Smallest number of stations per queue to test.", key="s_min"
        )
        s_max = _scol2.number_input(
            "Max Stations", min_value=1, max_value=50, value=10, step=1,
            help="Largest number of stations per queue to test.", key="s_max"
        )
        s_exp_trials = _scol3.number_input(
            "Trials per Config.", min_value=1, max_value=100, value=5, step=1,
            help="Simulation trials per station count.", key="s_trials"
        )

        if s_min >= s_max:
            st.warning("Min Stations must be less than Max Stations.")
        else:
            num_stations_list_exp = list(range(int(s_min), int(s_max) + 1))

            run_station_exp = st.button(
                "▶ Run Station Experiments", key="run_station_exp", use_container_width=True
            )

            if run_station_exp:
                with st.spinner("Running station-varying experiments…"):
                    try:
                        _orig_show = plt.show
                        plt.show = lambda *a, **kw: None

                        _wt_s = sim.analyze_waiting_time_vs_stations(
                            num_stations_list=num_stations_list_exp,
                            num_trials=int(s_exp_trials),
                            run_simulation_func=sim.run_simulation,
                            sim_params_base=_exp_params,
                            summary_start=0,
                            summary_end=int(run_until),
                            progress_bar=False,
                        )
                        _inc_s = sim.analyze_stations_vs_incarceration(
                            num_stations_list=num_stations_list_exp,
                            num_trials=int(s_exp_trials),
                            run_simulation_func=sim.run_simulation,
                            sim_params_base=_exp_params,
                            summary_start=0,
                            summary_end=int(run_until),
                            progress_bar=False,
                        )

                        plt.show = _orig_show
                        st.session_state["exp_wt_stations_result"]  = _wt_s
                        st.session_state["exp_inc_stations_result"] = _inc_s
                        st.success("Station experiments complete.")

                    except Exception as exc:
                        plt.show = _orig_show
                        st.error(f"Station experiment error: {exc}")

            _wt_s_res  = st.session_state.get("exp_wt_stations_result")
            _inc_s_res = st.session_state.get("exp_inc_stations_result")

            if _wt_s_res and exp_viz_wt_stations:
                st.markdown(
                    """
                    #### Impact of Station Count on Average Waiting Time
                    Adding more judge servers within a single queue reduces per-person
                    waiting time up to a point; beyond sufficient capacity the gains
                    diminish. This chart reveals where those diminishing returns begin.
                    """
                )
                try:
                    _orig_show = plt.show
                    plt.show = lambda *a, **kw: None
                    sim.plot_waiting_times_vs_stations(results_dict=_wt_s_res)
                    plt.show = _orig_show
                    capture_and_render_current_figure()
                except Exception as exc:
                    plt.show = _orig_show
                    st.warning(f"Chart error: {exc}")

            if _inc_s_res and exp_viz_inc_stations:
                st.markdown(
                    """
                    #### Impact of Station Count on Incarceration Population
                    This chart shows how total, pre-trial, and convicted populations
                    respond to varying server capacity within each queue, with the
                    capacity threshold lines drawn for reference.
                    """
                )
                try:
                    _orig_show = plt.show
                    plt.show = lambda *a, **kw: None
                    sim.plot_incarceration_vs_stations(
                        results_dict=_inc_s_res,
                        capacity_threshold=int(total_capacity),
                        pre_trial_capacity_threshold=int(pre_trial_capacity),
                    )
                    plt.show = _orig_show
                    capture_and_render_current_figure()
                except Exception as exc:
                    plt.show = _orig_show
                    st.warning(f"Chart error: {exc}")

        # ----------------------------------------------------------------
        # ARRIVAL RATE vs PRE-TRIAL RATIO
        # ----------------------------------------------------------------
        st.divider()
        section_title("Arrival Rate vs. Pre-Trial Detention Ratio")
        note(
            "This analysis shows how system-level arrest pressure relates to "
            "reliance on pre-trial detention. When the arrival rate is high relative "
            "to processing capacity, a larger fraction of the total incarcerated "
            "population consists of people still awaiting a verdict."
        )

        _rcol1, _rcol2 = st.columns(2)
        ratio_max_rate = _rcol1.number_input(
            "Max arrest rate to test (per month)",
            min_value=5.0, max_value=5000.0, value=200.0, step=10.0,
            key="ratio_max"
        )
        ratio_trials = _rcol2.number_input(
            "Trials per rate", min_value=1, max_value=50, value=5, step=1,
            key="ratio_trials"
        )
        _rate_points = st.number_input(
            "Number of test points", min_value=3, max_value=20, value=6, step=1,
            key="ratio_points"
        )
        _arrest_rates_exp = list(
            np.linspace(1.0, float(ratio_max_rate), int(_rate_points))
        )
        _arrest_rates_zoom = list(
            np.linspace(1.0, float(min(ratio_max_rate, 25.0)), int(_rate_points))
        )

        run_ratio_exp = st.button(
            "▶ Run Arrival Rate Analysis", key="run_ratio_exp", use_container_width=True
        )

        if run_ratio_exp:
            with st.spinner("Running arrival-rate analysis…"):
                try:
                    _orig_show = plt.show
                    plt.show = lambda *a, **kw: None

                    _ratio_r = sim.analyze_arrival_rate_vs_pretrial_ratio(
                        arrest_rates_list=_arrest_rates_exp,
                        num_trials=int(ratio_trials),
                        run_simulation_func=sim.run_simulation,
                        sim_params_base=_exp_params,
                        summary_start=0,
                        summary_end=int(run_until),
                        progress_bar=False,
                    )
                    _ratio_z = sim.analyze_arrival_rate_vs_pretrial_ratio(
                        arrest_rates_list=_arrest_rates_zoom,
                        num_trials=int(ratio_trials),
                        run_simulation_func=sim.run_simulation,
                        sim_params_base=_exp_params,
                        summary_start=0,
                        summary_end=int(run_until),
                        progress_bar=False,
                    )

                    plt.show = _orig_show
                    st.session_state["exp_ratio_result"]        = _ratio_r
                    st.session_state["exp_ratio_zoomed_result"] = _ratio_z
                    st.success("Arrival-rate analysis complete.")

                except Exception as exc:
                    plt.show = _orig_show
                    st.error(f"Arrival-rate experiment error: {exc}")

        _ratio_res  = st.session_state.get("exp_ratio_result")
        _ratio_zoom = st.session_state.get("exp_ratio_zoomed_result")

        if _ratio_res and exp_viz_ratio:
            st.markdown(
                """
                #### Arrival Rate vs. Pre-Trial Detention Ratio
                Each point on this chart represents the average ratio of pre-trial
                to total incarceration at a given arrest rate. A steep upward slope
                indicates that increased system inflow disproportionately expands
                the pre-trial detained population relative to convicted sentenced individuals.
                """
            )
            try:
                _fig = plot_pretrial_ratio_themed(
                    _ratio_res,
                    title="Arrival Rate vs. Pre-Trial Detention Ratio",
                )
                render_mpl_figure(_fig)
            except Exception as exc:
                st.warning(f"Chart error: {exc}")

        if _ratio_zoom and exp_viz_ratio_zoom:
            st.markdown(
                """
                #### Zoomed View — Low-Rate Regime
                This is the same analysis restricted to the lower end of the
                arrest-rate spectrum, making it easier to see behaviour before
                the system becomes heavily congested.
                """
            )
            try:
                _fig_z = plot_pretrial_ratio_themed(
                    _ratio_zoom,
                    title="Arrival Rate vs. Pre-Trial Ratio (Zoomed — Low Rates)",
                )
                render_mpl_figure(_fig_z)
            except Exception as exc:
                st.warning(f"Chart error: {exc}")


# ===========================================================================
# TAB 3 — THEORETICAL ANALYSIS
# ===========================================================================

with tab_theory:
    st.markdown(
        """
        ### Theoretical Utilisation Analysis
        Utilisation (ρ) is a measure of how busy the system is relative to its
        handling capacity. It is defined as:

        > **ρ = (λ / Nq) × S**

        where **λ** is the total arrival rate (arrests per month), **Nq** is the
        number of litigation queues, and **S** is the weighted average trial
        service time (months).

        - When **ρ < 1**, the system can process cases faster than they arrive —
          queues remain bounded and waiting times are manageable.
        - When **ρ ≥ 1**, the system is theoretically overloaded — the queue
          will grow without limit and waiting times will increase sharply.

        This chart does not run any simulation; it is purely mathematical and
        updates instantly as you adjust the sliders below.

        > **Why this matters:** Utilisation helps you identify the minimum number
        > of queues needed to keep the system stable. Below the stability threshold
        > line is safe territory; above it, the system is predicted to deteriorate
        > regardless of other interventions.
        """
    )

    st.divider()
    section_title("Utilisation Chart Controls")

    _th_col1, _th_col2 = st.columns(2)

    th_lambda = _th_col1.number_input(
        "Total Arrival Rate λ (arrests per month)",
        min_value=0.1,
        max_value=10000.0,
        value=float(arrests_per_month),
        step=1.0,
        format="%.1f",
        key="th_lambda",
        help=(
            "Monthly arrest rate. Pre-filled from the sidebar value — "
            "adjust independently here to explore theoretical scenarios."
        ),
    )

    # Compute weighted service time from current crime params
    _s_computed = compute_weighted_service_time(crime_params_ui)

    th_s = _th_col2.number_input(
        "Weighted Average Service Time S (months)",
        min_value=0.01,
        max_value=60.0,
        value=round(_s_computed, 3),
        step=0.01,
        format="%.3f",
        key="th_s",
        help=(
            "The arrival-probability-weighted mean trial duration across all crime "
            "groups. Auto-computed from your crime-profile settings — you can "
            "override it here for theoretical exploration."
        ),
    )

    _qr_col1, _qr_col2 = st.columns(2)
    th_min_q = _qr_col1.number_input(
        "Min Queues to Plot",
        min_value=1, max_value=200, value=1, step=1, key="th_min_q"
    )
    th_max_q = _qr_col2.number_input(
        "Max Queues to Plot",
        min_value=1, max_value=200, value=30, step=1, key="th_max_q"
    )

    if th_min_q >= th_max_q:
        st.warning("Min Queues must be less than Max Queues.")
    else:
        try:
            _util_fig = plot_utilization(
                lambda_total=float(th_lambda),
                s_weighted=float(th_s),
                num_queues_range=(int(th_min_q), int(th_max_q)),
            )
            render_mpl_figure(_util_fig)

            # Compute the critical Nq where rho first drops below 1
            _nq_stable = None
            for _nq in range(int(th_min_q), int(th_max_q) + 1):
                if (th_lambda / _nq) * th_s < 1.0:
                    _nq_stable = _nq
                    break

            if _nq_stable is not None:
                st.success(
                    f"With λ = {th_lambda:.1f} arrests/month and S = {th_s:.3f} months, "
                    f"the system reaches theoretical stability at **{_nq_stable} queue(s)**."
                )
            else:
                st.warning(
                    f"With the current parameters (λ = {th_lambda:.1f}, S = {th_s:.3f}), "
                    f"the system remains above the stability threshold across the entire range shown. "
                    "Try increasing the Max Queues value."
                )

        except Exception as exc:
            st.error(f"Could not render utilisation chart: {exc}")

    st.divider()
    st.markdown(
        """
        #### Understanding the Stability Threshold
        The red dashed line marks ρ = 1 — the boundary between a stable and an
        unstable system. In practice, systems often experience degradation well
        before ρ reaches 1, because utilisation does not account for variability
        in arrival times or service durations. Queue theory predicts that as ρ
        approaches 1 from below, mean waiting times grow very rapidly —
        this is why even moderate increases in arrest volume can produce
        disproportionately long backlogs in the simulated system.
        """
    )


# ---------------------------------------------------------------------------
# FOOTER
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Stacked Behind Bars Simulation · Project Developed by Gisele Fretta · "
    "App built with Streamlit · "
    "Visual story: lookerstudio.google.com/s/pIu3L0099UQ"
)
