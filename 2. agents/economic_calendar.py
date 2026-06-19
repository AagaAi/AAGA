# agents/economic_calendar.py
import datetime

# Major high-impact events (UTC times). Add more as needed.
HIGH_IMPACT_EVENTS = {
    "NFP": [
        datetime.datetime(2026, 1, 10, 13, 30),
        datetime.datetime(2026, 2, 7, 13, 30),
        datetime.datetime(2026, 3, 7, 13, 30),
        # ... add actual future dates
    ],
    "CPI": [
        datetime.datetime(2026, 1, 15, 13, 30),
        datetime.datetime(2026, 2, 12, 13, 30),
        # ...
    ],
    "FOMC": [
        datetime.datetime(2026, 1, 29, 19, 0),
        datetime.datetime(2026, 3, 19, 18, 0),
        # ...
    ],
}

def is_high_impact_now():
    """Check if the current time is within 15 minutes of any high-impact event."""
    now = datetime.datetime.utcnow()
    for event, dates in HIGH_IMPACT_EVENTS.items():
        for dt in dates:
            diff_minutes = abs((now - dt).total_seconds() / 60)
            if diff_minutes <= 15:
                return True, event, dt
    return False, None, None

def get_upcoming_events():
    """Return list of upcoming events within next 24 hours."""
    now = datetime.datetime.utcnow()
    upcoming = []
    for event, dates in HIGH_IMPACT_EVENTS.items():
        for dt in dates:
            if now < dt < now + datetime.timedelta(hours=24):
                upcoming.append({"event": event, "time": dt.isoformat()})
    return upcoming
