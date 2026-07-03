import numpy as np
import pandas as pd
from typing import TypeAlias
from dateutil.relativedelta import relativedelta as tdelta
from dateutil.easter import easter

Date: TypeAlias = pd.Timestamp


class Tenor:
    """Tenor string parser and comparator. Internally stores all tenors in month units."""

    MONTH_MAP = dict(D=1/30, W=7/30, M=1, Y=12)

    def __init__(self, tenor: str):
        self.tenor  = tenor.upper()
        self.months = self.__parse(self.tenor)
        self.tenor  = self.__month_to_str(self.months)  # normalise e.g. "12M" -> "1Y"
        self.amount = float(self.tenor[:-1])
        self.unit   = self.tenor[-1]

    def __parse(self, tenor: str) -> float:
        return float(tenor[:-1]) * self.MONTH_MAP[tenor[-1]]

    def __month_to_str(self, months: float) -> str:
        """Convert month float back to canonical tenor string."""
        if months % 12 == 0:
            return f"{int(months // 12)}Y"
        elif months % 1 == 0:
            return f"{int(months)}M"
        else:
            weeks = months * 30 / 7
            if weeks % 1 == 0:
                return f"{int(weeks)}W"
            return f"{int(months * 30)}D"

    # Arithmetic operators — all return a new Tenor
    def __add__(self, other):   return Tenor(self.__month_to_str(self.months + other.months))
    def __sub__(self, other):   return Tenor(self.__month_to_str(self.months - other.months))
    def __mul__(self, amount):  return Tenor(self.__month_to_str(self.months * amount))
    def __truediv__(self, amount): return Tenor(self.__month_to_str(self.months / amount))

    # Comparison operators — compare in month units
    def __lt__(self, other): return self.months < other.months
    def __le__(self, other): return self.months <= other.months
    def __gt__(self, other): return self.months > other.months
    def __ge__(self, other): return self.months >= other.months
    def __eq__(self, other): return self.months == other.months
    def __ne__(self, other): return self.months != other.months

    def __repr__(self): return f"Tenor({self.__month_to_str(self.months)})"


def target_holidays(year_start: int, year_end: int = None):
    """
    TARGET calendar (TE): 6 fixed holidays per year.
    Good Friday and Easter Monday are the only moving dates.
    """
    year_end = year_end or year_start
    holidays = []

    for year in range(year_start, year_end + 1):
        e = pd.to_datetime(easter(year))
        holidays.extend(sorted([
            pd.Timestamp(year=year, month=1,  day=1),   # New Year's Day
            e - tdelta(days=2),                          # Good Friday
            e + tdelta(days=1),                          # Easter Monday
            pd.Timestamp(year=year, month=5,  day=1),   # Labour Day
            pd.Timestamp(year=year, month=12, day=25),  # Christmas
            pd.Timestamp(year=year, month=12, day=26),  # Boxing Day
        ]))

    return [np.datetime64(i.date()) for i in holidays]


class BusinessDay:
    """Wraps a date with a TARGET calendar and a default roll convention."""

    def __init__(self, date, calendar=np.busdaycalendar(), roll="modifiedfollowing"):
        self.date     = np.datetime64(pd.to_datetime(date).date())
        self.roll     = roll
        self.calendar = calendar

    def shift(self, tenor, roll=None):
        """Shift date forward by a tenor, applying business day adjustment."""
        roll  = roll or self.roll
        tenor = Tenor(tenor) if isinstance(tenor, str) else tenor

        if tenor.unit in ("M", "Y"):
            # Add calendar months first, then adjust to nearest business day
            months = tenor.amount * (1 if tenor.unit == "M" else 12)
            date   = np.busday_offset(self.date + tdelta(months=months), 0, roll, busdaycal=self.calendar)
        elif tenor.unit == "D":
            date = np.busday_offset(self.date, tenor.amount, roll, busdaycal=self.calendar)
        elif tenor.unit == "W":
            date = np.busday_offset(self.date, tenor.amount * 5, roll, busdaycal=self.calendar)

        return BusinessDay(date, self.calendar, roll)

    def __repr__(self):
        return f"BusinessDay(date={self.date}, roll={self.roll})"


def day_count_fraction(start_date: Date, end_date: Date, convention: str) -> float:
    """
    Compute the day count fraction between two dates under a given convention.
    Supported: ACT/360, ACT/365, 30U/360 (ISDA), 30E/360 (Eurobond).
    """
    start, end = map(pd.to_datetime, (start_date, end_date))
    conv = convention.upper()

    if conv == "ACT/360":
        return (end - start).days / 360

    elif conv == "ACT/365":
        return (end - start).days / 365

    elif conv in ["30U/360", "30E/360"]:
        if conv == "30U/360":
            last_day = lambda x: x.month != (x + tdelta(days=1)).month

            if start.month == 2:
                if end.month == 2 and last_day(start) and last_day(end):
                    d1 = d2 = 30
                elif last_day(start):
                    d1 = 30
                    d2 = 30 if end.day == 31 else end.day
                else:
                    d1, d2 = start.day, end.day

            elif end.day == 31 and start.day in [30, 31]:
                d1 = d2 = 30
            elif start.day == 31:
                d1, d2 = 30, end.day
            else:
                d1, d2 = start.day, end.day

        else:  # 30E/360: both days capped at 30 independently, no February special case
            d1 = 30 if start.day == 31 else start.day
            d2 = 30 if end.day == 31 else end.day

        return (360 * (end.year - start.year) + 30 * (end.month - start.month) + (d2 - d1)) / 360
