import numpy as np
import pandas as pd
from typing import TypeAlias
from dateutil.relativedelta import relativedelta as tdelta
from dateutil.easter import easter

Date: TypeAlias = pd.Timestamp

class Tenor:
    # Map to month units
    MONTH_MAP = dict(
        D = 1/30,
        W = 7/30,
        M = 1,
        Y = 12,
    )

    def __init__(self, tenor: str):
        self.tenor  = tenor.upper()
        self.months = self.__parse(self.tenor)
        self.tenor  = self.__month_to_str(self.months)
        self.amount = float(self.tenor[:-1])
        self.unit   = self.tenor[-1]

    def __parse(self, tenor: str) -> float:
        tenor_amount = float(tenor[:-1])
        tenor_unit   = tenor[-1]
        return tenor_amount * self.MONTH_MAP[tenor_unit]
    
    def __month_to_str(self, months: float) -> str:
        if months % 12 == 0:
            return f"{int(months // 12)}Y"
        
        elif months % 1 == 0:
            return f"{int(months)}M"
        
        else:
            weeks = months * 30 / 7
            if weeks % 1 == 0:
                return f"{int(weeks)}W"
            return f"{int(months * 30)}D"
        
    def __add__(self, other: Tenor) -> Tenor:
        return Tenor(self.__month_to_str(self.months + other.months))
    
    def __sub__(self, other: Tenor) -> Tenor:
        return Tenor(self.__month_to_str(self.months - other.months))
    
    def __mul__(self, amount: float) -> Tenor:
        return Tenor(self.__month_to_str(self.months * amount))
    
    def __truediv__(self, amount: float) -> Tenor:
        return Tenor(self.__month_to_str(self.months / amount))
    
    def __lt__(self, other):
        return self.months < other.months
    
    def __le__(self, other):
        return self.months <= other.months
    
    def __gt__(self, other):
        return self.months > other.months
    
    def __ge__(self, other):
        return self.months >= other.months
    
    def __eq__(self, other):
        return self.months == other.months
    
    def __ne__(self, other):
        return self.months != other.months
    
    def __hash__(self):
        return hash(self.months)
    
    def __repr__(self) -> str:
        return f"Tenor({self.__month_to_str(self.months)})"

def target_holidays(year_start: int, year_end: int=None):
    """
    TE: Trans-European Automated Real-time Gross settlement Express Transfer
    Standard EUR Interbank settlement calendar.
    """
    year_end = year_end or year_start
    holidays = []

    for year in range(year_start, year_end + 1):
        e = pd.to_datetime(easter(year))
        dates = sorted([
            pd.Timestamp(year=year, month=1, day=1),   # New Year's Day
            e - tdelta(days=2),                        # Good Friday
            e + tdelta(days=1),                        # Easter Monday
            pd.Timestamp(year=year, month=5, day=1),   # Labour Day
            pd.Timestamp(year=year, month=12, day=25), # Christmas
            pd.Timestamp(year=year, month=12, day=26), # Boxing Day
        ])
        holidays.extend(dates)

    return [np.datetime64(i.date()) for i in holidays]

class BusinessDay:
    def __init__(self, date, calendar=np.busdaycalendar(), roll="modifiedfollowing"):
        self.date = np.datetime64(pd.to_datetime(date).date())
        self.roll = roll
        self.calendar = calendar

    def shift(self, tenor, roll=None):
        roll = roll or self.roll
        tenor = Tenor(tenor) if isinstance(tenor, str) else tenor
        
        if tenor.unit in ("M", "Y"):
            date = np.busday_offset(self.date + tdelta(months=tenor.amount * (1 if tenor.unit == "M" else 12)), 0, roll, busdaycal=self.calendar)
        
        elif tenor.unit == "D":
            date = np.busday_offset(self.date, tenor.amount, roll, busdaycal=self.calendar)
        
        elif tenor.unit == "W":
            date = np.busday_offset(self.date, tenor.amount * 5, roll, busdaycal=self.calendar)

        return BusinessDay(date, self.calendar, roll)
    
    def __repr__(self):
        return f"BusinessDay(date={self.date}, roll={self.roll})"

def day_count_fraction(start_date, end_date, convention: str):
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)
    conv  = convention.upper()

    start = pd.DatetimeIndex(np.atleast_1d(start))
    end   = pd.DatetimeIndex(np.atleast_1d(end))

    if conv == "ACT/360":
        result = (end - start).days / 360

    elif conv == "ACT/365":
        result = (end - start).days / 365

    elif conv in ["30U/360", "30E/360"]:
        start_day, start_month, start_year = start.day.values, start.month.values, start.year.values
        end_day, end_month, end_year       = end.day.values,   end.month.values,   end.year.values

        if conv == "30U/360":
            # last calendar day of month, vectorized via days_in_month
            is_last_day_start = start_day == start.days_in_month.values
            is_last_day_end   = end_day   == end.days_in_month.values

            d1 = start_day.copy()
            d2 = end_day.copy()

            feb_start = start_month == 2

            # Case: Feb EOM start AND Feb EOM end -> d1=d2=30
            cond1 = feb_start & (end_month == 2) & is_last_day_start & is_last_day_end
            # Case: Feb EOM start only -> d1=30, d2=30 if end.day==31 else end.day
            cond2 = feb_start & is_last_day_start & ~cond1
            # Case: not Feb start, end.day==31 and start.day in [30,31] -> d1=d2=30
            cond3 = (~feb_start) & (end_day == 31) & np.isin(start_day, [30, 31])
            # Case: not Feb start, start.day==31 -> d1=30, d2=end.day (unless end.day==31, handled by cond3 priority)
            cond4 = (~feb_start) & (start_day == 31) & ~cond3

            d1 = np.select([cond1, cond2, cond3, cond4], [30, 30, 30, 30], default=d1)
            d2 = np.select(
                [cond1, cond2 & (end_day == 31), cond2 & (end_day != 31), cond3, cond4],
                [30,    30,                      end_day,                 30,    end_day],
                default=d2
            )

        else:  # 30E/360
            d1 = np.where(start_day == 31, 30, start_day)
            d2 = np.where(end_day   == 31, 30, end_day)

        result = (360 * (end_year - start_year) + 30 * (end_month - start_month) + (d2 - d1)) / 360

    result = np.asarray(result, dtype=float)
    return result if result.size > 1 else result.item()

# Helpers
def clean_date(date):
    return np.datetime64(pd.to_datetime(date).date())

def clean_tenor(tenor):
    return Tenor(tenor) if isinstance(tenor, str) else tenor


