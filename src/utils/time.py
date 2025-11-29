from typing import Optional


def parse_time(time_str: str) -> Optional[float]:
    """Parse a time string and return seconds as a float for moviepy.

    Args:
        time_str (str): Time string in format 'HH:MM:SS.mmm', 'MM:SS.mmm', 'SS.mmm', or 'SS'

    Returns:
        float: Time in seconds

    Examples:
        >>> parse_time('01:23:45.678')
        5025.678
        >>> parse_time('12:34.5')
        754.5
        >>> parse_time('45.123')
        45.123
        >>> parse_time('30')
        30.0
    """
    parts = time_str.strip().split(':')
    
    if len(parts) == 3:  # HH:MM:SS.mmm
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    elif len(parts) == 2:  # MM:SS.mmm
        minutes = int(parts[0])
        seconds = float(parts[1])
        return minutes * 60 + seconds
    elif len(parts) == 1:  # SS.mmm or SS
        return float(parts[0])
    else:
        return None

def strip_date(timestamp: str) -> str:
    """Remove the date part from a timestamp string, keeping only the time.

    Args:
        timestamp (str): Timestamp string in format 'YYYY-MM-DD HH:MM:SS.mmm'

    Returns:
        str: Time portion only 'HH:MM:SS.mmm'

    Examples:
        >>> strip_date('2019-09-26 00:00:49.099999905')
        '00:00:49.099999905'
        >>> strip_date('2020-01-01 12:34:56.789')
        '12:34:56.789'
    """
    return timestamp.split(' ', 1)[1] if ' ' in timestamp else timestamp
