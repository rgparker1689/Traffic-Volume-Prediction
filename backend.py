import numpy as np
import pandas as pd

def predict(model, time_of_day, day_of_week, month, is_holiday, fahrenheit, cloud_density):
    time_day_dict = {'Early Morning': 1, 'Morning': 4, 'Late Morning': 3, 'Afternoon': 0, 'Evening': 2, 'Night': 5}
    day_week_dict = {'Sunday': 0, 'Monday': 1, 'Tuesday': 2, 'Wednesday': 3, 'Thursday': 4, 'Friday': 5, 'Saturday': 6}
    time_week_one_hot = list(np.zeros(6))
    day_week_one_hot = list(np.zeros(7))
    time_week_one_hot[time_day_dict[time_of_day]] = 1
    day_week_one_hot[day_week_dict[day_of_week]] = 1

    # Allowed temp range = -32 -> 120 degrees
    kelvin = (fahrenheit - 32) * (5/9) + 273.15
    is_winter = month in ['November', 'December', 'January', 'February']

    features = np.array([kelvin, cloud_density] + day_week_one_hot + [is_holiday] + time_week_one_hot + [is_winter])
    return int(np.round(model.predict(np.array([features]))[0]))

