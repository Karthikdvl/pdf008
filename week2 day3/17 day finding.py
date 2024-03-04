import datetime

def get_day_of_week(day, month, year):
    try:
        given_date = datetime.datetime(year, month, day)
        day_of_week = given_date.weekday()  # 0 (Monday) to 6 (Sunday)
        
        # Map day_of_week to the corresponding day name
        days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        day_name = days_of_week[day_of_week]
        
        return day_name
    except ValueError as e:
        return f"Error: {e}"

# Example usage
day = 31
month = 8
year = 2019
result = get_day_of_week(day, month, year)
print(f"Output: {result}")
