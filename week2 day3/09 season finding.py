def get_season(month, day):
    seasons = {
        "Spring": ("March", 20),
        "Summer": ("June", 21),
        "Fall": ("September", 22),
        "Winter": ("December", 21)
    }

    month = month.capitalize()
    for season, (start_month, start_day) in seasons.items():
        if (month == start_month and day >= start_day) or (month != "December" and month != "March"):
            return season

    if month == "December" and day >= 21:
        return "Winter"
    if month == "March" and day < 20:
        return "Winter"

    return "Unknown"

def main():
    try:
        input_month = input("Enter the month: ")
        input_day = int(input("Enter the date: "))
        print(f"The season is currently {get_season(input_month, input_day)}")
    except ValueError:
        print("Invalid input. Please enter a valid month and date.")

if __name__ == "__main__":
    main()
