def is_leap_year(year):
    if (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0):
        return True
    else:
        return False

def find_anniversary(date_str):
    try:
        month, day, year = map(int, date_str.split('/'))
        if month < 1 or month > 12 or day < 1 or day > 31:
            print("Invalid date. Please enter a valid date (MM/DD/YYYY).")
            return

        if is_leap_year(year):
            next_anniversary = f"{month}/{day}/{year + 1}"
            print(f"Given Anniversary Year: Leap Year. Next Anniversary Date: {next_anniversary}")
        else:
            prev_anniversary = f"{month}/{day}/{year - 1}"
            print(f"Given Anniversary Year: Non Leap Year. Previous Anniversary Date: {prev_anniversary}")
    except ValueError:
        print("Invalid input. Please enter a valid date (MM/DD/YYYY).")

if __name__ == "__main__":
    anniversary_date = input("Enter Date (MM/DD/YYYY): ")
    find_anniversary(anniversary_date)
