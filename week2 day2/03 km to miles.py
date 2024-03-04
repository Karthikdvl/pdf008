def km_to_miles(km):

    conv_fac = 0.621371
    miles = km * conv_fac
    return miles

km = float(input("Enter value in kilometers: "))
converted_miles = km_to_miles(km)
print(f"{km:.2f} kilometers is {converted_miles:.2f} miles")
