def print_the_high_temperatures_for_days_with_odd_indices(a_list_of_daily_temperatures):
    
    print("The high temperatures for days with odd indices: ");
    print({daily_temperature for indices,  daily_temperature in enumerate(a_list_of_daily_temperatures) if indices & 1 == 1 and daily_temperature == "high"});

a_list = ["daily temperatures"];

print_the_high_temperatures_for_days_with_odd_indices(a_list);
