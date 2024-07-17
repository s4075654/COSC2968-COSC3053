import re

def a_function_that_returns_the_shipping_fee(a_distance):
    
    return 5000 if a_distance == 1 else (5000 if a_distance <= 10 else 3000) + a_function_that_returns_the_shipping_fee(a_distance - 1);

a_distance = re.search(r'\d+',  "first 10km");

print(f"The shipping fee: {a_function_that_returns_the_shipping_fee(int(a_distance.group()))}");
