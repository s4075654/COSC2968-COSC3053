def determine_grade(score):
    if 80 <= score <= 100:
        
        return("HD");
        
    elif 70 <= score < 80:
        
        return("DI");
        
    elif 60 <= score < 70:
        
        return("CR");
        
    elif 50 <= score < 60:
        
        return("PA");
        
    elif 0 <= score < 50:
        
        return("NN");
        
print(f"Determine grade: {determine_grade(int(input("Input score:  ")))}");
