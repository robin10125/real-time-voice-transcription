from datetime import datetime, timedelta

# Define the start and end dates for January 2024
start_date = datetime(2024, 3, 1)
end_date = datetime(2024, 4, 1)

# Open a text file to write the data
with open('March_2024_Work_Log.txt', 'w') as file:
    # Loop through each day of January 2024
    current_date = start_date
    while current_date != end_date:
        # Format the date as 'Month Day, Year'
        date_string = current_date.strftime('%A %B %d, %Y')
        
        # Write the formatted date and the specified information to the file
        file.write(f'{date_string}\n')
        file.write('Hours worked: \n')
        file.write('Object of work: \n')
        file.write('Comments: \n \n \n')
        
        # Move to the next day
        current_date += timedelta(days=1)
file.close()
print("The work log for Feb 2024 has been created.")