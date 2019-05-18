import os

# Setting variables
one = 1
comment_char = "#"
space = " "
comment = "comment"
code_line = "line"
total_line = "line"
plural = "s"

# Initialising line counters
comment_counter = 0
code_counter = 0
line_counter = 0

# Getting file name from user
file_name = input("Please enter the name of file to checked: ")

# Checking if the file exists
file_exists = os.path.exists(file_name)

# Opens file if it exists
if file_exists:
    the_file = open(file_name)

    # Initalising for loop and total line counter
    for each_line in the_file:
        line_counter += one

        # Removing spaces if a line starts with a space
        if each_line.startswith(space):
            each_line.strip()

        # Counting comment lines
        if each_line.startswith(comment_char):
            comment_counter += one

        # Counting code lines
        elif each_line[0].isalnum():
           code_counter += one

    # Making "line" strings plural
    if comment_counter != one:
        comment = comment + plural
    if code_counter != one:
        code_line = code_line + plural
    if line_counter != one:
        total_line = total_line + plural

    print("{} contains {} {} and {} {} of code out of {} {} in total.".format(file_name, comment_counter, comment, code_counter, code_line, line_counter, total_line))
    
else:
    print("Sorry, I can't see a file called {}.".format(file_name))

the_file.close()