import requests
from bs4 import BeautifulSoup

# URL of the page to scrape
url = 'https://github.com/trinib/Linux-Bash-Commands/blob/main/README.md'

# Make a request to get the page content
response = requests.get(url)

# Use BeautifulSoup to parse the HTML content
soup = BeautifulSoup(response.text, 'html.parser')

# Find all tables on the page (assuming each table contains a list of commands)
tables = soup.find_all('table')

# Iterate over each table and extract commands and descriptions
commands = []

break_loop=False
for table in tables:
    if break_loop:
        break
    rows = table.find_all('tr')[1:]  # Skip the header row
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 2:  # Check if there are enough columns
            command = cols[0].text.strip()
            description = cols[1].text.strip()
            commands.append((command, description))
            #print(f'Command: {command}, Description: {description}')
            if command=="znew": #the last command
                break_loop=True
                break
# Optionally, write the commands to a file
with open('/home/aimaster/Desktop/STUDIA/NLP/Project/commands_desription.txt', 'w') as file:
    for command, description in commands:
        file.write(f'{command}:{description}\n')

print(commands)
command_set = {command for command, _ in commands}
print(command_set)
