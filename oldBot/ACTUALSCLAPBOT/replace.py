import json

def process_file(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Process each line
    filtered_lines = []
    for line in lines:
        # Find the start of the JSON object
        json_start = line.find('{')
        if json_start != -1:
            # Parse the JSON object
            json_data = json.loads(line[json_start:])

            # Check if the "coin_name" field is not an empty string
            if json_data.get('coin_name', '').strip():
                filtered_lines.append(line)

    # Write the filtered lines back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(filtered_lines)

def remove_hour_of_day(file_path):
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Process each line
    modified_lines = []
    for line in lines:
        # Find the start of the JSON object
        json_start = line.find('{')
        if json_start != -1:
            # Parse the JSON object
            json_data = json.loads(line[json_start:])

            # Remove the "hour_of_day" field from the "market_context" object
            if 'market_context' in json_data:
                json_data['market_context'].pop('hour_of_day', None)

            # Convert the modified JSON object back to a string
            modified_line = line[:json_start] + json.dumps(json_data) + '\n'
            modified_lines.append(modified_line)

    # Write the modified lines back to the file
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(modified_lines)

# Usage
file_path = 'logs/final-analysis-2025-01-05.log'
# process_file(file_path)
remove_hour_of_day(file_path)