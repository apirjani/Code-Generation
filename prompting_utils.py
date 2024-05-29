import os
import json
from collections import Counter
import re


def read_schema(schema_path):
    '''
    Read the .schema file
    '''
    with open(schema_path, 'r') as file:
        schema = json.load(file)
    
    ents = schema['ents']
    links = schema['links']
    prompt_addition = "Database Schema Information:\n"
    prompt_addition += "Entities and their attributes:\n"
    for entity_category, details in ents.items():
        prompt_addition += f"\n{entity_category} has attributes: "
        # Iterate over each entity in the category
        for entity in details.keys():
            # Add information about each entity to the prompt
            prompt_addition += f"{entity}, "
        prompt_addition = prompt_addition.rstrip(', ')
    prompt_addition += "\n\n"

    #INSERT CODE HERE TO INCLUDE LINKS IN PROMPT
    prompt_addition += "\nRelationships between entities:\n"
    for source_entity, relations in links.items():
        if relations:  # Check if the entity has any links
            for target_entity, link_field in relations.items():
                prompt_addition += f"{source_entity} is connected to {target_entity} by {link_field}\n"
        else:
            prompt_addition += f"{source_entity} has no direct links.\n"

    return prompt_addition

def extract_sql_query(response):
    '''
    Extract the SQL query from the model's response
    '''
    # Define the pattern to find the SQL query block
    pattern = r"SQL:\s*```sql\s*(.*?)\s*```<eos>"
    
    # Search for the pattern
    match = re.search(pattern, response, re.DOTALL)
    
    # If a match is found, remove newlines, then return the SQL query, otherwise return an empty string
    if match:
        # Remove newlines and return the cleaned SQL query
        return match.group(1).replace('\n', ' ')
    else:
        return ""

def save_logs(output_path, sql_em, record_em, record_f1, error_msgs):
    '''
    Save the logs of the experiment to files.
    You can change the format as needed.
    '''
    with open(output_path, "w") as f:
        f.write(f"SQL EM: {sql_em}\nRecord EM: {record_em}\nRecord F1: {record_f1}\nModel Error Messages: {error_msgs}\n")

def compute_freq_map(data):
    '''
    Compute the frequency map for the data
    '''
    if isinstance(data, str):
        # If the data is a single sentence, process it directly
        return list(Counter(data.split()).values())
    elif isinstance(data, list):
        # If the data is a list of sentences, process each sentence with a list comprehension
        return [list(Counter(sentence.split()).values()) for sentence in data]
    else:
        raise ValueError("Input must be a string or a list of strings.")

