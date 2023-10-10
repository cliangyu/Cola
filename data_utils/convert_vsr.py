import json

input_file_path = "converted_vsr_train_nonprocessed.json"
output_file_path = "converted_vsr_train.json"
# input_file_path='converted_vsr_test_nonprocessed.json'
# output_file_path='converted_vsr_test.json'

# Load the JSON file
with open(input_file_path, "r") as f:
    data = json.load(f)

# Replace the desired text in the JSON data
for item in data:
    item["question"] = item["question"].replace(
        "Can this image imply the following statement: ", 'Does this image describe "'
    )
    item["question"] = item["question"].replace("?", '" ?')

# Save the updated JSON data to the file
with open(output_file_path, "w") as f:
    json.dump(data, f)
