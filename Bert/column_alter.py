import csv

# Replace with your CSV file paths
input_csv = "dataset.csv"  # Input CSV file
output_csv = "final_dataset.csv"  # Output CSV file

# Columns
attack_column = "Attack_type"  # Column containing attack types
label_column = "Attack_label"  # Column to modify with new labels

try:
    with open(input_csv, mode='r') as infile, open(output_csv, mode='w', newline='') as outfile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames  # Keep all columns intact
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Create label mapping
        label_map = {"Normal": 0}
        current_label = 1
        
        for row in reader:
            attack_type = row[attack_column]
            
            # Assign new labels
            if attack_type not in label_map:
                label_map[attack_type] = current_label
                current_label += 1
            
            # Update the Attack_label column
            row[label_column] = label_map[attack_type]
            writer.writerow(row)
        
        print("Label mapping:")
        for attack, label in label_map.items():
            print(f"{attack}: {label}")
except FileNotFoundError:
    print(f"Error: The file '{input_csv}' was not found.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
