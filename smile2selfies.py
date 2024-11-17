import pandas as pd
import selfies as sf

# Load the CSV file
file_path = "All-DES.csv"  # Replace with your actual file path
data = pd.read_csv(file_path)

# Ensure the SMILES columns exist
if "Smiles 1" not in data.columns or "Smiles 2" not in data.columns:
    raise ValueError("The CSV file must contain 'Smile 1' and 'Smile 2' columns.")

# Convert SMILES to SELFIES and add new columns
data["SELFIES 1"] = data["Smiles 1"].apply(lambda x: sf.encoder(x))
data["SELFIES 2"] = data["Smiles 2"].apply(lambda x: sf.encoder(x))

# Save the updated DataFrame to a new CSV file
updated_file_path = "updated_All-DES.csv"
data.to_csv(updated_file_path, index=False)

print(f"Updated CSV file saved to: {updated_file_path}")

