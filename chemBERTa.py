import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

# 1. Load Data
file_path = "./All-DES.csv"  # Replace with the actual path
data = pd.read_csv(file_path)

# Ensure correct column names
assert "Smiles 1" in data.columns and "Smiles 2" in data.columns and "molar ratio of component 1" in data.columns

# Extract columns
smiles1 = data["Smiles 1"].tolist()
smiles2 = data["Smiles 2"].tolist()
molar_ratio = data["molar ratio of component 1"].values
#temp=data["DES melting temperature"].values

# 2. Load ChemBERTa Model and Tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load ChemBERTa Model and Tokenizer
tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_60k")
model = AutoModel.from_pretrained("seyonec/PubChem10M_SMILES_BPE_60k")

# Function to get embeddings
def get_embeddings(smiles_list):
    tokens = tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**tokens)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # Average pooling

# Generate embeddings for Smile 1 and Smile 2
embeddings1 = get_embeddings(smiles1)
embeddings2 = get_embeddings(smiles2)

# 3. Combine Features
X = np.hstack([embeddings1, embeddings2, molar_ratio.reshape(-1, 1)])
y = data["DES melting temperature"].values  # Replace with the actual target column

min_val=np.min(y)
max_val=np.max(y)

y = (y - min_val) / (max_val - min_val)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7, random_state=42)

# Convert to PyTorch tensors
X_train, X_test = torch.tensor(X_train, dtype=torch.float32), torch.tensor(X_test, dtype=torch.float32)
y_train, y_test = torch.tensor(y_train, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

X_train=X_train.to(device)
X_test=X_test.to(device)

y_train=y_train.to(device)
y_test=y_test.to(device)

# 4. Define PyTorch Model
class RegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(RegressionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

model = RegressionModel(X_train.shape[1]).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# 5. Train the Model
time1=time.time()
for epoch in range(6000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train).squeeze()
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
time2=time.time()

# 6. Evaluate the Model
model.eval()
with torch.no_grad():
    predictions = model(X_test).squeeze()

# Convert to NumPy for metrics
predictions_np = predictions.cpu().numpy()
y_test_np = y_test.cpu().numpy()

mae = mean_absolute_error(y_test_np, predictions_np)
mae_tv= mae*(max_val-min_val)
rmse = np.sqrt(mean_squared_error(y_test_np, predictions_np))
r2 = r2_score(y_test_np, predictions_np)
time3=time2-time1

print(f"time: {time3}")
print(f"MAE: {mae}, MAE_true: {mae_tv}, RMSE: {rmse}, R²: {r2}")

