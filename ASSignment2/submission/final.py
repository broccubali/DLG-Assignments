# DLG Assignment 2

### Submitted by:
- Namita Achyuthan: PES1UG22AM100
- Shusrith S: PES1UG22AM155
- Siddhi Zanwar: PES1UG22AM161

##### * Detailed assignment report available [here](https://docs.google.com/document/d/1etideCaU4pXN6Vzvw4E2q9FvwdNfgt0YcsWTcFIjKDw/edit?usp=sharing)
"""

! pip install torch_geometric

import pandas as pd
import json
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch_geometric.nn import HypergraphConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tqdm import tqdm
import numpy as np
import networkx as nx
import torch
import math
from copy import deepcopy
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

"""## Dataset Overview and Preprocessing

We began our exploration by examining the four provided files: `train_stock_data.csv`, `validation_stock_data.csv`, `hyperedges.json`, and `bling_test_cases.json`. The training and validation datasets contained **20,160** and **2,000** entries, respectively. The `hyperedges.json` file described **eight hyperedges**, and there were **120 test cases** in `bling_test_cases.json`.

Upon inspecting the training dataset, we found **20 unique tickers**, each represented by five attributes: **Open, Close, Volume, High, and Low**. This resulted in a sparse structure with **102 columns**. We decoded the hypergraph and assigned numerical identifiers to each ticker for easier referencing.

We encountered an issue with the **date format**, which was inconsistent. We resolved this by applying Pandas' `to_datetime` function and sorting by ticker and date. To enhance interpretability, we transformed the data into a more conventional time-series format with clear columns: **Date, Ticker, Open, High, Low, Close, and Volume**.

Visualizing the data revealed significant variations across companies, prompting us to consider **normalization techniques**. We tested three primary scaling methods:  
- **Standard Scaler**: Removes the mean and scales to unit variance.
- **Min-Max Scaler**: Scales features to a given range, typically between 0 and 1.
- **Robust Scaler**: Uses the median and interquartile range, making it less sensitive to outliers.



We ended up choosing the robust scaler since it gave us the least MAPE upon model evaluation.


---

At this stage, we had completed the foundational preprocessing steps, marking the point where our approaches began to diverge for downstream experimentation.
"""

train_data = pd.read_csv('/content/DLG-Assignments/ASSignment2/train_stock_data.csv')
train_data['Date'] = pd.to_datetime(train_data['Date'], format='%Y-%m-%d') # date is a string make it a datetime obj
train_data.sort_values(['Ticker', 'Date'], inplace=True) # sort the ticker by date

validation_data = pd.read_csv('/content/DLG-Assignments/ASSignment2/validation_stock_data.csv')
validation_data['Date'] = pd.to_datetime(validation_data['Date'], format='%Y-%m-%d')
validation_data.sort_values(['Ticker', 'Date'], inplace=True)

with open('/content/DLG-Assignments/ASSignment2/hyperedges.json', 'r') as f:
    hyperedges = json.load(f) # load the hyperedge defintions

with open('/content/DLG-Assignments/ASSignment2/blind_test_cases.json', 'r') as f:
    test_cases = json.load(f)

print(f"Training data shape: {train_data.shape}")
print(f"Validation data shape: {validation_data.shape}")
print(f"Number of hyperedges: {len(hyperedges)}")
print(f"Number of test cases: {len(test_cases)}")
print("Available tickers in training data:", train_data['Ticker'].unique())

l = []  # Initialize an empty list to store preprocessed rows

tickers = train_data['Ticker'].unique()  # Get a list of unique tickers

for ticker in tickers:
    data = train_data[train_data["Ticker"] == ticker]  # Filter data for the current ticker

    valid_cols = []  # List to store columns without missing values

    for i in data.columns:
        if data[i].isna().sum() == 0:  # Check if the column has no NaN values
            valid_cols.append(i)  # Add it to the valid columns list

    a = data[valid_cols].values  # Extract only the valid columns as a numpy array

    for i in a:
        l.append(i)  # Add each row to the final preprocessed list

# Convert the list of clean rows into a DataFrame with proper column names
df = pd.DataFrame(l, columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

df['Return'] = df.groupby('Ticker')['Close'].pct_change() # percentage change in closing price (i am finance bro now)
df['Volatility'] = df.groupby('Ticker')['Return'].rolling(window=30).std().reset_index(0, drop=True) # 30 day window for rolling volatility
df['Momentum'] = df.groupby('Ticker')['Close'].pct_change(periods=5) # 5 day momentum (price change)
df['Moving_Avg'] = df.groupby('Ticker')['Close'].rolling(window=30).mean().reset_index(0, drop=True)
df.fillna(0, inplace=True) # if nan make 0

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Volatility', 'Momentum', 'Moving_Avg']

# scaler = StandardScaler()
# scaler = MinMaxScaler() # loss was 52354522137231.3594 gg
scaler = RobustScaler() # Val Loss: 0.1693
feature_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# ticker to index mapping
sorted_tickers = sorted(tickers) # first sort alphabetically
ticker_to_idx = {ticker: idx for idx, ticker in enumerate(sorted_tickers)} # dict mapping
print("Ticker to index mapping:", ticker_to_idx)

"""## HGNN Model

### What is a HGNN?  
A Hypergraph Neural Network (HGNN) is a graph neural network designed for hypergraphs, where a single hyperedge can connect multiple nodes (unlike traditional graphs that connect only two). This makes HGNNs ideal for modeling higher-order relationships. In our context, stocks are nodes, and industries (like Tech and Healthcare) are hyperedges, connecting all stocks within a sector to capture sector-wise dependencies.

### Why HGNN?  
We chose HGNN because:  
- Sector Influence: Stocks in the same industry often respond similarly to economic or political trends.  
- Beyond Pairwise Modeling: Standard GNNs capture only pairwise relationships, while HGNNs model multi-node interactions via shared hyperedges.  
- Rich Contextual Learning: HGNNs excel at capturing sector-level co-movements and dependencies, essential for financial prediction tasks.  

The incidence matrix ensures industries share information within their hyperedges, leveraging HGNN's strength in higher-order reasoning.

---

### Implementation Overview  
#### Data Structure Preparation:  
- Nodes: Represent individual stocks.  
- Hyperedges: Connect stocks within the same industry to model group dynamics.  
- An incidence matrix encodes multi-node connectivity using `hyperedge_index`.  

#### Graph Snapshots (Temporal View):  

For each trading date:  
- Node features: Stock indicators like returns and volatility.  
- Node targets: Actual prices (Open, High, Low, Close, Volume).  

These snapshots are wrapped as `torch_geometric.data.Data` objects with consistent `hyperedge_index`.

#### Model Architecture:  
The model uses two stacked Hypergraph Convolutional Layers:  
1. The first layer transforms features into rich representations using ELU activation.  
2. The second layer outputs continuous values for the five price attributes.

#### Training Setup:  
- Loss function: Mean Squared Error (MSE).  
- Optimizer: Adam with learning rate scheduling for steady convergence.  
- Metrics: Root Mean Squared Error (RMSE) and Mean Absolute Percentage Error (MAPE).

### Temporal Handling and Attention Mechanisms  
The model treats time as independent static graph snapshots rather than explicitly modeling temporal dependencies (like RNNs or Transformers). Each snapshot captures day-wise spatial dependencies among stocks via hyperedges. Information is aggregated using uniform weights without learnable attention mechanisms over neighbors or hyperedges.
"""

# node to hyperedge maps (incidence matrix)
num_nodes = len(sorted_tickers) # number of stocks (nodes)
num_hyperedges = len(hyperedges) # number of hyperedges (industry)

# Create node-to-hyperedge incidence matrix
# For HypergraphConv we need:
# 1. hyperedge_index: list of [node_idx, hyperedge_idx] pairs
# 2. hyperedge_weight: weight for each hyperedge
# gpt helped here big time

hyperedge_index = [] # pairs of node_index, hyperedge_index
for he_idx, (he_name, he_tickers) in enumerate(hyperedges.items()):
    for ticker in he_tickers:
        node_idx = ticker_to_idx[ticker] # get node index frmo tivker
        hyperedge_index.append([node_idx, he_idx]) # add connection between node and hyperedge

hyperedge_index = torch.tensor(hyperedge_index, dtype=torch.long).t() # transpose for 2*N tensor
hyperedge_weight = torch.ones(num_hyperedges, dtype=torch.float) # all hyperedges get weight 1

# double check
print(f"Hyperedge index shape: {hyperedge_index.shape}")
print(f"Hyperedge weight shape: {hyperedge_weight.shape}")

def get_features_for_date(df, date, tickers, feature_cols):
    day_df = df[df['Date'] == date].set_index('Ticker') # filter data fr given date
    day_df = day_df.reindex(tickers)
    x = day_df[feature_cols].fillna(0).values
    y = day_df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0).values # target (output)
    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

# make each date a snapshot of the graph
dates = sorted(df['Date'].unique())
graph_snapshots = []
for date in dates:
    x_t, y_t = get_features_for_date(df, date, sorted_tickers, features)
    graph_snapshots.append((x_t, y_t))

# graph objects for pytorch geo
train_graphs = []
for x_t, y_t in graph_snapshots:
    # DATA OBJECT with node features, hyperedge
    graph = Data(x=x_t, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_weight, y=y_t)
    train_graphs.append(graph)

l_val = []
val_tickers = sorted(validation_data['Ticker'].unique())
for ticker in val_tickers:
    data = validation_data[validation_data["Ticker"] == ticker] # basically since 8 in the val and 20 in the train, you match the ticker in val to the one in train
    # nice
    valid_cols = []
    for i in data.columns:
        if data[i].isna().sum() == 0:
            valid_cols.append(i)
    a = data[valid_cols].values
    for i in a:
        l_val.append(i)
df_val = pd.DataFrame(l_val, columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])

df_val['Return'] = df_val.groupby('Ticker')['Close'].pct_change()
df_val['Volatility'] = df_val.groupby('Ticker')['Return'].rolling(window=30).std().reset_index(0, drop=True)
df_val['Momentum'] = df_val.groupby('Ticker')['Close'].pct_change(periods=5)
df_val['Moving_Avg'] = df_val.groupby('Ticker')['Close'].rolling(window=30).mean().reset_index(0, drop=True)
df_val.fillna(0, inplace=True)

# normalize
df_val[feature_cols] = scaler.transform(df_val[feature_cols])

val_dates = sorted(df_val['Date'].unique())
val_graph_snapshots = []
for date in val_dates:
    x_t, y_t = get_features_for_date(df_val, date, sorted_tickers, features)
    val_graph_snapshots.append((x_t, y_t))

val_graphs = []
for x_t, y_t in val_graph_snapshots:
    graph = Data(x=x_t, hyperedge_index=hyperedge_index, hyperedge_attr=hyperedge_weight, y=y_t)
    val_graphs.append(graph)

class StockHGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=5):
        super().__init__() # parent class
        # first hypergraph convolution layer: input_features -> hidden_features
        self.hgconv1 = HypergraphConv(in_channels, hidden_channels)

        # second hypergraph convolution layer: hidden_features -> output_features
        self.hgconv2 = HypergraphConv(hidden_channels, out_channels)

    def forward(self, x, hyperedge_index, hyperedge_weight=None):
        # apply first layer with ELU activation function
        x = F.elu(self.hgconv1(x, hyperedge_index, hyperedge_weight))

        # next is applying second layer (no activation for regression output)
        x = self.hgconv2(x, hyperedge_index, hyperedge_weight)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = StockHGNN(in_channels=len(features), hidden_channels=32, out_channels=5).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
loss_fn = torch.nn.MSELoss() # cuz regressuion
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

train_loader = DataLoader(train_graphs, batch_size=1, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=1, shuffle=False)

hyperedge_index = hyperedge_index.to(device)
hyperedge_weight = hyperedge_weight.to(device)

epochs = 50
train_losses, val_losses = [], []

for epoch in range(epochs):
    # Training
    model.train() # set to training mode
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False):
        batch = batch.to(device) # bacth data to devoce
        optimizer.zero_grad() # clear prev gradients
        out = model(batch.x, batch.hyperedge_index, batch.hyperedge_attr) # forward pass
        loss = loss_fn(out, batch.y) # get the loss
        loss.backward() # back prop
        optimizer.step() # update weights
        train_loss += loss.item() # accumulate loss
    train_loss /= len(train_loader) # avg loss
    train_losses.append(train_loss) # store for when we plot

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation", leave=False):
            batch = batch.to(device)
            out = model(batch.x, batch.hyperedge_index, batch.hyperedge_attr)
            loss = loss_fn(out, batch.y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

model.eval() # now set model to evaluation mode
preds_list, actuals_list = [], []
with torch.no_grad(): # no gradient evaluation
    for batch in tqdm(val_loader, desc="Making Predictions", leave=False):
        batch = batch.to(device)
        preds = model(batch.x, batch.hyperedge_index, batch.hyperedge_attr)
        preds_list.append(preds.cpu()) # store preds
        actuals_list.append(batch.y.cpu()) # store actual values

preds = torch.cat(preds_list, dim=0)
actuals = torch.cat(actuals_list, dim=0)

# kick out zero values
mask = actuals.abs().sum(dim=1) > 0  # mask for non-zero rows
preds = preds[mask].numpy()  # Apply mask  like we all do in life
actuals = actuals[mask].numpy()

rmse = np.sqrt(mean_squared_error(actuals, preds))
mape = np.mean(np.abs((actuals - preds) / np.clip(np.abs(actuals), 1e-10, None))) * 100

pred_df = pd.DataFrame(preds, columns=["Open", "High", "Low", "Close", "Volume"])
actual_df = pd.DataFrame(actuals, columns=["Open", "High", "Low", "Close", "Volume"])

# scale back up now
#copies for the inverse-transformed data
pred_df_og = pred_df.copy()
actual_df_og = actual_df.copy()

pred_df_og[feature_cols] = scaler.inverse_transform(pred_df[feature_cols])
actual_df_og[feature_cols] = scaler.inverse_transform(actual_df[feature_cols])

og_rmse = np.sqrt(mean_squared_error(actual_df_og, pred_df_og))
og_mape = np.mean(np.abs((actual_df_og - pred_df_og) / np.clip(np.abs(actual_df_og), 1e-10, None))) * 100
print(f"Original Scale RMSE: {og_rmse:.4f}")
print(f"Original Scale MAPE: {og_mape:.2f}%")

column_names = pred_df_og.columns.tolist()
fig, axs = plt.subplots(1, len(column_names), figsize=(20, 4), sharex=True)
for i, col in enumerate(column_names):
    axs[i].plot(actual_df_og[col], label="Actual", color="blue")
    axs[i].plot(pred_df_og[col], label="Predicted", color="red", linestyle='dashed')
    axs[i].set_title(f"{col}")
    axs[i].legend()
    axs[i].grid(True)

fig.suptitle("Actual vs Predicted - All Features (HGNN Model)", fontsize=16)
plt.tight_layout()
plt.show()

"""## GAT

### What is a GAT?  
A Graph Attention Network (GAT) is a type of neural network designed for graph-structured data. It improves upon traditional Graph Neural Networks (GNNs) by introducing an attention mechanism, which learns to weigh the importance of different nodes (or neighbors) during message-passing. Unlike standard GNNs that treat all neighbors equally, GATs focus more on relevant connections, enabling better aggregation of node information.

### Why GAT?  
We chose GAT for our stock hypergraph because:  
- Heterogeneous Relationships: In financial graphs, the influence of one stock on another varies over time. GAT’s attention mechanism adapts to these dynamic relationships.  
- Improved Accuracy: By emphasizing stronger connections, GAT captures nuanced dependencies, enhancing prediction performance.  

---

### Implementation Overview  
#### Model Architecture:  
- Two stacked GATConv layers:  
  - The first layer applies attention-based message passing to learn node embeddings.  
  - The second layer refines these embeddings.  
- Activation: ReLU is applied after the second layer.  
- Global Mean Pooling: Aggregates node-level information into a graph-level representation.  
- Fully Connected Layer: Performs classification based on pooled embeddings.

#### Temporal Handling and Attention Mechanism:  
- Self-Attention: Nodes learn to weigh their neighbors differently using attention coefficients, enabling localized focus on informative connections.  
- Unlike temporal models (e.g., RNNs), GAT operates on static graph snapshots, learning spatial relationships independently for each snapshot.

This setup allows the model to capture both local node-wise dependencies and holistic graph-level dynamics without introducing complex temporal layers.
"""

hyperedge_indices = []
for he_name, ticker in hyperedges.items():
    indices = [ticker_to_idx[t] for t in ticker]
    print(f"Hyperedge '{he_name}': {ticker} -> {indices}")
    hyperedge_indices.append(indices)

hyperedge_indices

# clique representation
adj_matrix = [[0 for i in range(20)] for j in range(20)]
for hyperedge in hyperedge_indices:
    for i in range(len(hyperedge)):
        for j in range(i+1, len(hyperedge)):
            adj_matrix[hyperedge[i]][hyperedge[j]] = 1
            adj_matrix[hyperedge[j]][hyperedge[i]] = 1
adj_matrix = np.array(adj_matrix)
adj_matrix

edge_index = torch.tensor(np.array(np.nonzero(adj_matrix)), dtype=torch.long)
edge_index

def get_features_for_date(df, date, tickers, feature_cols):
    day_df = df[df['Date'] == date].set_index('Ticker')
    day_df = day_df.reindex(tickers)
    x = day_df[feature_cols].fillna(0).values
    y = day_df[['Open', 'High', 'Low', 'Close', 'Volume']].fillna(0).values
    return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

target_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
dates = sorted(df['Date'].unique())
graph_snapshots = []
tickers = sorted(df['Ticker'].unique())
for date in dates:
    x_t, y_t = get_features_for_date(df, date, tickers, target_cols)
    graph_snapshots.append((x_t, y_t))

graphs = []

for x_t, y_t in graph_snapshots:
    graph = Data(x=x_t, edge_index=edge_index, y=y_t)
    graphs.append(graph)

graphs[0]

val_data = pd.read_csv('/content/DLG-Assignments/ASSignment2/validation_stock_data.csv')
val_data['Date'] = pd.to_datetime(val_data['Date'], format='%Y-%m-%d')
val_data.sort_values(['Ticker', 'Date'], inplace=True)
l = []
val_tickers = sorted(val_data['Ticker'].unique())
for ticker in val_tickers:
    data = val_data[val_data["Ticker"] == ticker]
    valid_cols = []
    for i in data.columns:
        if data[i].isna().sum() == 0:
            valid_cols.append(i)
    a = data[valid_cols].values
    for i in a:
        l.append(i)
df1 = pd.DataFrame(l, columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
df1

df2 = df1[feature_cols].copy()
df1[feature_cols] = scaler.transform(df1[feature_cols])

aapl = df[df["Ticker"] == "AAPL"]
plt.figure(figsize=(12, 6))
plt.plot(aapl.index, aapl['Close'], label='Close Price')
plt.title('AAPL Close Price (2019-2022)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
# scaled btw

dates = sorted(df1['Date'].unique())
val_graph_snapshots = []

for date in dates:
    x_t, y_t = get_features_for_date(df1, date, tickers, target_cols)
    val_graph_snapshots.append((x_t, y_t))

val_graphs = []

for x_t, y_t in val_graph_snapshots:
    graph = Data(x=x_t, edge_index=edge_index, y=y_t)
    val_graphs.append(graph)
val_graphs[0]

from torch_geometric.nn import GATConv
import torch.nn as nn
import torch.nn.functional as F

class StockGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels=5, heads=32, dropout=0.3):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(hidden_channels * heads)

        self.gat2 = GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(hidden_channels * heads)

        self.gat3 = GATConv(hidden_channels * heads, hidden_channels, heads=1)
        self.norm3 = nn.LayerNorm(hidden_channels)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, out_channels)
        )
        self.scale_factor = nn.Parameter(torch.ones(out_channels))
        self.scale_bias = nn.Parameter(torch.zeros(out_channels))


    def forward(self, x, edge_index):
        x = F.elu(self.gat1(x, edge_index))
        x = self.norm1(x)

        x = F.elu(self.gat2(x, edge_index))
        x = self.norm2(x)

        x = F.elu(self.gat3(x, edge_index))
        x = self.norm3(x)

        out = self.regressor(x)
        return out * self.scale_factor + self.scale_bias

model = StockGAT(
    in_channels=len(target_cols),
    hidden_channels=128,
    out_channels=5,
    heads=32,
    dropout=0.3
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)
loss_fn = torch.nn.MSELoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-5)

train_loader = DataLoader(graphs, batch_size=64, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
edge_index = edge_index.to(device)
epochs = 100
train_losses, val_losses = [], []

for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch in tqdm(train_loader, desc=f"[Epoch {epoch+1}] Training", leave=False):
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"[Epoch {epoch+1}] Validation", leave=False):
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index)
            loss = loss_fn(out, batch.y)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)
    scheduler.step(val_loss)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

model.eval()
preds_list, actuals_list = [], []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Making Predictions", leave=False):
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index)

        preds_list.append(preds.cpu())
        actuals_list.append(batch.y.cpu())

import numpy as np

preds = torch.cat(preds_list, dim=0)
actuals = torch.cat(actuals_list, dim=0)

mask = actuals.abs().sum(dim=1) > 0

preds = preds[mask].numpy()
actuals = actuals[mask].numpy()

from sklearn.metrics import mean_absolute_error, mean_squared_error

preds = scaler.inverse_transform(preds)
actuals = scaler.inverse_transform(actuals)


mae = mean_absolute_error(actuals, preds)
rmse = np.sqrt(mean_squared_error(actuals, preds))
mape = np.mean(np.abs((actuals - preds) / actuals)) * 100
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

fig, axs = plt.subplots(1, 5, figsize=(20, 4), sharex=True)

column_names = ["Open", "High", "Low", "Close", "Volume"]
for i in range(5):
    axs[i].plot(actuals[:, i], label="Actual", color="blue")
    axs[i].plot(preds[:, i], label="Predicted", color="red", linestyle='dashed')
    axs[i].set_title(column_names[i])
    axs[i].legend()
fig.suptitle("Actual vs Predicted - All Features", fontsize=16)
plt.tight_layout()
plt.show()

"""## Classic ML

### What Models Were Used?  
We used a RandomForestRegressor from scikit-learn for classic regression modeling.  
This model was chosen to predict the target variable using tabular features that were engineered from the hypergraph structure.

### Why RandomForestRegressor?  
We chose RandomForest because:  
- Robust to Overfitting: The ensemble nature of the model helps avoid overfitting on the training data.  
- Handles Mixed Features Well: Performs well with heterogeneous inputs like sector info and moving averages.  
- No Need for Graph-Specific Pipelines: Allows seamless use of features from the hypergraph in a standard ML workflow.  
- Effective for Tabular Data: Especially suited for structured datasets without needing neural architectures.

---

### Implementation Overview  
#### Feature Engineering:  
- Extracted features from the hypergraph such as:  
  - Moving Averages over time windows  
  - Sector Averages based on hyperedge definitions  
- Combined these features into a consolidated dataframe  

#### Training Process:  
- The final dataframe was split into training and testing sets  
- A `RandomForestRegressor` model was initialized and trained on the training set  
- Predictions were generated on the test set  

#### Model Architecture:  
- Ensemble-based Tree Model: Combines multiple decision trees  
- Non-Linear Learning: Captures complex patterns in the data through aggregated decisions from many trees  


### Temporal Handling and Attention Mechanism  
- No Attention Mechanism: This classical model doesn’t use attention layers  
- Temporal Awareness via Feature Engineering:  
  - Used manually engineered features like moving averages to incorporate historical patterns  
  - Avoided explicit temporal models (like RNNs and Transformers) to keep the pipeline lightweight  

This approach gave us a strong baseline using interpretable, classical methods without the complexity of deep learning.
"""

ticker_to_sector = {ticker: sector for sector, tickers in hyperedges.items() for ticker in tickers}

train_data = pd.read_csv('/content/DLG-Assignments/ASSignment2/train_stock_data.csv')
train_data['Date'] = pd.to_datetime(train_data['Date'])
train_data.sort_values(['Ticker', 'Date'], inplace=True)

clean_rows = [] # clean rows of data
for ticker in train_data['Ticker'].unique(): # go voer every unique ticker
    data = train_data[train_data['Ticker'] == ticker]
    valid_cols = data.columns[data.isna().sum() == 0]
    clean_rows.extend(data[valid_cols].values)

train_df = pd.DataFrame(clean_rows, columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
train_df["Sector"] = train_df["Ticker"].map(ticker_to_sector) # ticker mapped to sector as a way to work with the hyperedges. we're considering it an attribute of each ticker itself
train_df = train_df.sort_values(["Ticker", "Date"]) # sort by ticker and date

# some feature engineeringl ike before
train_df["Return_1d"] = train_df.groupby("Ticker")["Open"].pct_change()
train_df["MA_5"] = train_df.groupby("Ticker")["Open"].transform(lambda x: x.rolling(5).mean())
train_df["MA_10"] = train_df.groupby("Ticker")["Open"].transform(lambda x: x.rolling(10).mean())
train_df["Volatility_5"] = train_df.groupby("Ticker")["Open"].transform(lambda x: x.rolling(5).std())

sector_return = train_df.groupby(["Date", "Sector"])["Return_1d"].mean().reset_index() # average one day return for each sector
sector_return.rename(columns={"Return_1d": "Sector_Avg_Return"}, inplace=True)
train_df = pd.merge(train_df, sector_return, on=["Date", "Sector"], how="left") # here we keep all the original rows from train_df. The merge keys we're using are date and sector

# now we create target columns for the next day's values by shifting the data forward
train_df["Target_Open"] = train_df.groupby("Ticker")["Open"].shift(-1)
train_df["Target_High"] = train_df.groupby("Ticker")["High"].shift(-1)
train_df["Target_Low"] = train_df.groupby("Ticker")["Low"].shift(-1)
train_df["Target_Close"] = train_df.groupby("Ticker")["Close"].shift(-1)
train_df["Target_Volume"] = train_df.groupby("Ticker")["Volume"].shift(-1)

val_data = pd.read_csv('/content/DLG-Assignments/ASSignment2/validation_stock_data.csv')
val_data['Date'] = pd.to_datetime(val_data['Date'])
val_data.sort_values(['Ticker', 'Date'], inplace=True)

# now we do with val data whatever we did with the train data
val_rows = []
for ticker in val_data['Ticker'].unique():
    data = val_data[val_data['Ticker'] == ticker]
    valid_cols = data.columns[data.isna().sum() == 0]
    val_rows.extend(data[valid_cols].values)

val_df = pd.DataFrame(val_rows, columns=["Date", "Ticker", "Open", "High", "Low", "Close", "Volume"])
val_df["Sector"] = val_df["Ticker"].map(ticker_to_sector)
val_df = val_df.sort_values(["Ticker", "Date"])

val_df["Return_1d"] = val_df.groupby("Ticker")["Open"].pct_change()
val_df["MA_5"] = val_df.groupby("Ticker")["Open"].transform(lambda x: x.rolling(5).mean())
val_df["MA_10"] = val_df.groupby("Ticker")["Open"].transform(lambda x: x.rolling(10).mean())
val_df["Volatility_5"] = val_df.groupby("Ticker")["Open"].transform(lambda x: x.rolling(5).std())

combined = pd.concat([train_df, val_df])
combined["Date"] = pd.to_datetime(combined["Date"])
sector_return = combined.groupby(["Date", "Sector"])["Return_1d"].mean().reset_index()
sector_return.rename(columns={"Return_1d": "Sector_Avg_Return"}, inplace=True)
val_df = pd.merge(val_df, sector_return, on=["Date", "Sector"], how="left")

# Validation targets. everything after this is the same ML workflow of getting the
# targets, the features, and then model.fit()
val_df["Target_Open"] = val_df.groupby("Ticker")["Open"].shift(-1)
val_df["Target_High"] = val_df.groupby("Ticker")["High"].shift(-1)
val_df["Target_Low"] = val_df.groupby("Ticker")["Low"].shift(-1)
val_df["Target_Close"] = val_df.groupby("Ticker")["Close"].shift(-1)
val_df["Target_Volume"] = val_df.groupby("Ticker")["Volume"].shift(-1)

features = ["Return_1d", "MA_5", "MA_10", "Volatility_5", "Sector_Avg_Return"]
targets = ["Target_Open", "Target_High", "Target_Low", "Target_Close", "Target_Volume"]

train_df_model = train_df.dropna(subset=features + targets)
val_df_model = val_df.dropna(subset=features + targets)

X_train = train_df_model[features]
y_train = train_df_model[targets]

X_val = val_df_model[features]
y_val = val_df_model[targets]

base_model = RandomForestRegressor(n_estimators=100, random_state=42)
multi_model = MultiOutputRegressor(base_model)
multi_model.fit(X_train, y_train)

def evaluate_model(X, y_true, dataset_name=""):
    y_pred = multi_model.predict(X)
    print(f"\nEvaluation on {dataset_name}:")
    for i, target in enumerate(targets):
        rmse = mean_squared_error(y_true.iloc[:, i], y_pred[:, i])
        mape = mean_absolute_percentage_error(y_true.iloc[:, i], y_pred[:, i])
        print(f"{target}: RMSE = {rmse:.4f}, MAPE = {mape:.4f}")

evaluate_model(X_train, y_train, "TRAINING") # train
evaluate_model(X_val, y_val, "VALIDATION") # val

def predict_test_cases(test_cases, full_data, model, feature_columns):
    # we don't want to mutate the original list
    updated_cases = deepcopy(test_cases)

    # just making sure it's a normal datetime object
    full_data["Date"] = pd.to_datetime(full_data["Date"])

    for case in updated_cases:
        ticker = case["ticker"]
        target_date = pd.to_datetime(case["date"])

        # Filter past data for that specific ticker
        ticker_data = full_data[full_data["Ticker"] == ticker].copy()
        ticker_data = ticker_data[ticker_data["Date"] < target_date].sort_values("Date")

        if ticker_data.empty:
            print(f"Not enough data for {ticker} before {target_date}")
            continue

        # we basically replicate our full training feature engineering here
        ticker_data["Return_1d"] = ticker_data["Open"].pct_change()
        ticker_data["MA_5"] = ticker_data["Open"].rolling(5).mean()
        ticker_data["MA_10"] = ticker_data["Open"].rolling(10).mean()
        ticker_data["Volatility_5"] = ticker_data["Open"].rolling(5).std()

        # Drop the nans cuz they'll be at the beginning due to rolling windows
        ticker_data = ticker_data.dropna(subset=feature_columns)

        if ticker_data.empty:
            print(f"Not enough data with features for {ticker} before {target_date}")
            continue

        # last row of features as input
        input_features = ticker_data.iloc[-1][feature_columns].values.reshape(1, -1)

        preds = model.predict(input_features)[0] # unpack what we have

        # now we'll map the predictions into the dict
        keys = ["predicted_open", "predicted_high", "predicted_low", "predicted_close", "predicted_volume"]
        for key, val in zip(keys, preds):
            case[key] = float(val)  # converting to float for json compatibility

    return updated_cases

feature_columns = ['Return_1d', 'MA_5', 'MA_10', 'Volatility_5', 'Sector_Avg_Return']

updated_test_cases = predict_test_cases(
    test_cases=test_cases,
    full_data=train_df,   # we need to include past data for all tickers
    model=multi_model,  # our trained multi-target regression model
    feature_columns=feature_columns
)

with open("predictedFromTheBlind.json", "w") as f:
    json.dump(updated_test_cases, f, indent=4)
