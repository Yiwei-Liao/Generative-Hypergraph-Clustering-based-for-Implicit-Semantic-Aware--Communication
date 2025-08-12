# Generative Hypergraph Modularity for Semantic-Aware Communication Networks



## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

*   Python 3.8+
*   pip

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Yiwei-Liao/Hypergraph-MLE-based-Implicit-Semantic-Recovery-for-Semantic-Communication.git
    ```

2.  **Install dependencies**
    Install all the required packages using `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

## Usage: Running the Experiment

Follow these steps to reproduce the experimental results.

### Step 1: Prepare the Dataset

First, you need to prepare the dataset.

1.  Unzip the original dataset file located in the `Data_origin/` directory.
2.  Run the processing script to prepare the data for the model. The following command processes the `cora` dataset.
    ```bash
    python Load_process.py --dataset cora
    ```
    This script will generate the processed data files required for the next step.

### Step 2: Run the Main Experiment

Once the dataset is prepared, you can run the main experiment script.

```bash
python main.py

