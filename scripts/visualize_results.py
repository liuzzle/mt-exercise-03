import re
import pandas as pd
import matplotlib.pyplot as plt

# Paths to the log files
log_files = {
    "Baseline": "configs/baseline.log",
    "Prenorm": "logs/deen_transformer_regular/err_pre",
    "Postnorm": "logs/deen_transformer_regular/err_post",
}

# Regular expression to extract validation perplexities
ppl_pattern = re.compile(r"Evaluation result \(greedy\).*ppl:\s+([\d.]+)")

# Function to extract validation perplexities from a log file
def extract_validation_ppl(log_file):
    steps = []
    perplexities = []
    with open(log_file, "r") as file:
        for line in file:
            if "Evaluation result (greedy)" in line and "ppl:" in line:
                match = ppl_pattern.search(line)
                if match:
                    step = len(steps) * 500 + 500  # validation is done every 500 steps
                    steps.append(step)
                    perplexities.append(float(match.group(1)))
    return steps, perplexities

# Extract validation perplexities for each model
data = {}
for model, log_file in log_files.items():
    steps, perplexities = extract_validation_ppl(log_file)
    data[model] = perplexities

# Create a DataFrame for the table
df = pd.DataFrame(data, index=steps)
df.index.name = "Validation PPL"
print(df)

# Save the table as a CSV file
df.to_csv("validation_perplexities.csv")

# Plot the validation perplexities
plt.figure(figsize=(10, 6))
for model in log_files.keys():
    plt.plot(df.index, df[model], label=model)

plt.title("Validation Perplexities")
plt.xlabel("Steps")
plt.ylabel("Perplexity")
plt.legend()
plt.tight_layout()

# Save the plot
plt.savefig("validation_perplexities.png")

plt.show()