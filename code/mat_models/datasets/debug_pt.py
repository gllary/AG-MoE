import torch

pt = "/Users/yuxia.guan/mat_agent_project/model_agent_project/data/matbench_phonons.pt"

data = torch.load(pt)

print("Type:", type(data))

if isinstance(data, list):
    print("List length:", len(data))
    print("First element type:", type(data[0]))
    print("First element:", data[0])
else:
    print("Loaded object:", data)