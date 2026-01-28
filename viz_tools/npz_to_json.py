import torch
import json 

method_type = "raw"
p = 90
p1 = 90
p2 = 60

num_layers = 36
num_heads = 8

frequency = 1
max_step = 30

def convert(path=f"./demo_logs/{method_type}_selected_topp_indices_p{p}.pt", num_topp_path=f"./demo_logs/{method_type}_num_topp_p{p}.pt"):
    data = torch.load(path)
    data_num = torch.load(num_topp_path)
    data_num = torch.stack(data_num, dim=0)
    
    data_num = data_num.reshape(-1, num_layers, num_heads)

    steps = [step * 32 + 512 for step in list(range(len(data) // num_layers))]
    
    json_output = []
    
    for step_i in range(len(data) // num_layers):
        if step_i % frequency != 0:
             continue
        if step_i >= max_step:
            break
        json_data = []
        for layer_id in range(num_layers):
            i = step_i * num_layers + layer_id
            selected_indices = data[i]
            
            num = data_num[step_i][layer_id]
            heatmap = torch.zeros_like(selected_indices)
            indices_list = []
            for head_i in range(selected_indices.shape[0]):
                indices = selected_indices[head_i][:num[head_i]].tolist()
                
                # make tuple of (head_i, index)
                for idx in indices:
                    assert idx < heatmap.shape[-1]
                tupled_indices = [[head_i, idx] for idx in indices]
                indices_list.extend(tupled_indices)
            
            heatmap[tuple(zip(*indices_list))] = 1.0
            json_data.append(heatmap)
        json_data = torch.stack(json_data, dim=0).cpu().numpy().tolist()
        json_output.append({
            "shape": [num_layers, num_heads, heatmap.shape[-1]],
            "data": json_data,
        })
    
    with open(f"./json_data/{method_type}_selected_topp_indices_p{p}.json", "w") as f:
        json.dump(json_output, f)

def convert_diff(path_p1=f"./demo_logs/{method_type}_selected_topp_indices_p{p1}.pt", 
                 path_p2=f"./demo_logs/{method_type}_selected_topp_indices_p{p2}.pt", 
                 num_topp_path_p1=f"./demo_logs/{method_type}_num_topp_p{p1}.pt", 
                 num_topp_path_p2=f"./demo_logs/{method_type}_num_topp_p{p2}.pt"):
    data_p1 = torch.load(path_p1)
    data_p2 = torch.load(path_p2)
    data_num_p1 = torch.load(num_topp_path_p1)
    data_num_p2 = torch.load(num_topp_path_p2)
    data_num_p1 = torch.stack(data_num_p1, dim=0)
    data_num_p2 = torch.stack(data_num_p2, dim=0)
    
    data_num_p1 = data_num_p1.reshape(-1, num_layers, num_heads)
    data_num_p2 = data_num_p2.reshape(-1, num_layers, num_heads)

    steps = [step * 32 + 512 for step in list(range(len(data_p1) // num_layers))]
    
    json_output = []
    
    for step_i in range(len(data_p1) // num_layers):
        if step_i % frequency != 0:
             continue
        if step_i >= max_step:
            break
        json_data = []
        for layer_id in range(num_layers):
            i = step_i * num_layers + layer_id
            selected_indices_p1 = data_p1[i]
            selected_indices_p2 = data_p2[i]
            
            num_1 = data_num_p1[step_i][layer_id]
            num_2 = data_num_p2[step_i][layer_id]
            heatmap = torch.zeros_like(selected_indices_p2)
            indices_list = []
            for head_i in range(selected_indices_p2.shape[0]):
                indices_p1 = set(selected_indices_p1[head_i][:num_1[head_i]].tolist())
                indices_p2 = set(selected_indices_p2[head_i][:num_2[head_i]].tolist())

                diff_indices = indices_p1 - indices_p2
                
                for idx in diff_indices:
                    assert idx < heatmap.shape[-1]
                tupled_indices = [[head_i, idx] for idx in diff_indices]
                indices_list.extend(tupled_indices)
            if len(indices_list) > 0:
                heatmap[tuple(zip(*indices_list))] = 1.0

            json_data.append(heatmap)
        
        json_data = torch.stack(json_data, dim=0).cpu().numpy().tolist()
        json_output.append({
            "shape": [num_layers, num_heads, heatmap.shape[-1]],
            "data": json_data,
        })
    
    with open(f"./json_data/{method_type}_selected_topp_diff_indices_p{p1}_p{p2}.json", "w") as f:
        json.dump(json_output, f)

if __name__ == "__main__":
    convert()
    # convert_diff()