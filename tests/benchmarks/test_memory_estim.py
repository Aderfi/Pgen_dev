class GlobalGNNMemoryEstimator:
    def __init__(self, dtype_bytes=4, safety_margin=1.3):
        self.dtype_bytes = dtype_bytes
        self.margin = safety_margin

    def calculate_vram(self, datasets, model_config):
        total_nodes = 0
        total_edges = 0
        total_static_data = 0
        total_activations = 0
        
        # 1. Calculate Static Data (Features & Structure)
        for ds in datasets:
            n_total = ds['count'] * ds['avg_nodes']
            e_total = ds['count'] * ds['avg_edges']
            
            # Feature memory: (Nodes * N_feat + Edges * E_feat) * bytes
            feat_mem = (n_total * ds['n_feat'] + e_total * ds['e_feat']) * self.dtype_bytes
            # Adjacency: 2 * E * 8 bytes (Int64)
            adj_mem = e_total * 2 * 8
            
            total_static_data += (feat_mem + adj_mem)
            total_nodes += n_total
            total_edges += e_total

        # 2. Calculate Model Parameters (once)
        # Assuming GATv2 Layer: W is [In x Out], a is [Out]
        h_dims = [datasets[0]['n_feat']] + model_config['hidden_dims']
        heads = model_config['heads']
        total_params_mem = 0
        
        for i in range(len(h_dims) - 1):
            # Param count for GATv2 per layer
            p = (h_dims[i] * h_dims[i+1] * heads) + (h_dims[i+1] * heads)
            total_params_mem += p * self.dtype_bytes

        # 3. Calculate Activations (The bottleneck for 'all-at-once')
        # GATv2 materializes (Edges * Heads) tensors for attention
        for i in range(len(h_dims) - 1):
            # Node activations: Nodes * Out_dim * Heads
            act_n = total_nodes * h_dims[i+1] * heads
            # Edge activations: Edges * Heads (Critical for GATv2)
            act_e = total_edges * heads 
            total_activations += (act_n + act_e) * self.dtype_bytes

        # 4. Optimizer & Gradients (Adam = 3x Params + 1x Params for Grads)
        overhead = total_params_mem * 4

        grand_total = (total_static_data + total_params_mem + total_activations + overhead) * self.margin
        
        return {
            "Static Graph Data (GB)": total_static_data / 1e9,
            "Activations (GB)": total_activations / 1e9,
            "Total VRAM Required (GB)": grand_total / 1e9
        }

# --- Specific Dataset Stats ---
# Update avg_nodes/edges with your actual dataset means
data_specs = [
    {"name": "Chemical", "count": 5150, "avg_nodes": 30, "avg_edges": 65, "n_feat": 25, "e_feat": 9},
    {"name": "Genomic", "count": 9100, "avg_nodes": 150, "avg_edges": 400, "n_feat": 9, "e_feat": 3}
]

model_specs = {"hidden_dims": [64, 64], "heads": 4}

estimator = GlobalGNNMemoryEstimator()
report = estimator.calculate_vram(data_specs, model_specs)

print(f"Estimation for loading {sum(d['count'] for d in data_specs)} graphs:")
for k, v in report.items():
    print(f"{k}: {v:.2f}")