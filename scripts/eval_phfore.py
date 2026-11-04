from torch_geometric.data import HeteroData

data = HeteroData()
data['atom'].x = atom_x            # [N, F_atom]
data['atom'].pos = atom_pos        # [N, 3]
data['atom', 'bond', 'atom'].edge_index = bond_index

data['pharm'].x = pharm_x          # [K, F_pharm]  (tipo one-hot + radio)
data['pharm', 'dist', 'pharm'].edge_index = pharm_edge_index
data['pharm', 'dist', 'pharm'].edge_attr = pharm_dist     # [K(K-1), 1]

# correspondencia átomo -> feature (la "matriz de match" en formato disperso)
data['atom', 'matches', 'pharm'].edge_index = match_index  # [2, M]z
