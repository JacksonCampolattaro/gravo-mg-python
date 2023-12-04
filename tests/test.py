import os
import gravomg
import torch
import torch_geometric
import torch_cluster
import polyscope

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), f'cube.obj')
cube = torch_geometric.io.read_obj(path)
data = torch_geometric.transforms.sample_points.SamplePoints(num=4096, include_normals=True)(cube)
batch = torch.zeros(data.pos.shape[0], dtype=torch.int64)
k = 20

neighbor_indices = torch_cluster.knn_graph(
    data.pos, k=k, batch=batch,
    loop=True, flow='target_to_source'
)[1].reshape(data.pos.shape[0], k)

samples = gravomg.sampling.fast_disc_sample(data.pos, neighbor_indices, radius=0.3)
samples = torch.LongTensor(samples)
sampled_positions = torch.gather(
    data.pos,
    index=samples.unsqueeze(-1).expand(-1, data.pos.shape[-1]),
    dim=0
)

polyscope.init()
polyscope.register_point_cloud("positions", data.pos)
polyscope.register_point_cloud("sampled-positions", sampled_positions)
polyscope.show()