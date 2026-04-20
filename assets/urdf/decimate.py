import pymeshlab as ml

ms = ml.MeshSet()
ms.load_new_mesh('rgm3_base_link.STL')
# 保留 50% 面片，可多次运行或调比例
ms.meshing_decimation_quadric_edge_collapse(targetfacenum=180000)
ms.save_current_mesh('rgm3_base_link_dec.stl', binary=True)