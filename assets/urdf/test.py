import mujoco
from mujoco import MjModel

model = MjModel.from_xml_path("test.xml")

print("nv =", model.nv)
for dof_id in range(model.nv):
    j = model.dof_jntid[dof_id]
    print(f"DOF {dof_id} → joint {model.joint(j).name}")
