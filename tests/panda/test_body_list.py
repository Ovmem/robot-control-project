import mujoco

model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

print()

print("number of bodies =")

print(model.nbody)

print()

for i in range(model.nbody):

    print(
        i,
        model.body(i).name
    )