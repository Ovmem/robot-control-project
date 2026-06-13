import mujoco

model = mujoco.MjModel.from_xml_path(
    "models/panda/panda.xml"
)

print()

print("number of sites =")

print(model.nsite)

print()

for i in range(model.nsite):

    print(
        i,
        model.site(i).name
    )