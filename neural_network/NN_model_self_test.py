from srmt.planning_scene import PlanningScene
import numpy as np
from math import pi
import matplotlib.pyplot as plt
from models_self import SelfCollNet
import torch



# Parameters
joint_limit = np.array([[-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973],
                        [ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973]])

# Create Planning Scene
pc = PlanningScene(arm_names=["panda"], arm_dofs=[7], base_link="world")

# NN model load
date = "2024_07_03_11_07_18/"
model_file_name = "self_collision.pkl"

model_dir = "model/self/" + date + model_file_name
device = torch.device('cpu')

model = SelfCollNet(
    fc_layer_sizes=[7, 128, 64, 32, 1],
    batch_size=1,
    nerf=True,
    device=device).to(device)

model_state_dict = torch.load(model_dir, map_location=device)
model.load_state_dict(model_state_dict)


plt.ion()
fig, ax = plt.subplots(1, 1, figsize=(6, 2))
lines1 = []
lines2 = []

line1, = ax.plot([],[], label='ans', color="blue", linewidth=4.0, linestyle='--')
line2, = ax.plot([],[], label='pred', color = "red", linewidth=2.0)
ax.legend()
ax.set_ylim([-5,15])
ax.grid()


def plt_func(fig, line1, line2, x_data, y_data, y_hat_data):
    if x_data.shape[0] > 10:
        x_data = x_data[-10:]
        y_data = y_data[-10:]
        y_hat_data = y_hat_data[-10:]
    line1.set_data(x_data, y_data)
    line2.set_data(x_data, y_hat_data)
    ax.set_xlim(x_data[0], x_data[-1])
    fig.canvas.draw()
    fig.canvas.flush_events()

x_data = np.zeros((1))
y_data = np.zeros((1))
y_hat_data = np.zeros((1))



# q_set = np.loadtxt("/home/yoonjunheon/git/MPCC/C++/debug.txt")
# print(q_set.shape)

for iter in range(1,1000):
# for iter, q in enumerate(q_set):
    print("=================================")
    joint_state =  np.random.uniform(low=joint_limit[0], high=joint_limit[1], size=7)
    # joint_state =  qq
    print(joint_state)
    # joint_state =  np.array([0,0,0,-pi/2,0,pi/2,pi/4])
    pc.display(joint_state)
    min_dist = pc.min_distance(joint_state)*100
    with torch.no_grad():
        model.eval()
        x = torch.from_numpy(joint_state.reshape(1, -1).astype(np.float32)).to(device)
        jac = torch.autograd.functional.jacobian(model, x)
        NN_output = model(x)
    min_dist_pred = NN_output.cpu().detach().numpy()[0]
    
    x_data = np.append(x_data, np.array([iter]), axis=0)
    y_data = np.append(y_data, np.array([min_dist]), axis=0)
    y_hat_data = np.append(y_hat_data, min_dist_pred, axis=0)
    

    plt_func(fig, line1, line2, x_data, y_data, y_hat_data)
    
    print(min_dist)
    print(min_dist_pred)
    print(jac)
    print("=================================")

    plt.pause(0.5)
    # if min_dist == 0:
    #     plt.pause(0.1)

plt.show()