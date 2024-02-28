# Franka Panda Neural Distance Function
This code is an algorithm that uses a neural network to calculate the minimum distance per link based on the configuration of the franka panda manipulator.
If you enter an arbitrary joint angle (7 dimensions), it will return the minimum distance (1 dimension).

## Configuration of Code
1. collision_description

    Description file about Franka Panda

3. collision_moveit_config

    Moveit code about Franka Panda

4. data_gen

    Generate data about self collision

5. neural_network

    Neural Network code

## Pre-required
- [Suhan Robot Motion Tools](https://github.com/JunHeonYoon/suhan_robot_model_tools)
- [MoveIt](https://github.com/JunHeonYoon/moveit)
- [Franka Panda](https://github.com/frankaemika/franka_ros)

## Usage
### Data generate
```
roslaunch collision_moveit_connfig demo.launch
```
```
cd data_gen && mkdir data
```
```
python3 data_generate_mt.py --num_th 8 --num_q 10000000
```
- num_th: Number of thread you wnat to use
- num_q: Number of Joint angle data

### Learning NN
```
cd neural_network && mkdir dataset
```
```
cp -r ../data_gen/data/2024_02_19_18_09_35 ./dataset
```
```
python3 train_self.py
```

### Test Learned NN
```
python3 NN_model_self_test.py
```
