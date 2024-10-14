Source code for the paper
# On explainability of reinforcement learning-based machine learning agents trained with Proximal Policy Optimization that utilizes visual sensor data

Abstract:

In this paper, we will address the issues of the explainability of reinforcement learning-based machine learning agents trained with Proximal Policy Optimization (PPO) that utilizes visual sensor data. We will propose an algorithm that allow an effective and intuitive approximation of the PPO-trained neural network. We will also conduct several experiments to confirm our method's effectiveness. Our proposed method works well for scenarios where semantic clustering of the scene is possible. It is based on the solid theoretical foundation of the Gradient-weighted Class Activation Mapping (GradCAM) and Classification And Regression Tree with additional proxy geometry heuristics. It excelled in the explanation process in a virtual simulation system based on a video system with relatively low resolution. Depending on the convolutional features extractor of the PPO-trained neural network, our method obtained 0.945 to 0.968 accuracy of approximation of the black-box model. We have published all source codes so our experiments can be reproduced.

