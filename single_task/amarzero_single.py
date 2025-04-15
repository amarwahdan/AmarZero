!wget https://motchallenge.net/data/MOT17.zip
!unzip MOT17.zip


import cv2
import numpy as np
from torchvision.models import vit_b_16
from torchvision import transforms
import torch
import torch.nn as nn
import torch.optim as optim
import math
import pandas as pd
from IPython.display import display
from PIL import Image
import os

os.makedirs('amarzero_results/frames', exist_ok=True)

Vision Transformer (ViT)
vit = vit_b_16(pretrained=True)
vit.eval()

# Kalman Filter
class KalmanFilter:
    def __init__(self, initial_x, initial_y):
        self.dt = 1.0
        self.A = np.array([[1, 0, self.dt, 0], 
                          [0, 1, 0, self.dt], 
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]])
        self.H = np.array([[1, 0, 0, 0], 
                          [0, 1, 0, 0]])
        self.P = np.eye(4) * 10 
        self.Q = np.eye(4) * 0.01 
        self.R = np.eye(2) * 0.5 
        self.x = np.array([[initial_x], [initial_y], [0], [0]])  

    def predict(self):
        self.x = np.dot(self.A, self.x)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        return self.x[:2]

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(4) - np.dot(K, self.H)), self.P)

# World Model (Neural Network)
class WorldModel(nn.Module):
    def __init__(self):
        super(WorldModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(6, 128), 
            nn.ReLU(),
            nn.Dropout(0.3),   
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )
    
    def forward(self, state):
        return self.fc(state)

# MCTS Node
class MCTSNode:
    def __init__(self, state):
        self.state = state
        self.visits = 0
        self.value = 0
        self.children = {}

# MCTS
class MCTS:
    def __init__(self, model):
        self.model = model
    
    def select(self, node):
        best_score = -float('inf')
        best_action = None
        for action, child in node.children.items():
            ucb = child.value / (child.visits + 1e-5) + math.sqrt(math.log(node.visits + 1) / (child.visits + 1e-5))
            if ucb > best_score:
                best_score = ucb
                best_action = action
        return best_action
    
    def expand(self, node, actions):
        for action in actions:
            action_tuple = tuple(action)
            if action_tuple not in node.children:
                next_state = self.model(torch.FloatTensor(node.state + action).unsqueeze(0))
                node.children[action_tuple] = MCTSNode(next_state.detach().numpy()[0].tolist())
    
    def simulate(self, node, depth=5):
        if depth == 0:
            return 0
        action = np.random.uniform(-1, 1, size=2).tolist()
        next_state = self.model(torch.FloatTensor(node.state + action).unsqueeze(0))
        
        position_error = np.linalg.norm(next_state[0, :2].detach().numpy())
        velocity = next_state[0, 2:4].detach().numpy()
        velocity_penalty = 0.1 * np.linalg.norm(velocity)
        reward = - (position_error + velocity_penalty)
        return reward + self.simulate(MCTSNode(next_state.detach().numpy()[0].tolist()), depth - 1)
    
    def backpropagate(self, path, reward):
        for node in path:
            node.visits += 1
            node.value += reward
    
    def run(self, root, actions, simulations=50):
        for _ in range(simulations):
            node = root
            path = [node]
            while node.children:
                action = self.select(node)
                node = node.children[action]
                path.append(node)
            self.expand(node, actions)
            reward = self.simulate(node)
            self.backpropagate(path, reward)
        return self.select(root)


preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),  
    transforms.ColorJitter(brightness=0.2, contrast=0.2), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


cap = cv2.VideoCapture('MOT17/train/MOT17-02-FRCNN/img1/%06d.jpg')


initial_x, initial_y = 960, 538 
kf = KalmanFilter(initial_x, initial_y)
wm = WorldModel()
mcts = MCTS(wm)
optimizer = optim.Adam(wm.parameters(), lr=0.001, weight_decay=1e-4)  
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5) 

frame_count = 0
max_frames = 20
pred_data = []

while cap.isOpened() and frame_count < max_frames:
    ret, frame = cap.read()
    if not ret:
        break

    
    input_tensor = preprocess(frame).unsqueeze(0)
    with torch.no_grad():
        features = vit(input_tensor)

   
    cx = frame.shape[1] // 2
    cy = frame.shape[0] // 2
    center = np.array([[cx], [cy]])
    pred = kf.predict()
    kf.update(center)

    
    state = torch.FloatTensor(kf.x.flatten()).unsqueeze(0)
    dummy_action = torch.zeros(1, 2)  
    input_to_wm = torch.cat([state, dummy_action], dim=1)
    next_state = wm(input_to_wm)

   
    root = MCTSNode(kf.x.flatten().tolist())
    actions = [[dx, dy] for dx in [-1, 0, 1] for dy in [-1, 0, 1] if not (dx == 0 and dy == 0)]
    best_action = mcts.run(root, actions)

   
    target = torch.FloatTensor(kf.x.flatten())
    loss = nn.MSELoss()(next_state.squeeze(), target)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(wm.parameters(), max_norm=1.0)  # Gradient Clipping
    optimizer.step()
    scheduler.step()

   
    pred_x, pred_y = int(pred[0][0]), int(pred[1][0])
    pred_data.append({'frame': frame_count, 'pred_x': pred_x, 'pred_y': pred_y})

    
    cv2.circle(frame, (pred_x, pred_y), 10, (0, 255, 0), -1)
    cv2.line(frame, (pred_x, pred_y), (pred_x + int(best_action[0]*10), pred_y + int(best_action[1]*10)), (255, 0, 0), 2)

    
    save_path = f'amarzero_results/frames/frame_{frame_count}.jpg'
    cv2.imwrite(save_path, frame)

    frame_count += 1

cap.release()
cv2.destroyAllWindows()



gt_file = 'MOT17/train/MOT17-02-FRCNN/gt/gt.txt'
gt_data = []
with open(gt_file, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        if int(parts[1]) == 1:
            frame = int(parts[0])
            x = float(parts[2]) + float(parts[4]) / 2
            y = float(parts[3]) + float(parts[5]) / 2
            gt_data.append({'frame': frame, 'gt_x': x, 'gt_y': y})


pred_df = pd.DataFrame(pred_data)
gt_df = pd.DataFrame(gt_data)
result = pd.merge(pred_df, gt_df, on='frame')


result['distance'] = np.sqrt((result['pred_x'] - result['gt_x'])**2 + (result['pred_y'] - result['gt_y'])**2)
print(result[['frame', 'pred_x', 'pred_y', 'gt_x', 'gt_y', 'distance']])
print('Average Error:', result['distance'].mean())


result.to_csv('amarzero_results/evaluation.csv', index=False)
print('Evaluation report saved to amarzero_results/evaluation.csv')












