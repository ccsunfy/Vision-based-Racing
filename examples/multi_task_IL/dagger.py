import torch as th
from torch.utils.data import Dataset, DataLoader
from torch import optim
import numpy as np

from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
from tqdm import tqdm

class DAggerDataset(Dataset):
    def __init__(self):
        self.states = []
        self.teacher_actions = []
    
    def add_data(self, states, teacher_actions):
        self.states.extend(states)
        self.teacher_actions.extend(teacher_actions)
    
    def __len__(self):
        return len(self.states)
    
    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.teacher_actions[idx]
        
        if state is None or action is None:
            raise ValueError(f"Invalid data at index {idx}")
        
        return state, action
        
class DAgger():
    def __init__(
                self,
                env: Union[GymEnv, str],
                learning_rate: Union[float, Schedule] = 3e-4,
                learning_steps: int = int(5e7),
                device: Union[th.device, str] = "cpu",
                seed: Optional[int] = None,
                student: Optional[th.nn.Module] = None,
                teachers: Optional[List[th.nn.Module]] = None,
                save_folder: str = "saved",
                latent_dim: int = 256,
                loss_type = "mse",
                num_episodes_per_iter: int = 256,
                batch_size: int = 512,
                 ):
        self.env = env
        self.learning_steps = learning_steps
        self.learning_rate = learning_rate
        self.save_folder = save_folder
        self.latent_dim = latent_dim
        self.loss_type = loss_type
        self.seed = seed
        self.device = th.device(device if th.cuda.is_available() else "cpu")
        self.student = student.to(self.device)
        self.teachers = teachers
        for teacher in self.teachers:
            teacher.to(self.device)  
        self.learning_rate = learning_rate
        self.num_episodes_per_iter = num_episodes_per_iter
        self.batch_size = batch_size
        
        self.states = []
        self.depths = []
        self.obss = []
        self.dagger_train()
    
    def dagger_train(self):
        dataset = DAggerDataset()
        optimizer = optim.Adam(self.student.parameters(), lr=1e-3)
        
        for step in range(int(self.learning_steps)):
            # 1. student trajectories
            self.collect_student_trajectories()
            # 2. teacher actions
            teacher_actions = []
            for obs in self.obss:
                depth_value = obs['depth']
                obs.pop('depth') # bug
                # teacher_action = [teacher.predict(state, deterministic=True) for teacher in self.teachers]
                actions = []
                
                # average the actions from all tasks
                for teacher in self.teachers:
                    action, _ = teacher.predict(obs, deterministic=True)
                    actions.append(action)
                avg_action = np.mean(actions, axis=0)
                
                teacher_actions.append(avg_action)
                obs['depth'] = depth_value
                
            # 3. DAgger
            dataset.add_data(self.obss, teacher_actions)
            
            # 4. train student policy
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
            for epoch in range(100):  
                total_loss = 0.0
                for state, teacher_action in dataloader:
                    state = state.to(self.device)
                    teacher_action = teacher_action.to(self.device)
                    optimizer.zero_grad()
                    student_actions = self.student(state['depth'])
                    # if self.loss_type == "l2":
                    #     loss = self.l2_loss(student_actions, teacher_action)
                    # elif self.loss_type == "cross_entropy":
                    #     loss = self.cross_entropy_loss(student_actions, teacher_action) 
                        
                    loss = self.mse_loss(student_actions, teacher_action)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                print(f"Iter {step}, Epoch {epoch}, Loss: {total_loss / len(dataloader):.4f}")

            # self.evaluate_policy()

    def collect_student_trajectories(self):
        for _ in range(self.num_episodes_per_iter):
            obs = self.env.reset()
            done_all = th.full((self.env.num_envs,), False)
            if not done_all.all():
                self.obss.append(obs)
                self.states.append(obs['state'])
                self.depths.append(obs['depth'])
                depth = th.FloatTensor(obs['depth']).to(self.device)
                with th.no_grad():
                    action = self.student(th.FloatTensor(depth)) 
                    clipped_actions = th.clip(action, -1, 1)
                    next_obs, _, done, _ = self.env.step(clipped_actions)
                obs = next_obs

    # def kl_divergence(self, student_logits, teacher_probs):
    #     student_probs = th.softmax(student_logits, dim=1)
    #     return th.sum(teacher_probs * (th.log(teacher_probs + 1e-8) - th.log(student_probs + 1e-8)), dim=1).mean()

    def l2_loss(self, student_actions, teacher_actions):
        return th.mean(th.square(student_actions - teacher_actions))
    
    def cross_entropy_loss(self, student_actions, teacher_actions):
        return th.nn.functional.cross_entropy(student_actions, teacher_actions)
    
    def mse_loss(self, student_actions, teacher_actions):
        return th.nn.functional.mse_loss(student_actions, teacher_actions)
    
    def kl_loss(self, student_actions, teacher_actions):
        return th.nn.functional.kl_div(th.log_softmax(student_actions, dim=1), teacher_actions, reduction='batchmean')
    
    def evaluate_policy(self):
        total_reward = 0.0
        for _ in range(self.num_episodes_per_iter):
            obs = self.env.reset()
            done = False
            while not done:
                with th.no_grad():
                    action = th.argmax(self.policy(th.FloatTensor(obs['depth']))).item()
                obs, reward, done, _ = self.env.step(action)
                total_reward += reward
        avg_reward = total_reward / self.num_episodes
        print(f"Evaluation: Average Reward = {avg_reward:.1f}")
        return avg_reward