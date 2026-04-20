import os, sys
sys.path.append(os.path.abspath('../gym_dcmm'))
import math
import time
import torch
import torch.distributed as dist

import wandb

import numpy as np

from .experience import ExperienceBuffer
from .models_track import ActorCritic
from .utils import AverageScalarMeter, RunningMeanStd

from tensorboardX import SummaryWriter

class PPO_Catch_OneStage(object):
    def __init__(self, env, output_dif, full_config):#full_config就是config文件
        self.rank = -1#表示不使用分布式训练，只在一台计算机上训练
        self.device = full_config['rl_device']#指定训练所使用的设备
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env#强化学习的环境实例
        self.num_actors = int(self.ppo_config['num_actors'])#并行执行环境的智能体数量（或称为 “演员数量”），用于高效收集多线程 / 多进程的经验数据。
        self.actions_num = self.env.call("act_c_dim")[0]#动作空间的维度（即智能体可执行的动作数量或连续动作的维度）自由度
        self.actions_low = self.env.call("actions_low")[0]#动作空间的下界（连续动作时每个维度的最小值）每个动作能做的最大最小限度
        self.actions_high = self.env.call("actions_high")[0]#动作空间的上界（连续动作时每个维度的最大值）
        self.obs_shape = (self.env.call("obs_c_dim")[0],)#观测空间的形状（即观测数据的维度）
        self.full_action_dim = self.env.call("act_c_dim")[0]#完整的动作维度（与 actions_num 一致，可能用于后续动作补全或校验）。

        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'separate_value_mlp': self.network_config.get('separate_value_mlp', True),#.get是获取字典中括号里面的键对应的值
        }#如果配置中没有 separate_value_mlp 这个参数（比如被注释或遗漏），就自动返回 True
        print("net_config: ", net_config)
        self.model = ActorCritic(net_config)
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)#标准化输入
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)#标准化价值估计
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config['learning_rate'])#从配置中获取初始学习率
        self.last_lr = float(self.ppo_config['learning_rate'])#记录当前的学习率
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.init_lr, eps=1e-5)#创建一个adam优化器
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']#剪辑系数
        self.action_track_denorm = self.ppo_config['action_track_denorm']#动作反归一化系数PPO 输出的动作通常经过 tanh 归一化到 [-1, 1]，这两个参数用于将归一化动作转换为环境可执行的实际物理动作（如机械臂关节角度范围 [-π, π]）。
        self.action_catch_denorm = self.ppo_config['action_catch_denorm']#同上
        self.clip_value = self.ppo_config['clip_value']#布尔值，如果是True就会对价值进行剪辑
        self.entropy_coef = self.ppo_config['entropy_coef']#熵奖励系数，在损失函数中加入动作分布的熵（entropy），系数越大，越鼓励策略保持探索性（避免过早收敛到局部最优）。
        self.critic_coef = self.ppo_config['critic_coef']#价值损失系数（控制价值函数（Critic）损失在总损失中的占比，平衡策略更新与价值估计的学习速度。
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']#动作边界损失系数作用：当动作接近环境允许的边界（如关节角度上限）时，添加额外惩罚损失，鼓励策略输出更安全的动作，避免触发环境约束报错。

        self.gamma = self.ppo_config['gamma']#折扣因子
        self.tau = self.ppo_config['tau']#GAE（Generalized Advantage Estimation）的平滑系数
        self.truncate_grads = self.ppo_config['truncate_grads']#布尔值，防止出现梯度爆炸的情况
        self.grad_norm = self.ppo_config['grad_norm']#梯度裁剪的阈值，如果梯度超过这个阈值，就会进行缩小
        self.value_bootstrap = self.ppo_config['value_bootstrap']#布尔值
        self.normalize_advantage = self.ppo_config['normalize_advantage']#布尔值，是否标准化优势函数
        self.normalize_input = self.ppo_config['normalize_input']#是否标准化观测输入
        self.normalize_value = self.ppo_config['normalize_value']#是否标准化价值估计
        self.reward_scale_value = self.ppo_config['reward_scale_value']#奖励缩放系数
        self.clip_value_loss = self.ppo_config['clip_value_loss']#价值损失剪辑开关
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']#单个智能体（或单个独立环境实例）在一次连续交互中，从初始状态开始，执行动作、接收奖励，直到停止收集数据为止的总步数。每一个时间步都包含很多数据
        self.batch_size = self.horizon_length * self.num_actors#总运行批量的大小，num_actor并行执行环境的智能体数量（或称为 “演员数量”），用于高效收集多线程 / 多进程的经验数据。
        self.minibatch_size = self.ppo_config['minibatch_size']#小批量样本大小
        self.mini_epochs_num = self.ppo_config['mini_epochs']#小批量迭代次数，总批量样本被重复用于训练的次数
        assert self.batch_size % self.minibatch_size == 0 or full_config.test#batch_size 必须是 minibatch_size 的整数倍；仅在测试模式下允许不满足该条件。
        # ---- scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':#基于 KL 散度的自适应学习率）学习率是朝着梯度方向调整的幅度
            self.kl_threshold = self.ppo_config['kl_threshold']#KL 散度阈值（浮点数，如 0.01 或 0.02），用于判断策略更新是否 “过于激进”。
            self.scheduler = AdaptiveScheduler(self.kl_threshold)#调整学习率
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(
                self.init_lr,
                self.ppo_config['max_agent_steps'])
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']#模型定期保存频率，神经网络参数（以及相关训练状态）的保存频率
        self.save_best_after = self.ppo_config['save_best_after']#开始保存 “最佳模型” 的阈值，前多少步不考虑保存最佳模型
        # ---- Tensorboard Logger ----
        self.extra_info = {}#用于存储训练过程中产生的额外信息（非核心指标但可能需要记录的辅助数据）。
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer#创建一个 SummaryWriter 实例，后续通过该实例将训练指标（如损失、奖励、成功率）写入日志文件，供 TensorBoard 可视化。
        #用来储存ppo算法的关键数据

        self.episode_rewards = AverageScalarMeter(200)#初始化一个滑动平均计算器，用于记录训练过程中 “每个 episode 的奖励值” 的滑动平均值。一个episode就是一个回合，即从游戏开始到游戏结束
        self.episode_lengths = AverageScalarMeter(200)#初始化滑动平均计算器，记录 “每个 episode 的长度（步数）” 的滑动平均值。
        self.episode_success = AverageScalarMeter(200)#初始化滑动平均计算器，记录 “每个 episode 的成功率” 的滑动平均值。
        self.episode_test_rewards = AverageScalarMeter(self.ppo_config['test_num_episodes'])#初始化测试阶段的奖励滑动平均计算器。
        self.episode_test_lengths = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_success = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        #前面三个参数是训练阶段，后面三个参数是测试阶段
        self.obs = None#初始化观测数据存储变量 self.obs
        self.epoch_num = 0#初始化训练轮次计数器 self.epoch_num
        self.storage = ExperienceBuffer(#创建经验缓冲区实例，用于存储智能体与环境交互产生的经验数据（观测、动作、奖励等），供 PPO 算法训练时采样使用。
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
            self.obs_shape[0], self.actions_num, self.device,
        )
        '''
        self.num_actors:并行环境（智能体）的数量，经验数据来自多个并行环境。
        self.horizon_length:单个环境在一次数据收集中的交互步数（每个 actor 收集的步数）。
        self.batch_size:总经验批量大小(num_actors * horizon_length)。
        self.minibatch_size:每次训练时从总批量中采样的小批量大小。
        self.obs_shape[0]：观测数据的维度（如观测向量的长度）。
        self.actions_num:动作空间的维度（智能体可执行的动作数量）。
        self.device:数据存储的设备（如 GPU 或 CPU)。
        '''

        batch_size = self.num_actors#将并行环境的数量 self.num_actors 赋值给 batch_size（此处的 batch_size 特指并行环境数量，与前文的总批量大小不同）。
        current_rewards_shape = (batch_size, 1)#定义当前奖励张量的形状为 (并行环境数量, 1)。
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        #初始化一个全零张量，用于记录每个并行环境在当前 episode 中的累积奖励。
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        #初始化一个全零张量，用于记录每个并行环境在当前 episode 中的步数（长度）。
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        #初始化一个全为 1 的张量，用于标记每个并行环境的 episode 是否结束（1 表示结束，0 表示未结束）。
        self.agent_steps = 0#初始化智能体总交互步数计数器，初始值为 0。记录智能体与环境交互的总步数（所有并行环境的步数之和），用于控制训练终止（当达到 max_agent_steps 时停止训练）和学习率调度。
        self.max_agent_steps = self.ppo_config['max_agent_steps']#所有智能体（并行环境）的总交互步数之和，而不是单个智能体的最大步数。
        self.max_test_steps = self.ppo_config['max_test_steps']#从配置中读取测试的最大总交互步数，作为测试终止条件。
        self.best_rewards = -10000#初始化最佳平均奖励的记录值，初始值设为一个较小的负数。
        # ---- Timing
        self.data_collect_time = 0#累计智能体与环境交互（收集经验）的总时间。
        #收集经验” 是训练过程的核心环节之一，指的是智能体与环境进行交互，执行动作、获取观测、奖励和终止信号，并将这些交互数据存储起来用于后续模型训练的过程。
        self.rl_train_time = 0#累计模型参数更新（训练）的总时间。
        self.all_time = 0#累计整个训练过程的总时间（后续在训练结束时计算）。

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        '''
        a_losses:演员(Actor)损失的列表（每个小批量的损失值）。
        c_losses:评论家(Critic)损失的列表。
        b_losses:动作边界损失(Bounds Loss)的列表。
        entropies:动作分布的熵（鼓励探索）的列表。
        kls:新旧策略之间的 KL 散度（衡量策略更新幅度）的列表。
        '''
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,#训练阶段的帧率,总交互步数（self.agent_steps）÷ 累计训练时间（self.rl_train_time）。
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,#数据收集阶段的帧率
            'losses/actor_loss': torch.mean(torch.stack(a_losses)).item(),#演员损失的平均值（策略优化的核心损失）。
            'losses/bounds_loss': torch.mean(torch.stack(b_losses)).item(),#动作边界损失的平均值（惩罚接近动作空间边界的动作）。
            'losses/critic_loss': torch.mean(torch.stack(c_losses)).item(),#评论家损失的平均值（价值估计的误差）。
            'losses/entropy': torch.mean(torch.stack(entropies)).item(),#动作分布熵的平均值（熵越高，策略探索性越强）。
            'info/last_lr': self.last_lr,#当前学习率（self.last_lr，可能通过调度器动态调整）。
            'info/e_clip': self.e_clip,#PPO 的剪辑系数（self.e_clip，控制策略更新幅度）。
            'info/kl': torch.mean(torch.stack(kls)).item(),#KL 散度的平均值（衡量新旧策略差异，过大可能导致训练不稳定）。
        }
        for k, v in self.extra_info.items():#补充额外信息到日志字典
            log_dict[f'{k}'] = v

        # log to wandb
        wandb.log(log_dict, step=self.agent_steps)#记录日志到 wandb，参数 step=self.agent_steps：以总交互步数为横坐标，确保不同指标在时间轴上对齐。

        # log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)
        #记录日志到 tensorboard，将标量 v 以 k 为名称、step 为横坐标记录到 tensorboard。
    def set_eval(self):
        self.model.eval()
        if self.normalize_input:
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()

    def set_train(self):
        self.model.train()
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def train(self):
        start_time = time.time()#记录训练开始的总时间
        _t = time.time()#记录累计帧率计算的起始时间
        _last_t = time.time()#记录最近一轮迭代的起始时间
        reset_obs, _ = self.env.reset()#重置环境，返回初始观测 reset_obs（如机械臂初始状态、物体位置等）。
        self.obs = {'obs': self.obs2tensor(reset_obs)}#返回初始观测 reset_obs（如机械臂初始状态、物体位置等）
        #将环境重置后的原始观测数据（reset_obs）经过处理后，存储为模型可直接使用的张量格式
        self.agent_steps = self.batch_size
        #记录智能体与环境的总交互步数，初始化为一个批次大小（batch_size），表示首次迭代前的初始状态。

        while self.agent_steps < self.max_agent_steps:
            #只要总交互步数 self.agent_steps 未达到预设的最大步数 self.max_agent_steps
            self.epoch_num += 1#初始化训练轮次计数器 self.epoch_num，迭代轮次（epoch）加 1，记录当前是第几次完整的 “收集 + 训练” 循环。
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            #执行单轮核心训练流程（包含经验收集和模型更新），返回训练过程中的关键指标列表：
            self.storage.data_dict = None#清空经验缓冲区的数据（释放内存，为下一轮收集做准备）。

            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)

            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (
                self.batch_size) \
                / (time.time() - _last_t)
            _last_t = time.time()
            info_string = f'Agent Steps: {int(self.agent_steps // 1e3):04}K | FPS: {all_fps:.1f} | ' \
                            f'Last FPS: {last_fps:.1f} | ' \
                            f'Collect Time: {self.data_collect_time / 60:.1f} min | ' \
                            f'Train RL Time: {self.rl_train_time / 60:.1f} min | ' \
                            f'Current Best: {self.best_rewards:.2f}'
            print(info_string)#打印训练状态信息

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)
            #调用 write_stats 方法，将损失、熵、KL 散度等指标记录到 WandB（云端可视化）和 TensorBoard（本地可视化）。
            mean_rewards = self.episode_rewards.get_mean()
            #self.episode_rewards：记录每个 episode 的奖励，get_mean() 返回最近 200 个 episode 的平均奖励（由 AverageScalarMeter 维护）。
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()
            # print("mean_rewards: ", mean_rewards)
            self.writer.add_scalar(
                'metrics/episode_rewards_per_step', mean_rewards, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_lengths_per_step', mean_lengths, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_success_per_step', mean_success, self.agent_steps)
            #将上述指标分别写入 TensorBoard 和 WandB，以 self.agent_steps 为横坐标，便于追踪性能随步数的变化。
            wandb.log({
                'metrics/episode_rewards_per_step': mean_rewards,
                'metrics/episode_lengths_per_step': mean_lengths,
                'metrics/episode_success_per_step': mean_success,
            }, step=self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'
            #生成一个包含训练关键信息的文件名，便于后续识别模型训练进度和性能。
            if self.save_freq > 0:#定期保存模型
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, f'last'))

            if mean_rewards > self.best_rewards:#保存最佳性能
                print(f'save current best reward: {mean_rewards:.2f}')
                # remove previous best file
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def save(self, name):#保存模型参数到文件
        weights = {
            'model': self.model.state_dict(),
            'tracking_mlp': self.model.actor_mlp.state_dict(),
            'tracking_mu': self.model.mu.state_dict(),
            'tracking_sigma': self.model.sigma.data
        }
        if self.running_mean_std:
            weights['running_mean_std'] = self.running_mean_std.state_dict()
        if self.value_mean_std:
            weights['value_mean_std'] = self.value_mean_std.state_dict()
        torch.save(weights, f'{name}.pth')

    def restore_train(self, fn):#恢复训练状态（用于继续训练）
        #该方法用于从保存的模型文件（.pth）中加载参数，恢复模型的训练状态，以便在之前训练的基础上继续训练，而不是从头开始。这在实际训练中非常重要，
        if not fn:
            return
        checkpoint = torch.load(fn, map_location = self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

    def restore_test(self, fn):#方法用于从保存的模型文件（.pth）中加载参数，恢复模型的测试状态，以便使用训练好的模型进行推理或评估（而非继续训练）
        checkpoint = torch.load(fn, map_location = self.device)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input:
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
    #上面这两个函数通常用的是同一个.pth文件
    def train_epoch(self):
        # collect minibatch data
        _t = time.time() # 记录数据收集开始时间（用于统计耗时）
        self.set_eval()# 将模型切换为评估模式（关闭 dropout、 BatchNorm 等训练特有的操作，确保数据收集稳定）
        self.play_steps()# 与环境交互，收集一批经验数据（存储到 self.storage 中，包含观测、动作、奖励等）
        self.data_collect_time += (time.time() - _t) # 累计数据收集耗时（用于性能统计）
        # update network
        _t = time.time()# 记录模型更新开始时间
        self.set_train()# 将模型切换为训练模式（启用 dropout、BatchNorm 等训练操作）
        a_losses, b_losses, c_losses = [], [], [] # 初始化列表，分别存储演员损失、边界损失、评论家损失
        entropies, kls = [], []# 初始化列表，分别存储动作熵（用于鼓励探索）和 KL 散度（用于策略更新约束）
        for mini_ep in range(0, self.mini_epochs_num):# 按配置的小批量迭代次数（mini_epochs_num）重复训练
            ep_kls = []# 存储当前小批量迭代中各批次的 KL 散度（用于调整学习率）
            for i in range(len(self.storage)):# 遍历经验缓冲区中的所有小批次（每个小批次是 batch_size 的子集）
                # 从缓冲区中获取当前小批次数据
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs = self.storage[i]
                 # 解析：
                # - value_preds: 旧策略的价值估计（用于价值损失计算）
                # - old_action_log_probs: 旧策略的动作对数概率（用于计算 PPO 比率 ratio）
                # - advantage: 优势函数（衡量动作好坏的相对值，用于指导策略更新）
                # - old_mu/old_sigma: 旧策略的动作分布参数（高斯分布的均值和标准差，用于计算 KL 散度）
                # - returns: 折扣回报（目标值，用于训练价值函数）
                # - actions: 已执行的动作（用于策略更新时的输入）
                # - obs: 观测数据（当前状态输入）

                obs = self.running_mean_std(obs)# 对观测数据进行标准化（使用训练中统计的均值和方差，确保分布稳定）
                batch_dict = {# 构建模型输入字典
                    'prev_actions': actions,# 历史动作（可能用于循环神经网络等依赖历史的模型）
                    'obs': obs,# 标准化后的观测数据
                }
                res_dict = self.model(batch_dict)# 将输入传入模型，得到当前策略的输出
                action_log_probs = res_dict['prev_neglogp']# 当前策略的动作负对数概率（取负后与 old_action_log_probs 对应）
                values = res_dict['values']# 当前策略的价值估计
                entropy = res_dict['entropy'] # 当前策略的动作分布熵（鼓励探索，熵越大策略越随机）
                mu = res_dict['mus']# 当前策略的动作分布均值（高斯分布参数）
                sigma = res_dict['sigmas']# 当前策略的动作分布标准差（高斯分布参数）

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)# 新旧策略概率比的指数（即 π_new / π_old）
                surr1 = advantage * ratio# 未剪辑的优势项（用于策略提升）
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)# 剪辑后的优势项（限制策略更新幅度）
                a_loss = torch.max(-surr1, -surr2)# 演员损失（取两个优势项的最大值的负数，确保策略更新不超过剪辑范围）
                # critic loss# 计算评论家损失（Value Loss）：训练价值函数预测折扣回报
                if self.clip_value_loss:# 如果启用价值剪辑（PPO 可选技巧）
                    # 剪辑价值估计，使其不超过旧估计的 ±e_clip 范围
                    value_pred_clipped = value_preds + \
                        (values - value_preds).clamp(-self.e_clip, self.e_clip)
                    value_losses = (values - returns) ** 2# 未剪辑的价值损失（MSE）
                    value_losses_clipped = (value_pred_clipped - returns) ** 2# 剪辑后的价值损失
                    c_loss = torch.max(value_losses, value_losses_clipped)# 取两者最大值作为价值损失（限制价值函数更新幅
                else:
                    c_loss = (values - returns) ** 2# 不剪辑，直接使用 MSE 损失
                # bounded loss # 计算边界损失（Bounds Loss）：惩罚动作接近边界的情况（避免策略输出极端值）
                if self.bounds_loss_coef > 0: # 如果启用边界损失
                    soft_bound = 1.1# 软边界（略大于动作空间的 [-1,1] 归一化范围，允许一定容错）
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2# 当动作均值 mu 超过上边界 soft_bound 时，计算惩罚（平方项，值越大惩罚越重）
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2 # 当动作均值 mu 低于下边界 -soft_bound 时，计算惩罚
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)# 总边界损失（按动作维度求和）
                else:
                    b_loss = 0 # 不启用边界损失时，损失为 0
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]# 计算各损失的均值（将批次内的损失聚合为标量）

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef \
                    + b_loss * self.bounds_loss_coef# 总损失计算（加权求和）
                # 解析：
                # - 演员损失（a_loss）：直接累加，目标是最大化策略性能
                # - 评论家损失（c_loss）：乘以系数 self.critic_coef，平衡与演员损失的权重；0.5 是 MSE 损失的常见系数
                # - 熵（entropy）：减去熵项（负号），鼓励策略探索（熵越大，惩罚越小）
                # - 边界损失（b_loss）：乘以系数 self.bounds_loss_coef，控制边界惩罚的强度
                self.optimizer.zero_grad()# 清空优化器的梯度缓存（避免梯度累积）
                loss.backward(retain_graph=True)# 计算损失对模型参数的梯度（retain_graph=True 保留计算图，可能用于多轮反向传播）

                if self.truncate_grads:# 如果启用梯度裁剪（防止梯度爆炸）
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
                self.optimizer.step()# 执行参数更新（根据梯度调整模型权重）
                # 计算新旧策略的 KL 散度（衡量策略变化幅度）
                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)

                kl = kl_dist
                # 记录各损失和统计量
                a_losses.append(a_loss)# 记录演员损失
                c_losses.append(c_loss)# 记录评论家损失
                ep_kls.append(kl)# 记录当前小批次的 KL 散度
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())

            av_kls = torch.mean(torch.stack(ep_kls))
            kls.append(av_kls)

            if self.lr_schedule == 'kl':
                self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = self.adjust_learning_rate_cos(mini_ep)

            for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.last_lr

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls
    
    def obs2tensor(self, obs):
        # Map the step result to tensor
        if self.env.call('task')[0] == 'Catching':
            obs_array = np.concatenate((
                        obs["base"]["v_lin_3d"],
                        #obs["base"]["v_lin_2d"], 
                        obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],#机械臂末端的三维位置，以及末端的四元数
                        obs["object"]["pos3d"], obs["object"]["v_lin_3d"], #物体三维线速度和三维位置
                        obs["hand"],
                        obs["object"]["pos_history"],
                        ), axis=1)
            # obs_array = np.concatenate((
            #     obs["base"]["v_lin_3d"][:, :2],
            #     #obs["base"]["v_lin_2d"], 
            #     obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],#机械臂末端的三维位置，以及末端的四元数
            #     obs["object"]["pos3d"], obs["object"]["v_lin_3d"], #物体三维线速度和三维位置
            #     obs["hand"],
            #     ), axis=1)
        #         self.observation_space = spaces.Dict(#定义机器人环境的观测空间，一共30维
        #     {
        #         "base": spaces.Dict({
        #             "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),#spaces.Box(low, high, shape, dtype)

        #         }),
        #         "arm": spaces.Dict({
        #             "ee_pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
        #             "ee_quat": spaces.Box(-1, 1, shape=(4,), dtype=np.float32),
        #             "ee_v_lin_3d": spaces.Box(-1, 1, shape=(3,), dtype=np.float32),
        #             "joint_pos": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in range(9, 15)]),#（9,15)代表的是6个关节，这是给6个关节找上下限
        #                                     high = np.array([self.Dcmm.model.jnt_range[i][1] for i in range(9, 15)]),
        #                                     dtype=np.float32),
        #         }),
        #         "hand": spaces.Box(low = np.array([self.Dcmm.model.jnt_range[i][0] for i in hand_joint_indices]),
        #                            high = np.array([self.Dcmm.model.jnt_range[i][1] for i in hand_joint_indices]),
        #                            dtype=np.float32),#DcmmCfg.hand_mask这个是几维的，上面这个hand就是几维的。
        #         "object": spaces.Dict({
        #             "pos3d": spaces.Box(-10, 10, shape=(3,), dtype=np.float32),
        #             "v_lin_3d": spaces.Box(-4, 4, shape=(3,), dtype=np.float32),
        #             # -------- 【这里是新增的行】 --------
        #             # 3 (x,y,z) * 3 (历史长度) = 9维
        #             "pos_history": spaces.Box(-10, 10, shape=(3 * self.obj_history_len,), dtype=np.float32),
        #             # -
        #             ## TODO: to be determined
        #             # "shape": spaces.Box(-5, 5, shape=(2,), dtype=np.float32),
        #         }),
        #     }
        # )
        else:
            obs_array = np.concatenate((
                    obs["base"]["v_lin_3d"][:, :2],
                    # obs["base"]["v_lin_2d"], #底座的二维速度
                    obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],
                    obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
                    # obs["hand"],# TODO: TEST
                    ), axis=1)
        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
        return obs_tensor

    def action2dict(self, actions):
        actions = actions.cpu().numpy()
        # De-normalize the actions
        if self.env.call('task')[0] == 'Tracking':
            base_tensor = actions[:, :3] * self.action_track_denorm[0]
            arm_tensor = actions[:, 3:6] * self.action_track_denorm[1]
            hand_tensor = actions[:, 6:] * self.action_track_denorm[2]
        else:
            base_tensor = actions[:, :3] * self.action_catch_denorm[0]
            arm_tensor = actions[:, 3:6] * self.action_catch_denorm[1]
            hand_tensor = actions[:, 6:] * self.action_catch_denorm[2]
        actions_dict = {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor
        }
        return actions_dict
    
    def model_act(self, obs_dict, inference=False):
        processed_obs = self.running_mean_std(obs_dict['obs'])
        input_dict = {
            'obs': processed_obs,
        }
        if not inference:
            res_dict = self.model.act(input_dict)
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        else:
            res_dict = {}
            res_dict['actions'] = self.model.act_inference(input_dict)
        return res_dict

    def play_steps(self):#训练阶段的数据收集
        for n in range(self.horizon_length):
            res_dict = self.model_act(self.obs)
            # Collect o_t
            self.storage.update_data('obses', n, self.obs['obs'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # Do env step
            # Clamp the actions of the action space
            actions = res_dict['actions']
            actions[:,:] = torch.clamp(actions[:,:], -1, 1)
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # Update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)
            shaped_rewards = self.reward_scale_value * rewards.clone()
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_lengths += 1
            # print("self.dones: ", self.dones)
            done_indices = self.dones.nonzero(as_tuple=False)
            # print("done_indices: ", done_indices)
            self.episode_rewards.update(self.current_rewards[done_indices])
            self.episode_lengths.update(self.current_lengths[done_indices])
            self.episode_success.update(torch.tensor(truncates, dtype=torch.float32, device=self.device)[done_indices])
            assert isinstance(infos, dict), 'Info Should be a Dict'
            # print("infos: ", infos)
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones

        res_dict = self.model_act(self.obs)
        last_values = res_dict['values']

        self.agent_steps = (self.agent_steps + self.batch_size)
        self.storage.compute_return(last_values, self.gamma, self.tau)
        self.storage.prepare_training()

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def play_test_steps(self):#测试阶段的数据收集
        for _ in range(self.horizon_length):
            res_dict = self.model_act(self.obs, inference=True)
            # Do env step
            # Clamp the actions of the action space 
            actions = res_dict['actions']
            actions[:,:] = torch.clamp(actions[:,:], -1, 1)
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
            actions_dict = self.action2dict(actions)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)
            # Map the dones
            dones = terminates | truncates
            self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)
            # Update dones and rewards after env step
            self.current_rewards += rewards
            self.current_lengths += 1
            done_indices = self.dones.nonzero(as_tuple=False)
            self.episode_test_rewards.update(self.current_rewards[done_indices])
            self.episode_test_lengths.update(self.current_lengths[done_indices])
            self.episode_test_success.update(torch.tensor(truncates, dtype=torch.float32, device=self.device)[done_indices])
            assert isinstance(infos, dict), 'Info Should be a Dict'
            for k, v in infos.items():
                # only log scalars
                if isinstance(v, float) or isinstance(v, int) or (isinstance(v, torch.Tensor) and len(v.shape) == 0):
                    self.extra_info[k] = v

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
        res_dict = self.model_act(self.obs)
        self.agent_steps = (self.agent_steps + self.batch_size)

    def test(self):#该函数是测试阶段的入口，负责启动并控制测试过程
        self.set_eval()
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        self.test_steps = self.batch_size

        while self.test_steps < self.max_test_steps:
            self.play_test_steps()
            self.storage.data_dict = None
            mean_rewards = self.episode_test_rewards.get_mean()
            mean_lengths = self.episode_test_lengths.get_mean()
            mean_success = self.episode_test_success.get_mean()
            print("## Sample Length %d ##" % len(self.episode_test_rewards))
            print("mean_rewards: ", mean_rewards)
            print("mean_lengths: ", mean_lengths)
            print("mean_success: ", mean_success)
            # wandb.log({
            #     'metrics/episode_test_rewards': mean_rewards,
            #     'metrics/episode_test_lengths': mean_lengths,
            # }, step=self.agent_steps)
    

    def adjust_learning_rate_cos(self, epoch):
        lr = self.init_lr * 0.5 * (
            1. + math.cos(
                math.pi * (self.agent_steps + epoch / self.mini_epochs_num) / self.max_agent_steps))
        return lr


def policy_kl(p0_mu, p0_sigma, p1_mu, p1_sigma):
    c1 = torch.log(p1_sigma/p0_sigma + 1e-5)
    c2 = (p0_sigma ** 2 + (p1_mu - p0_mu) ** 2) / (2.0 * (p1_sigma ** 2 + 1e-5))
    c3 = -1.0 / 2.0
    kl = c1 + c2 + c3
    kl = kl.sum(dim=-1)  # returning mean between all steps of sum between all actions
    return kl.mean()


class AdaptiveScheduler(object):
    def __init__(self, kl_threshold=0.008):
        super().__init__()
        self.min_lr = 1e-6
        self.max_lr = 1e-2
        self.kl_threshold = kl_threshold

    def update(self, current_lr, kl_dist):
        lr = current_lr
        if kl_dist > (2.0 * self.kl_threshold):
            lr = max(current_lr / 1.5, self.min_lr)
        if kl_dist < (0.5 * self.kl_threshold):
            lr = min(current_lr * 1.5, self.max_lr)
        return lr


class LinearScheduler:
    def __init__(self, start_lr, max_steps=10000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)