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
'''
PPO = PPO_Track if config.task == 'Tracking' else \
          PPO_Catch_TwoStage if config.task == 'Catching_TwoStage' else \
          PPO_Catch_OneStage
'''
"""actor损失 优势函数(真实值-预测值)*新旧策略比   优势函数是正数代表做完这个动作的时候比预期的要好,说明动作好,新旧策略比如果大于1的话也说明这个动作好
critic损失 (预测值-真实值)^2"""
class PPO_Track(object):
    def __init__(self, env, output_dif, full_config):
        self.rank = -1
        self.device = full_config['rl_device']
        self.network_config = full_config.train.network
        self.ppo_config = full_config.train.ppo
        # ---- build environment ----
        self.env = env
        self.num_actors = int(self.ppo_config['num_actors'])#并行环境数量
        print("num_actors: ", self.num_actors)
        self.actions_num = self.env.call("act_t_dim")[0]
        print("actions_num: ", self.actions_num)
        self.actions_low = self.env.call("actions_low")[0]
        self.actions_high = self.env.call("actions_high")[0]
        # self.obs_shape = self.env.observation_space.shape
        self.obs_shape = (self.env.call("obs_t_dim")[0],)
        print("obs_shaqpe:",self.obs_shape)
        self.full_action_dim = self.env.call("act_c_dim")[0]
        # ---- Model ----
        net_config = {
            'actor_units': self.network_config.mlp.units,#MLP 隐藏层
            'actions_num': self.actions_num,
            'input_shape': self.obs_shape,
            'separate_value_mlp': self.network_config.get('separate_value_mlp', True),#“Actor 和 Critic 用两套不同的 MLP”
        }
        print("net_config: ", net_config)
        self.model = ActorCritic(net_config)#创建神经网络，初始化权重矩阵
        self.model.to(self.device)
        self.running_mean_std = RunningMeanStd(self.obs_shape).to(self.device)#观测归一化，输入一个，然后把之前的和输入的放在一起算均值和方差，然后再根据均值和方差对输入进来的东西做归一化
        self.value_mean_std = RunningMeanStd((1,)).to(self.device)#对 Critic 输出的 value（标量）做 running mean / std 归一化，
        # ---- Output Dir ----
        # allows us to specify a folder where all experiments will reside
        self.output_dir = output_dif
        self.nn_dir = os.path.join(self.output_dir, 'nn')
        self.tb_dif = os.path.join(self.output_dir, 'tb')
        os.makedirs(self.nn_dir, exist_ok=True)
        os.makedirs(self.tb_dif, exist_ok=True)
        # ---- Optim ----
        self.init_lr = float(self.ppo_config['learning_rate'])
        self.last_lr = float(self.ppo_config['learning_rate'])#学习率每次梯度更新时，参数走多大一步，梯度大的话步子就小，梯度小步子大
        #参数主要是权重矩阵和偏置，损失函数（loss）对这些参数的偏导数。梯度 = “参数往哪个方向改，loss 会下降得最快”
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), self.init_lr, eps=1e-5)#是一个优化器，只需要知道梯度就可以自己修改参数
        # ---- PPO Train Param ----
        self.e_clip = self.ppo_config['e_clip']#策略裁减系数，新旧策略变化不能太大
        self.action_track_denorm = self.ppo_config['action_track_denorm']
        self.action_catch_denorm = self.ppo_config['action_catch_denorm']
        self.clip_value = self.ppo_config['clip_value']#防止critic跳变太快
        self.entropy_coef = self.ppo_config['entropy_coef']
        self.critic_coef = self.ppo_config['critic_coef']
        self.bounds_loss_coef = self.ppo_config['bounds_loss_coef']
        self.gamma = self.ppo_config['gamma']
        self.tau = self.ppo_config['tau']
        self.truncate_grads = self.ppo_config['truncate_grads']
        self.grad_norm = self.ppo_config['grad_norm']
        self.value_bootstrap = self.ppo_config['value_bootstrap']
        self.normalize_advantage = self.ppo_config['normalize_advantage']
        self.normalize_input = self.ppo_config['normalize_input']
        self.normalize_value = self.ppo_config['normalize_value']
        self.reward_scale_value = self.ppo_config['reward_scale_value']
        self.clip_value_loss = self.ppo_config['clip_value_loss']
        # ---- PPO Collect Param ----
        self.horizon_length = self.ppo_config['horizon_length']#每次更新策略要走多少个策略步
        self.batch_size = self.horizon_length * self.num_actors#更新一次用的样本数，各个环境里面用策略都是相同的
        self.minibatch_size = self.ppo_config['minibatch_size']#把所有顺序全都shuffle（打乱），然后按顺序一批一批提取直到所有都训练完 
        self.mini_epochs_num = self.ppo_config['mini_epochs']#同一批数据被反复训练几次，重复进行训练，数据都是之前的策略产生的，但是进行重复训练修改策，一个mini_batch修改一次策略
        #略然后把新策略下产生这个动作的概率除以旧策略下产生这个动作的概率。获取数据和修改参数再时间上是完全分开交替进行的
        #assert self.batch_size % self.minibatch_size == 0 or full_config.test
        # ---- scheduler ----
        self.lr_schedule = self.ppo_config['lr_schedule']
        if self.lr_schedule == 'kl':
            self.kl_threshold = self.ppo_config['kl_threshold']
            self.scheduler = AdaptiveScheduler(self.kl_threshold)
        elif self.lr_schedule == 'linear':
            self.scheduler = LinearScheduler(
                self.init_lr,
                self.ppo_config['max_agent_steps'])
        # ---- Snapshot
        self.save_freq = self.ppo_config['save_frequency']
        self.save_best_after = self.ppo_config['save_best_after']
        # ---- Tensorboard Logger ----
        self.extra_info = {}
        writer = SummaryWriter(self.tb_dif)
        self.writer = writer

        self.episode_rewards = AverageScalarMeter(200)
        self.episode_lengths = AverageScalarMeter(200)
        self.episode_success = AverageScalarMeter(200)#上面这三个是训练过程中的在线统计
        self.episode_test_rewards = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_lengths = AverageScalarMeter(self.ppo_config['test_num_episodes'])
        self.episode_test_success = AverageScalarMeter(self.ppo_config['test_num_episodes'])#测试过程中的统计

        self.obs = None
        self.epoch_num = 0#ppo参数update的次数
        # print("self.obs_shape[0]: ", type(self.obs_shape[0]))
        self.storage = ExperienceBuffer(
            self.num_actors, self.horizon_length, self.batch_size, self.minibatch_size,
            self.obs_shape[0], self.actions_num, self.device,
        )

        batch_size = self.num_actors
        current_rewards_shape = (batch_size, 1)
        self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.device)
        self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.device)
        self.agent_steps = 0
        self.max_agent_steps = self.ppo_config['max_agent_steps']
        self.max_test_steps = self.ppo_config['max_test_steps']
        self.best_rewards = -10000
        self.best_success = -1.0
        # ---- Timing
        self.data_collect_time = 0
        self.rl_train_time = 0
        self.all_time = 0


        self.debug_prev_mu_vector = None

    def write_stats(self, a_losses, c_losses, b_losses, entropies, kls):
        log_dict = {
            'performance/RLTrainFPS': self.agent_steps / self.rl_train_time,#gpu更新权重的速度
            'performance/EnvStepFPS': self.agent_steps / self.data_collect_time,#衡量mujoco方仿真和数据采集的速度
            'losses/actor_loss': torch.mean(torch.stack(a_losses)).item(),
            'losses/bounds_loss': torch.mean(torch.stack(b_losses)).item(),
            'losses/critic_loss': torch.mean(torch.stack(c_losses)).item(),
            'losses/entropy': torch.mean(torch.stack(entropies)).item(),
            'info/last_lr': self.last_lr,
            'info/e_clip': self.e_clip,
            'info/kl': torch.mean(torch.stack(kls)).item(),
        }
        for k, v in self.extra_info.items():
            log_dict[f'{k}'] = v

        # log to wandb
        wandb.log(log_dict, step=self.agent_steps)

        # log to tensorboard
        for k, v in log_dict.items():
            self.writer.add_scalar(k, v, self.agent_steps)

    def set_eval(self):
        self.model.eval()#self.model = ActorCritic(net_config)这个类里面没有eval，但是继承了nn.module,因此也可以用eval
        if self.normalize_input:#self.normalize_input = self.ppo_config['normalize_input']
            self.running_mean_std.eval()
        if self.normalize_value:
            self.value_mean_std.eval()#
        '''  normalize_input: True
        #是否对输入的观测数据（如状态、图像特征）进行标准化处理
        normalize_value: True
        #是否对 Critic 网络输出的状态价值(Value)进行标准化处理
  #     '''

    def set_train(self):
        self.model.train()#设置为训练模式
        if self.normalize_input:
            self.running_mean_std.train()
        if self.normalize_value:
            self.value_mean_std.train()

    def train(self):
        start_time = time.time()
        _t = time.time()
        _last_t = time.time()#获得此刻的时间
        reset_obs, _ = self.env.reset()#在一个episode开始的时候reset一次
        self.obs = {'obs': self.obs2tensor(reset_obs)}#转成tensor
        self.agent_steps = self.batch_size#ppo一次更新用的样本数
        #统计的是「已经收集并用于训练的样本数」，在进入while循环之前，第一次的数据已经手机好了
        while self.agent_steps < self.max_agent_steps:
            self.epoch_num += 1#+1是指经过了一个batch
            a_losses, c_losses, b_losses, entropies, kls = self.train_epoch()
            with torch.no_grad():
                mu_vector = torch.cat([
                    p.detach().flatten()
                    for p in self.model.mu.parameters()
                ])

                if self.debug_prev_mu_vector is None:
                    self.debug_prev_mu_vector = mu_vector.clone()
                else:
                    mu_delta = (mu_vector - self.debug_prev_mu_vector).abs().mean().item()
                    self.debug_prev_mu_vector = mu_vector.clone()

                    # if self.epoch_num % 10 == 0:
                    #     mean_kl = torch.mean(torch.stack(kls)).item()
                    #     mean_entropy = torch.mean(torch.stack(entropies)).item()

                    #     print(
                    #         "[PARAM_UPDATE_DEBUG] "
                    #         f"epoch={self.epoch_num}, "
                    #         f"mu_delta_per_epoch={mu_delta:.8f}, "
                    #         f"last_lr={self.last_lr:.8f}, "
                    #         f"mean_kl={mean_kl:.8f}, "
                    #         f"mean_entropy={mean_entropy:.6f}"
                    #     )
            self.storage.data_dict = None

            if self.lr_schedule == 'linear':
                self.last_lr = self.scheduler.update(self.agent_steps)
            
            all_fps = self.agent_steps / (time.time() - _t)
            last_fps = (
                self.batch_size
            ) / (time.time() - _last_t)
            _last_t = time.time()

            # ======================================================
            # 先计算 mean_rewards / mean_lengths / mean_success
            # 后面 info_string、tensorboard、wandb、保存 best 都要用它们
            # ======================================================
            mean_rewards = self.episode_rewards.get_mean()
            mean_lengths = self.episode_lengths.get_mean()
            mean_success = self.episode_success.get_mean()

            info_string = (
                f'Agent Steps: {int(self.agent_steps // 1e3):04}K | '
                f'FPS: {all_fps:.1f} | '
                f'Last FPS: {last_fps:.1f} | '
                f'Collect Time: {self.data_collect_time / 60:.1f} min | '
                f'Train RL Time: {self.rl_train_time / 60:.1f} min | '
                f'Mean Reward: {mean_rewards:.2f} | '
                f'Mean Length: {mean_lengths:.2f} | '
                f'Mean Success: {mean_success:.3f} | '
                f'Best Reward: {self.best_rewards:.2f} | '
                f'Best Success: {self.best_success:.3f}'
            )
            print(info_string)

            self.write_stats(a_losses, c_losses, b_losses, entropies, kls)
            # print("mean_rewards: ", mean_rewards)
            self.writer.add_scalar(
                'metrics/episode_rewards_per_step', mean_rewards, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_lengths_per_step', mean_lengths, self.agent_steps)
            self.writer.add_scalar(
                'metrics/episode_success_per_step', mean_success, self.agent_steps)
            wandb.log({
                'metrics/episode_rewards_per_step': mean_rewards,
                'metrics/episode_lengths_per_step': mean_lengths,
                'metrics/episode_success_per_step': mean_success,
            }, step=self.agent_steps)
            checkpoint_name = f'ep_{self.epoch_num}_step_{int(self.agent_steps // 1e6):04}m_reward_{mean_rewards:.2f}'

            if self.save_freq > 0:
                if (self.epoch_num % self.save_freq == 0) and (mean_rewards <= self.best_rewards):
                    self.save(os.path.join(self.nn_dir, checkpoint_name))
                self.save(os.path.join(self.nn_dir, f'last'))#每一轮 PPO 更新后都会保存 last.pth

            if mean_rewards > self.best_rewards:
                print(f'save current best reward: {mean_rewards:.2f}')
                # remove previous best file
                prev_best_ckpt = os.path.join(self.nn_dir, f'best_reward_{self.best_rewards:.2f}.pth')
                if os.path.exists(prev_best_ckpt):
                    os.remove(prev_best_ckpt)
                self.best_rewards = mean_rewards
                self.save(os.path.join(self.nn_dir, f'best_reward_{mean_rewards:.2f}'))
            if mean_success > self.best_success and mean_success > 0.0:
                print(f'save current best success: {mean_success:.3f}')

                prev_best_success_ckpt = os.path.join(
                    self.nn_dir,
                    f'best_success_{self.best_success:.3f}.pth'
                )

                if os.path.exists(prev_best_success_ckpt):
                    os.remove(prev_best_success_ckpt)

                self.best_success = mean_success
                self.save(os.path.join(self.nn_dir, f'best_success_{mean_success:.3f}'))

        print('max steps achieved')
        print('data collect time: %f min' % (self.data_collect_time / 60.0))
        print('rl train time: %f min' % (self.rl_train_time / 60.0))
        print('all time: %f min' % ((time.time() - start_time) / 60.0))

    def save(self, name):
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
    def restore_train(self, fn):
        """
        按照 two-stage 的完整 checkpoint 继续训练方式加载。

        加载内容：
            1. model
            2. running_mean_std
            3. value_mean_std

        注意：
            不额外修改 sigma。
            checkpoint 里保存的 sigma 是多少，就继续用多少。
        """

        if not fn:
            return

        checkpoint = torch.load(fn, map_location=self.device)

        # 1. 完整加载模型参数
        self.model.load_state_dict(checkpoint["model"], strict=True)
        print("[TRAIN LOADED] model with strict=True")

        # 2. 完整加载 obs 归一化参数
        if self.normalize_input and ("running_mean_std" in checkpoint):
            self.running_mean_std.load_state_dict(
                checkpoint["running_mean_std"],
                strict=True
            )
            print("[TRAIN LOADED] running_mean_std")

        # 3. 完整加载 value 归一化参数
        if self.normalize_value and ("value_mean_std" in checkpoint):
            self.value_mean_std.load_state_dict(
                checkpoint["value_mean_std"],
                strict=True
            )
            print("[TRAIN LOADED] value_mean_std")

        with torch.no_grad():
            print(
                "[TRAIN LOADED SIGMA] "
                f"logstd_mean={self.model.sigma.data.mean().item():.6f}, "
                f"logstd_min={self.model.sigma.data.min().item():.6f}, "
                f"logstd_max={self.model.sigma.data.max().item():.6f}, "
                f"actual_sigma_mean={torch.exp(self.model.sigma.data).mean().item():.6f}, "
                f"actual_sigma_min={torch.exp(self.model.sigma.data).min().item():.6f}, "
                f"actual_sigma_max={torch.exp(self.model.sigma.data).max().item():.6f}"
            )

        self.set_train()
        print("========== Restore Train Done ==========")
    def restore_actor_from_tracking(self, fn):
        """
        按照 ppo_dcmm_catch_two_stage.py 的方式加载 tracking checkpoint。

        加载内容：
            1. tracking_mlp    -> self.model.actor_mlp
            2. tracking_mu     -> self.model.mu
            3. tracking_sigma  -> self.model.sigma
            4. running_mean_std -> self.running_mean_std

        不加载：
            1. critic / value_mlp
            2. value_mean_std

        注意：
            这里不重置 mu。
            这里不重置 sigma。
            这里不强制把 sigma 改成 -0.5。
        """

        if not fn:
            return

        checkpoint = torch.load(fn, map_location=self.device)

        # 1. 加载 tracking actor_mlp
        if "tracking_mlp" in checkpoint:
            self.model.actor_mlp.load_state_dict(
                checkpoint["tracking_mlp"],
                strict=True
            )
            print("[TWOSTAGE STYLE LOADED] tracking_mlp -> actor_mlp")
        else:
            raise KeyError("checkpoint does not contain tracking_mlp")

        # 2. 加载 tracking_mu
        if "tracking_mu" in checkpoint:
            self.model.mu.load_state_dict(
                checkpoint["tracking_mu"],
                strict=True
            )
            print("[TWOSTAGE STYLE LOADED] tracking_mu -> mu")
        else:
            raise KeyError("checkpoint does not contain tracking_mu")

        # 3. 加载 tracking_sigma
        if "tracking_sigma" in checkpoint:
            self.model.sigma.data.copy_(checkpoint["tracking_sigma"])
            print("[TWOSTAGE STYLE LOADED] tracking_sigma -> sigma")
        else:
            raise KeyError("checkpoint does not contain tracking_sigma")

        # 4. 加载 running_mean_std
        if self.normalize_input and ("running_mean_std" in checkpoint):
            self.running_mean_std.load_state_dict(
                checkpoint["running_mean_std"],
                strict=True
            )
            print("[TWOSTAGE STYLE LOADED] running_mean_std")
        else:
            print("[TWOSTAGE STYLE WARNING] running_mean_std not loaded")

        # 5. 不加载 value_mean_std / critic
        print("[TWOSTAGE STYLE SKIP] value_mean_std / critic")

        with torch.no_grad():
            print(
                "[TWOSTAGE STYLE SIGMA] "
                f"logstd_mean={self.model.sigma.data.mean().item():.6f}, "
                f"logstd_min={self.model.sigma.data.min().item():.6f}, "
                f"logstd_max={self.model.sigma.data.max().item():.6f}, "
                f"actual_sigma_mean={torch.exp(self.model.sigma.data).mean().item():.6f}, "
                f"actual_sigma_min={torch.exp(self.model.sigma.data).min().item():.6f}, "
                f"actual_sigma_max={torch.exp(self.model.sigma.data).max().item():.6f}"
            )

        self.set_train()
        print("========== Restore Actor From Tracking Done: TwoStage Style ==========")
    def restore_like_twostage(self, checkpoint_tracking=None, checkpoint_catching=None):
        """
        模仿 ppo_dcmm_catch_two_stage.py 的 checkpoint 加载优先级。

        优先级：
            1. 如果 checkpoint_catching 有值：
                  加载完整 catching checkpoint
                  不加载 checkpoint_tracking

            2. 如果 checkpoint_catching 为空，但 checkpoint_tracking 有值：
                  加载 tracking_mlp
                  加载 tracking_mu
                  加载 tracking_sigma
                  加载 running_mean_std

            3. 两个都没有：
                  从头训练
        """

        if checkpoint_catching:
            print("[TWOSTAGE STYLE] checkpoint_catching is provided.")
            print("[TWOSTAGE STYLE] load catching checkpoint and skip tracking checkpoint.")
            self.restore_train(checkpoint_catching)
            return

        if checkpoint_tracking:
            print("[TWOSTAGE STYLE] checkpoint_catching is empty.")
            print("[TWOSTAGE STYLE] load tracking checkpoint.")
            self.restore_actor_from_tracking(checkpoint_tracking)
            return

        print("[TWOSTAGE STYLE] no checkpoint loaded, train from scratch.")
    def restore_test(self, fn):
        """
        测试时应该完整加载 checkpoint。

        注意：
        这里不跳过 sigma；
        不跳过 running_mean_std；
        不做 compatible_obs_dim 截断；
        要求当前模型结构和 checkpoint 完全一致。
        """
        if not fn:
            return

        checkpoint = torch.load(fn, map_location=self.device)

        # 1. 完整加载模型参数
        self.model.load_state_dict(checkpoint["model"], strict=True)
        print("[TEST LOADED] model with strict=True")

        # 2. 完整加载 obs normalization
        if self.normalize_input and ("running_mean_std" in checkpoint):
            self.running_mean_std.load_state_dict(checkpoint["running_mean_std"])
            print("[TEST LOADED] running_mean_std")

        # 3. 完整加载 value normalization
        if self.normalize_value and ("value_mean_std" in checkpoint):
            self.value_mean_std.load_state_dict(checkpoint["value_mean_std"])
            print("[TEST LOADED] value_mean_std")

        # 4. 切换 eval
        self.set_eval()

    def train_epoch(self):#这个函数到结束完成了就数据的收集并对数据进行训练修改了网络的参数
        # collect minibatch data
        _t = time.time()
        self.set_eval()
        self.play_steps()
        self.data_collect_time += (time.time() - _t)
        # update network
        _t = time.time()
        self.set_train()
        a_losses, b_losses, c_losses = [], [], []
        entropies, kls = [], []
        for mini_ep in range(0, self.mini_epochs_num):#同一批数据被反复训练几次
            ep_kls = []
            for i in range(len(self.storage)):#len得到的是单个环境的步数
                value_preds, old_action_log_probs, advantage, old_mu, old_sigma, \
                    returns, actions, obs = self.storage[i]#这里的bos就是这个时刻的obs，不是新的obs

                obs = self.running_mean_std(obs)
                batch_dict = {
                    'prev_actions': actions,
                    'obs': obs,
                }
                res_dict = self.model(batch_dict)#在第一次循环的时候用的是同一个网络，但是在以后的循环里用的就是新网络
                action_log_probs = res_dict['prev_neglogp']
                values = res_dict['values']
                entropy = res_dict['entropy']
                mu = res_dict['mus']
                sigma = res_dict['sigmas']#在这个循环开始的第一次其实还没有更新策略

                # actor loss
                ratio = torch.exp(old_action_log_probs - action_log_probs)#把所有维度取的动作的概率加起来，新策略比旧策略更（或更不）喜欢这个动作多少
                surr1 = advantage * ratio#advantage表示这个动作比“平均水平”好还是坏，adv这个动作好不好。ratio新策略对于这个动作的态度
                surr2 = advantage * torch.clamp(ratio, 1.0 - self.e_clip, 1.0 + self.e_clip)#如果更新太猛，我就把更新幅度剪掉，、adv的正负号决定往哪个方向改
                a_loss = torch.max(-surr1, -surr2)#想得到最小值就加-得到最大值，防止策略变化太大
                # critic loss
                if self.clip_value_loss:
                    value_pred_clipped = value_preds + \
                        (values - value_preds).clamp(-self.e_clip, self.e_clip)#不允许 value 一次改变超过 ±ε，新网络的v-旧网络的v
                    value_losses = (values - returns) ** 2#新网络的value-旧网络的return
                    value_losses_clipped = (value_pred_clipped - returns) ** 2
                    c_loss = torch.max(value_losses, value_losses_clipped)
                else:
                    c_loss = (values - returns) ** 2
                # bounded loss
                if self.bounds_loss_coef > 0:
                    soft_bound = 1.1
                    mu_loss_high = torch.clamp_min(mu - soft_bound, 0.0) ** 2
                    mu_loss_low = torch.clamp_max(mu + soft_bound, 0.0) ** 2
                    b_loss = (mu_loss_low + mu_loss_high).sum(axis=-1)
                else:
                    b_loss = 0
                a_loss, c_loss, entropy, b_loss = [
                    torch.mean(loss) for loss in [a_loss, c_loss, entropy, b_loss]]

                loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef \
                    + b_loss * self.bounds_loss_coef

                self.optimizer.zero_grad()#把 上一轮反向传播累积在参数上的梯度清零
                loss.backward(retain_graph=True)#它从 loss 开始，沿着神经网络倒着走，计算每一个参数（权重）对误差贡献了多少。计算结果存放在 parameter.grad 里。

                if self.truncate_grads:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)#把梯度限制在一个范围内

                self.optimizer.step()#使用刚刚算好的梯度，更新参数

                # ==========================================================
                # Catching_Finetune 阶段限制 sigma，防止 optimizer 又把 sigma 学回 0 附近
                #
                # 限制范围：
                #   sigma_param ∈ [-4.0, -2.0]
                #   actual_sigma ∈ [exp(-4), exp(-2)]
                #               ≈ [0.018, 0.135]
                #
                # 这样训练动作仍有随机探索，但不会严重破坏测试时已经会接的均值动作。
                # ==========================================================
                # try:
                #     task_name = str(self.env.call("task")[0])
                # except Exception:
                #     task_name = ""

                # if "Catching" in task_name:
                #     with torch.no_grad():
                #         self.model.sigma.data.clamp_(min=-2, max=-1.5)

                with torch.no_grad():
                    kl_dist = policy_kl(mu.detach(), sigma.detach(), old_mu, old_sigma)#不参与训练的情况下，计算新策略和旧策略之间的 KL 距离，用来监控 PPO 更新是不是走太远了
                kl = kl_dist
                a_losses.append(a_loss)#就是刚才求出来的a_loss，a_losses是一个列表，就是放在列表里
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.storage.update_mu_sigma(mu.detach(), sigma.detach())#把当前网络的mu和sigma存起来作为下回合的旧策略用
            #print(f"DEBUG: len(ep_kls) = {len(ep_kls)}")
            # if len(ep_kls) > 0:
            #     av_kls = torch.mean(torch.stack(ep_kls))
            # else:
            #     print("!!! CRITICAL: No KL data collected in this epoch !!!")
            #     av_kls = torch.tensor(0.0, device=self.device)
            #kl是根据状态进行调节，如果新旧变化太大的话就小一点，lr是线性调节，一开始大到最后变为0，cos类似于线性但是是按照余弦函数的趋势一样调节学习率的 
            av_kls = torch.mean(torch.stack(ep_kls))
            kls.append(av_kls)

            if self.lr_schedule == 'kl':
                self.last_lr = self.scheduler.update(self.last_lr, av_kls.item())
            elif self.lr_schedule == 'cos':
                self.last_lr = self.adjust_learning_rate_cos(mini_ep)#调整学习率

            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.last_lr

        self.rl_train_time += (time.time() - _t)
        return a_losses, c_losses, b_losses, entropies, kls
    
    def obs2tensor(self, obs):
        obs_array = np.concatenate((
            obs["base1"]["v_lin_3d"],
            obs["base1"]["base_pos"],
            obs["base2"]["v_lin_3d"],
            obs["base2"]["base_pos"],

            obs["arm1"]["ee_pos3d"],
            obs["arm1"]["ee_quat"],
            obs["arm1"]["ee_v_lin_3d"],
            obs["arm1"]["joint_pos"],

            obs["arm2"]["ee_pos3d"],
            obs["arm2"]["ee_quat"],
            obs["arm2"]["ee_v_lin_3d"],
            obs["arm2"]["joint_pos"],

            obs["object"]["pos3d"],
            obs["object"]["v_lin_3d"],

            obs["plate"]["plate_pos"],
        ), axis=1)

        obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
        return obs_tensor
    # def obs2tensor(self, obs):
    #     # Map the step result to tensor
    #     if self.env.call('task')[0] == 'Catching':
    #         obs_array = np.concatenate((
    #                     obs["base"]["v_lin_2d"], 
    #                     obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],
    #                     obs["object"]["pos3d"], obs["object"]["v_lin_3d"], 
    #                     obs["hand"],
    #                     ), axis=1)
    #     else:
    #         obs_array = np.concatenate((
    #                 obs["base"]["v_lin_2d"], 
    #                 obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],
    #                 obs["object"]["pos3d"], obs["object"]["v_lin_3d"],
    #                 # === 【新增下面这行】 ===
    #                 # 必须把 pos_history 拼接到输入张量里，否则网络接收不到这 9 个数
    #                 obs["object"]["pos_history"], # 9维
    #                 # ======================
    #                 # obs["hand"],# TODO: TEST
    #                 ), axis=1)#拼成一行
    #     obs_tensor = torch.tensor(obs_array, dtype=torch.float32).to(self.device)
    #     return obs_tensor
    def _to_1d_float_np(self, x):
        """
        把标量 / list / np.ndarray / torch.Tensor 统一转成 1D float32 numpy。
        适配单环境和多环境。
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 0:
            x = np.array([x.item()], dtype=np.float32)

        return x.reshape(-1)


    # def _extract_success_tensor(self, infos, truncates):
    #     """
    #     提取 episode 是否成功。

    #     优先级：
    #         1. infos["done_success"]：环境明确告诉 PPO 这个 done 是成功 done；
    #         2. infos["episode_success"]：环境明确告诉 PPO done episode 是否成功；
    #         3. infos["is_success"] 和 truncates 合并，作为兼容旧逻辑；
    #         4. 如果什么都没有，最后才退回 truncates。
    #     """

    #     trunc_np = self._to_1d_float_np(truncates)

    #     if isinstance(infos, dict) and ("done_success" in infos):
    #         success_np = self._to_1d_float_np(infos["done_success"])

    #     elif isinstance(infos, dict) and ("episode_success" in infos):
    #         success_np = self._to_1d_float_np(infos["episode_success"])

    #     elif isinstance(infos, dict) and ("is_success" in infos):
    #         info_success_np = self._to_1d_float_np(infos["is_success"])

    #         # 兼容旧环境：
    #         # 如果成功 done 用 truncated=True 表示，而 is_success 某些时候没对上，
    #         # 这里至少不会漏掉 truncated 成功。
    #         success_np = np.maximum(info_success_np, trunc_np)

    #     else:
    #         success_np = trunc_np

    #     success_tensor = torch.tensor(
    #         success_np,
    #         dtype=torch.float32,
    #         device=self.device
    #     )

    #     return success_tensor
    def _to_1d_float_np(self, x):
        """
        把标量 / list / np.ndarray / torch.Tensor 统一转成 1D float32 numpy。
        适配单环境和多环境。
        """
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()

        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 0:
            x = np.array([float(x.item())], dtype=np.float32)

        return x.reshape(-1).astype(np.float32)


    def _expand_to_num_actors(self, x_np):
        """
        保证 x_np 的 shape 是 [num_actors]。
        """
        x_np = np.asarray(x_np, dtype=np.float32).reshape(-1)

        if x_np.size == 1 and self.num_actors > 1:
            x_np = np.full(
                (self.num_actors,),
                float(x_np.item()),
                dtype=np.float32
            )

        if x_np.size != self.num_actors:
            print(
                "\n[WARNING SUCCESS SHAPE] "
                f"x_np.shape={x_np.shape}, "
                f"num_actors={self.num_actors}, "
                f"x_np={x_np}"
            )

            fixed = np.zeros((self.num_actors,), dtype=np.float32)
            copy_len = min(x_np.size, self.num_actors)
            fixed[:copy_len] = x_np[:copy_len]
            x_np = fixed

        return x_np


    def _extract_success_tensor(self, infos, truncates):
        """
        提取 episode 是否成功。

        关键：
            不要让 infos["done_success"] 的 0 覆盖 truncates 的 1。
            训练并行环境中，infos 可能和最后 done 帧不完全对齐，
            但 truncates 通常还能正确表示这个 env 刚刚成功结束。

        统计逻辑：
            success = max(truncates, done_success, episode_success, is_success)
        """

        # 1. 先用 truncates 作为基础成功信号
        trunc_np = self._expand_to_num_actors(
            self._to_1d_float_np(truncates)
        )

        success_np = trunc_np.copy()

        # 2. 合并 done_success
        if isinstance(infos, dict) and "done_success" in infos:
            done_success_np = self._expand_to_num_actors(
                self._to_1d_float_np(infos["done_success"])
            )
            success_np = np.maximum(success_np, done_success_np)

        # 3. 合并 episode_success
        if isinstance(infos, dict) and "episode_success" in infos:
            episode_success_np = self._expand_to_num_actors(
                self._to_1d_float_np(infos["episode_success"])
            )
            success_np = np.maximum(success_np, episode_success_np)

        # 4. 合并 is_success
        if isinstance(infos, dict) and "is_success" in infos:
            is_success_np = self._expand_to_num_actors(
                self._to_1d_float_np(infos["is_success"])
            )
            success_np = np.maximum(success_np, is_success_np)

        success_np = np.clip(success_np, 0.0, 1.0).astype(np.float32)

        return torch.tensor(
            success_np,
            dtype=torch.float32,
            device=self.device
        ).view(-1)
    def action2dict(self, actions):
        if torch.isnan(actions).any() or torch.isinf(actions).any():
            print("!!! 警告: 神经网络输出原始 actions 包含 NaN/Inf !!!")

        actions = actions.cpu().numpy()
        current_batch = actions.shape[0]

        # Tracking 和 Catching_Finetune 都使用同一套 10 维动作：
        # [base(2), base_copy(2), arm(3), arm_copy(3)]
        base_tensor = actions[:, 0:2] * self.action_track_denorm[0]
        base_tensor_copy = actions[:, 2:4] * self.action_track_denorm[0]

        arm_tensor = actions[:, 4:7] * self.action_track_denorm[1]
        arm_tensor_copy = actions[:, 7:10] * self.action_track_denorm[1]
        # ==========================================================
        # ACTION_DENORM_DEBUG
        # 目的：
        #   检查网络输出经过 clamp + denorm 后，真正送进环境的动作有多大。
        #
        # 如果 base_mean / arm_mean 很小：
        #   说明动作太小，盘子当然不靠近。
        #
        # 如果 raw_action 很大但 denorm 后还是小：
        #   说明 action_track_denorm 太小。
        # ==========================================================
        if not hasattr(self, "debug_action2dict_counter"):
            self.debug_action2dict_counter = 0

        self.debug_action2dict_counter += 1

        # 不要每一步都打印，否则日志太多。
        # 这里每 200 次 action2dict 打印一次。
        # if self.debug_action2dict_counter % 200 == 1:
        #     raw_base = actions[:, 0:2]
        #     raw_base_copy = actions[:, 2:4]
        #     raw_arm = actions[:, 4:7]
        #     raw_arm_copy = actions[:, 7:10]

        #     debug_raw_action_abs_mean = float(np.abs(actions[:, :10]).mean())
        #     debug_raw_action_abs_max = float(np.abs(actions[:, :10]).max())

        #     debug_base_mean = float(np.abs(base_tensor).mean())
        #     debug_base_max = float(np.abs(base_tensor).max())

        #     debug_base_copy_mean = float(np.abs(base_tensor_copy).mean())
        #     debug_base_copy_max = float(np.abs(base_tensor_copy).max())

        #     debug_arm_mean = float(np.abs(arm_tensor).mean())
        #     debug_arm_max = float(np.abs(arm_tensor).max())

        #     debug_arm_copy_mean = float(np.abs(arm_tensor_copy).mean())
        #     debug_arm_copy_max = float(np.abs(arm_tensor_copy).max())

        #     print(
        #         "[ACTION_DENORM_DEBUG] "
        #         f"raw_action_abs_mean={debug_raw_action_abs_mean:.4f}, "
        #         f"raw_action_abs_max={debug_raw_action_abs_max:.4f}, "
        #         f"base_mean={debug_base_mean:.4f}, "
        #         f"base_max={debug_base_max:.4f}, "
        #         f"base_copy_mean={debug_base_copy_mean:.4f}, "
        #         f"base_copy_max={debug_base_copy_max:.4f}, "
        #         f"arm_mean={debug_arm_mean:.4f}, "
        #         f"arm_max={debug_arm_max:.4f}, "
        #         f"arm_copy_mean={debug_arm_copy_mean:.4f}, "
        #         f"arm_copy_max={debug_arm_copy_max:.4f}"
        #     )

        #     # 也写进 extra_info，方便 wandb / tensorboard 看趋势
        #     self.extra_info["debug_raw_action_abs_mean"] = debug_raw_action_abs_mean
        #     self.extra_info["debug_raw_action_abs_max"] = debug_raw_action_abs_max
        #     self.extra_info["debug_denorm_base_mean"] = debug_base_mean
        #     self.extra_info["debug_denorm_base_max"] = debug_base_max
        #     self.extra_info["debug_denorm_base_copy_mean"] = debug_base_copy_mean
        #     self.extra_info["debug_denorm_base_copy_max"] = debug_base_copy_max
        #     self.extra_info["debug_denorm_arm_mean"] = debug_arm_mean
        #     self.extra_info["debug_denorm_arm_max"] = debug_arm_max
        #     self.extra_info["debug_denorm_arm_copy_mean"] = debug_arm_copy_mean
        #     self.extra_info["debug_denorm_arm_copy_max"] = debug_arm_copy_max
        # 当前任务是盘子接物体，不用手，hand 始终置零
        hand_tensor = np.zeros((current_batch, 2), dtype=np.float32)
        hand_tensor_copy = np.zeros((current_batch, 2), dtype=np.float32)

        actions_dict = {
            'arm': arm_tensor,
            'base': base_tensor,
            'hand': hand_tensor,
            'base_copy': base_tensor_copy,
            'arm_copy': arm_tensor_copy,
            'hand_copy': hand_tensor_copy,
        }

        return actions_dict
    # def action2dict(self, actions):
    #     actions = actions.cpu().numpy()
    #     # De-normalize the actions
    #     if self.env.call('task')[0] == 'Tracking':
    #         base_tensor = actions[:, :2] * self.action_track_denorm[0]#actions是个两维的，这是因为有多个并行的环境，第一个元素就是表示是第几个环境
    #         arm_tensor = actions[:, 2:5] * self.action_track_denorm[1]
    #         hand_tensor = actions[:, 5:] * self.action_track_denorm[2]
    #     else:
    #         base_tensor = actions[:, :2] * self.action_catch_denorm[0]
    #         arm_tensor = actions[:, 2:5] * self.action_catch_denorm[1]
    #         hand_tensor = actions[:, 5:] * self.action_catch_denorm[2]
    #     actions_dict = {
    #         'arm': arm_tensor,
    #         'base': base_tensor,
    #         'hand': hand_tensor
    #     }
    #     return actions_dict
    def model_act(self, obs_dict, inference=False):
        processed_obs = self.running_mean_std(obs_dict['obs'])#归一化，来一个obs就更新mean和var，然后根据更新的mean和var对obs进行归一化
        input_dict = {
            'obs': processed_obs,#只对当前输入的Obs的归一化
        }
        if not inference:#true就是test
            res_dict = self.model.act(input_dict)
            res_dict['values'] = self.value_mean_std(res_dict['values'], True)
        else:
            res_dict = {}
            res_dict['actions'] = self.model.act_inference(input_dict)
        return res_dict
        '''
            def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value = self._actor_critic(obs_dict)
        return mu
                result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),#把刚刚采样得到的那个动作的概率求和动作维度求和把所有动作维度的 log 概率加起来，
            #是为#了得到“这一次采样得到的整个动作向量，在当前策略下出现的概率”。
            'values': value,
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        '''
    def play_steps(self):
        # ======================================================
        # rollout 级别成功日志
        # 这些 key 不能用最后一步覆盖，否则成功那一帧会被后面的 0 覆盖。
        # 所以每个 rollout 开始先清零，后面用 max 记录。
        # ======================================================
        success_log_keys = {
            "is_success",
            "raw_success",
            "success_counter",
            "success_object_on_plate",
            "success_safe",
            "success_has_success",
            "success_bonus_given",
            "reward_object_on_plate",
            "reward_success",
            "done_success",
            "episode_success",
        }

        for key in success_log_keys:
            self.extra_info[key] = 0.0

        for n in range(self.horizon_length):#在一次 rollout 中，每个并行环境最多执行 horizon_length（一次 rollout 中，策略连续与环境交互的“最大步数”） 次 env.step()
            res_dict = self.model_act(self.obs)
            self.storage.update_data('obses', n, self.obs['obs'])
            for k in ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']:
                self.storage.update_data(k, n, res_dict[k])
            # Do env step
            # Clamp the actions of the action space
            actions = res_dict['actions']
            # ==========================================================
            # ACTION_BOUND_DEBUG：训练时检查采样动作 clamp 前是否越界
            # ==========================================================
            before_clamp = actions.detach().clone()

            clip_high_mask = before_clamp > 1.0
            clip_low_mask = before_clamp < -1.0
            clip_mask = clip_high_mask | clip_low_mask
            near_boundary_mask = (before_clamp > 0.95) | (before_clamp < -0.95)

            debug_action_clip_rate = float(clip_mask.float().mean().item())
            debug_action_near_boundary_rate = float(near_boundary_mask.float().mean().item())
            debug_action_before_clip_abs_mean = float(before_clamp.abs().mean().item())
            debug_action_before_clip_abs_max = float(before_clamp.abs().max().item())

            debug_base_clip_rate = float(clip_mask[:, 0:2].float().mean().item())
            debug_base_copy_clip_rate = float(clip_mask[:, 2:4].float().mean().item())
            debug_arm_clip_rate = float(clip_mask[:, 4:7].float().mean().item())
            debug_arm_copy_clip_rate = float(clip_mask[:, 7:10].float().mean().item())

            self.extra_info["debug_action_clip_rate"] = debug_action_clip_rate
            self.extra_info["debug_action_near_boundary_rate"] = debug_action_near_boundary_rate
            self.extra_info["debug_action_before_clip_abs_mean"] = debug_action_before_clip_abs_mean
            self.extra_info["debug_action_before_clip_abs_max"] = debug_action_before_clip_abs_max
            self.extra_info["debug_base_clip_rate"] = debug_base_clip_rate
            self.extra_info["debug_base_copy_clip_rate"] = debug_base_copy_clip_rate
            self.extra_info["debug_arm_clip_rate"] = debug_arm_clip_rate
            self.extra_info["debug_arm_copy_clip_rate"] = debug_arm_copy_clip_rate


            actions[:,:] = torch.clamp(actions[:,:], -1, 1)#防止数值越界
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)#需要补多少维度，补上的维度全部设置为0
            actions_dict = self.action2dict(actions)#反归一化
            for k, v in actions_dict.items():
                 if np.isnan(v).any() or np.isinf(v).any():
                     print(f"!!! 发现坏数据: 字典中的 {k} 包含 NaN/Inf")
            #print("actions_dict: ", actions_dict)
            obs, r, terminates, truncates, infos = self.env.step(actions_dict)
            #print("###############################ppo######################")
            #print(obs)
            # Map the obs
            self.obs = {'obs': self.obs2tensor(obs)}
            # Map the rewards
            r = torch.tensor(r, dtype=torch.float32).to(self.device)
            rewards = r.unsqueeze(1)#增加一个列维度，rewards.shape = [N, 1]
            # Map the dones
            
            # dones = terminates | truncates#如果既没终止也没截断，就会返回两个false，都是false游戏继续，按位进行或
            # self.dones = torch.tensor(dones, dtype=torch.uint8).to(self.device)#把dones变成0或1放到gpu上，看看环境是不是还活着
            terminates = np.asarray(terminates, dtype=bool).reshape(-1)
            truncates = np.asarray(truncates, dtype=bool).reshape(-1)
            dones = terminates | truncates

            self.dones = torch.tensor(
                dones,
                dtype=torch.uint8,
                device=self.device
            )
            # Update dones and rewards after env step
            self.storage.update_data('dones', n, self.dones)#把“这一刻是否结束”存进时间轴，n就是代表时间
            shaped_rewards = self.reward_scale_value * rewards.clone()#奖励缩放
            if self.value_bootstrap and 'time_outs' in infos:#  self.value_bootstrap = self.ppo_config['value_bootstrap']决定当一个episode因为时间；到了被叫停后要不要用critic的来补未来回报
                shaped_rewards += self.gamma * res_dict['values'] * infos['time_outs'].unsqueeze(1).float()#中断后把critic网络的v*gamma当作后续的reward   
            self.storage.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards#当前 episode 的累计 reward
            self.current_lengths += 1#当前episode的累积长度
            # print("self.dones: ", self.dones)
           
           
            done_indices = self.dones.nonzero(as_tuple=False).squeeze(-1)

            if done_indices.numel() > 0:
                self.episode_rewards.update(self.current_rewards[done_indices])
                self.episode_lengths.update(self.current_lengths[done_indices])

                # success_tensor = self._extract_success_tensor(infos, truncates)
                # self.episode_success.update(success_tensor[done_indices])
                success_tensor = self._extract_success_tensor(infos, truncates).view(-1)
                selected_success = success_tensor[done_indices].view(-1)

                self.episode_success.update(selected_success.detach().view(-1))

            assert isinstance(infos, dict), 'Info Should be a Dict'#isinstance(infos, dict)就是判断第一个是不是第二个，这里就是判断infos是不是dict，是的话就返回true
            # print("infos: ", infos)
            for k, v in infos.items():
                # ======================================================
                # 把不同类型的 info 统一转成 float，用于 TensorBoard
                # ======================================================
                if isinstance(v, float) or isinstance(v, int):
                    val = float(v)

                elif isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        val = float(v.item())
                    else:
                        val = float(v.float().mean().item())

                elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    val = float(np.mean(v))

                else:
                    continue

                # ======================================================
                # 成功类日志用 max，防止成功那一帧被后面的 0 覆盖
                # ======================================================
                if k in success_log_keys:
                    self.extra_info[k] = max(
                        float(self.extra_info.get(k, 0.0)),
                        val
                    )
                else:
                    self.extra_info[k] = val

            not_dones = 1.0 - self.dones.float()#把 done 掩码取反

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)#清零已结束环境的累计 reward
            self.current_lengths = self.current_lengths * not_dones#清零已结束环境的累计长度

        res_dict = self.model_act(self.obs)#有可能episode并没有走完，可能是rolout步数到了但是一个episode并没有走完，所以再让他走一步

        last_values = res_dict['values']

        self.agent_steps = (self.agent_steps + self.batch_size)#统计agent一共走了多少步
        self.storage.compute_return(last_values, self.gamma, self.tau)#计算return和advantage
        self.storage.prepare_training()#整理数据，准备进行ppo更新

        returns = self.storage.data_dict['returns']
        values = self.storage.data_dict['values']
        if self.normalize_value:
            self.value_mean_std.train()
            values = self.value_mean_std(values)
            returns = self.value_mean_std(returns)
            self.value_mean_std.eval()
        self.storage.data_dict['values'] = values
        self.storage.data_dict['returns'] = returns

    def play_test_steps(self):
        for _ in range(self.horizon_length):
            res_dict = self.model_act(self.obs, inference=True)
            # Do env step
            # Clamp the actions of the action space 
            actions = res_dict['actions']
            #print(f"DEBUG 1 - Raw Action from Model: {actions.abs().sum().item()}")
            actions[:,:] = torch.clamp(actions[:,:], -1, 1)
            #print(f"DEBUG 2 - Before Pad Shape: {actions.shape}")
            actions = torch.nn.functional.pad(actions, (0, self.full_action_dim-actions.size(1)), value=0)
            #print(f"DEBUG 3 - After Pad Shape: {actions.shape}")
            actions_dict = self.action2dict(actions)
            #print("actions_dict: ", actions_dict)
            #print("actions_dict: ", actions_dict)
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
            done_indices = self.dones.nonzero(as_tuple=False).squeeze(-1)

            if done_indices.numel() > 0:
                self.episode_test_rewards.update(self.current_rewards[done_indices])
                self.episode_test_lengths.update(self.current_lengths[done_indices])

                success_tensor = self._extract_success_tensor(infos, truncates)
                self.episode_test_success.update(success_tensor[done_indices])
            assert isinstance(infos, dict), 'Info Should be a Dict'
            for k, v in infos.items():
                if isinstance(v, float) or isinstance(v, int):
                    self.extra_info[k] = float(v)

                elif isinstance(v, torch.Tensor):
                    if v.numel() == 1:
                        self.extra_info[k] = float(v.item())
                    else:
                        self.extra_info[k] = float(v.float().mean().item())

                elif isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                    self.extra_info[k] = float(np.mean(v))

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
        
        res_dict = self.model_act(self.obs)
        self.agent_steps = (self.agent_steps + self.batch_size)
        self.test_steps += self.num_actors###############################################################后加的
    def test(self):
        self.set_eval()#把策略/价值网络切到测试模式
        reset_obs, _ = self.env.reset()
        self.obs = {'obs': self.obs2tensor(reset_obs)}
        '''
        if self.env.call('task')[0] == 'Catching':
            obs_array = np.concatenate((
                        obs["base"]["v_lin_2d"], 
                        obs["arm"]["ee_pos3d"], obs["arm"]["ee_quat"], obs["arm"]["ee_v_lin_3d"],
                        obs["object"]["pos3d"], obs["object"]["v_lin_3d"], 
                        obs["hand"],
                        ), axis=1)
                        '''
        self.test_steps = self.batch_size#一次 PPO 更新中，用来训练网络的“总样本数（transition 数）
        # self.batch_size = self.horizon_length * self.num_actors
        while self.test_steps < self.max_test_steps:# 这里的test_steps一直没有变       self.max_test_steps = self.ppo_config['max_test_steps']
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

class AdaptiveScheduler(object):#根据“新旧策略差异（KL 散度）大小，自动调大或调小学习率，
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
    def __init__(self, start_lr, max_steps=1000000):
        super().__init__()
        self.start_lr = start_lr
        self.min_lr = 1e-06
        self.max_steps = max_steps

    def update(self, steps):
        lr = self.start_lr - (self.start_lr * (steps / float(self.max_steps)))
        return max(self.min_lr, lr)
