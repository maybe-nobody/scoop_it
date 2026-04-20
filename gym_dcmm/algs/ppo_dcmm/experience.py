import torch
from torch.utils.data import Dataset


def transform_op(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()#获取该张量所有维度的形状信息
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])#交换张量的第 0 维和第 1 维，然后展成一维的


class ExperienceBuffer(Dataset):
    def __init__(
        self, num_envs, horizon_length, batch_size, minibatch_size, obs_dim, act_dim, device):
        self.device = device
        self.num_envs = num_envs#并行运行的环境数量，horizon_length每个环境运行的时间步数，batch_size运行时使用的总样本数=horizon_length * num_envs
        self.transitions_per_env = horizon_length

        self.data_dict = None
        self.obs_dim = obs_dim#观测维度
        self.act_dim = act_dim#动作维度
        self.storage_dict = {
            'obses': torch.zeros(#存储观测数据
                (self.transitions_per_env, self.num_envs, self.obs_dim),#每个环境收集的时间步数，并行运行的环境数量，单个观测的维数
                dtype=torch.float32, device=self.device),#指定存储设备
            'rewards': torch.zeros(#存储即时奖励
                (self.transitions_per_env, self.num_envs, 1),
                dtype=torch.float32, device=self.device),
            'values': torch.zeros(#存储状态价值估计
                (self.transitions_per_env, self.num_envs,  1),
                dtype=torch.float32, device=self.device),
            'neglogpacs': torch.zeros(#存储动作的负对数概率
                (self.transitions_per_env, self.num_envs),
                dtype=torch.float32, device=self.device),
            'dones': torch.zeros(#存储终止标志
                (self.transitions_per_env, self.num_envs),
                dtype=torch.uint8, device=self.device),
            'actions': torch.zeros(#存储智能体执行的动作
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            'mus': torch.zeros(#存储动作分布的均值
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            'sigmas': torch.zeros(#存储动作分布的标准差
                (self.transitions_per_env, self.num_envs, self.act_dim),
                dtype=torch.float32, device=self.device),
            'returns': torch.zeros(#存储折扣回报
                (self.transitions_per_env, self.num_envs,  1),
                dtype=torch.float32, device=self.device),
        }

        self.batch_size = batch_size#表示一次完整数据收集后，可用于训练的总样本数量。
        self.minibatch_size = minibatch_size#每次参数更新时实际使用的样本数量
        self.length = self.batch_size // self.minibatch_size#表示一个完整批次（batch_size）可以拆分成的小批量数量

    def __len__(self):
        return self.length#单轮训练内的小批量数量

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size#确定当前小批量数据在总样本中的起始和结束位置。
        self.last_range = (start, end)#记录当前小批量范围
        input_dict = {}#初始化小批量数据字典
        for k, v in self.data_dict.items():#返回键值对
            if type(v) is dict:
                v_dict = {kd: vd[start:end] for kd, vd in v.items()}
                input_dict[k] = v_dict
            else:
                input_dict[k] = v[start:end]
        return input_dict['values'], input_dict['neglogpacs'], input_dict['advantages'], \
            input_dict['mus'], input_dict['sigmas'], input_dict['returns'], input_dict['actions'], \
            input_dict['obses']

    def update_mu_sigma(self, mu, sigma):
        start = self.last_range[0]
        end = self.last_range[1]
        self.data_dict['mus'][start:end] = mu#存储动作分布的均值
        self.data_dict['sigmas'][start:end] = sigma#存储动作分布的标准差

    def update_data(self, name, index, val):
        if type(val) is dict:
            for k, v in val.items():
                self.storage_dict[name][k][index,:] = v
        else:
            self.storage_dict[name][index,:] = val

    def compute_return(self, last_values, gamma, tau):#gamma折扣因子（0<γ≤1），表示未来奖励的衰减程度（γ 越大，未来奖励越重要）。tau：GAE 的平滑参数（0<τ≤1），控制优势估计的偏差与方差平衡。
        last_gae_lam = 0#记录下一时刻的A（用于反向计算）
        mb_advs = torch.zeros_like(self.storage_dict['rewards'])# 存储优势函数的张量（形状与奖励相同），准备advantage
        for t in reversed(range(self.transitions_per_env)):#gae就是对td误差delta按时间进行加权求和，GAE 需要基于 “未来的奖励和价值” 计算当前优势，因此必须从后往前算。 self.transitions_per_env = horizon_length
            if t == self.transitions_per_env - 1:#如果到一个roll_out的最后一步，要用下一步的V就要用刚算出来的last_value来代替
                next_values = last_values
            else:
                next_values = self.storage_dict['values'][t + 1]
            next_nonterminal = 1.0 - self.storage_dict['dones'].float()[t]#判断未来是不是还存在
            next_nonterminal = next_nonterminal.unsqueeze(1)
            delta = self.storage_dict['rewards'][t] + \
                gamma * next_values * next_nonterminal - self.storage_dict['values'][t]#计算 TD 误差（delta，delta大于0就是实际比预期好
            mb_advs[t] = last_gae_lam = delta + gamma * tau * next_nonterminal * last_gae_lam#计算优势函数，tau = “我信未来 TD 误差信到多远”，gamma是未来奖励值多少钱，一个作用在误差身上，一个作用在reward身上
            self.storage_dict['returns'][t, :] = mb_advs[t] + self.storage_dict['values'][t]#计算折扣回报
            '''优势用于指导策略(Actor)更新,让智能体倾向于选择更好的动作；
            回报用于训练价值函数(Critic),提高价值估计的准确性
            '''
    def prepare_training(self):#将经验缓冲区（storage_dict）中的原始数据转换为适合模型训练的格式
        self.data_dict = {}
        for k, v in self.storage_dict.items():
            self.data_dict[k] = transform_op(v)
        advantages = self.data_dict['returns'] - self.data_dict['values']#计算优势函数返回的是一个张量
        self.data_dict['advantages'] = (
            (advantages - advantages.mean()) / (advantages.std() + 1e-8)).squeeze(1)#标准化优势函数
        return self.data_dict
