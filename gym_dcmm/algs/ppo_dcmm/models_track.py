import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, units, input_size):#units是隐藏层，input_size为输入数据的维度。维度就是神经元的个数
        super(MLP, self).__init__()
        layers = []#用于存储网络层的列表。
        for output_size in units:#units有两个数，分别代表两个隐藏层的神经元个数
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size#表示下一层的输入维度等于当前层的输出维度，
            #一层隐藏层到下一层隐藏层就是通过权重矩阵和偏置变换过去的，下面哪个初始权重就是在初始权重矩阵
        self.mlp = nn.Sequential(*layers)#神经网络的封装，units 列表中有几个数，就代表这个神经网络有几层隐藏层

        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))#对self.mlp中所有全连接层的权重进行正交初始化，且每个隐藏层的初始化增益均为np.sqrt(2)，以优化网络的训练稳定性和收敛速度。
        #len(units)是2，这是因为输入层到第一层隐藏层会有一个权重矩阵，第一层隐藏层到第二个隐藏层也会有一个权重矩阵
    def forward(self, x):
        return self.mlp(x)#x是一个张量，这里是将张量输入到神经网络里面，输出也是一个张量

    @staticmethod
    def init_weights(sequential, scales):#scales = [√2, √2]
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]#只遍历 sequential 中的 nn.Linear 层，
        #跳过 ELU、ReLU 等激活层对神经网络的全连接层进行正交初始化
        #就是随机了一个正交矩阵然后把正交矩阵乘以根号2。正交矩阵的作用：在前向和反向传播中，尽量“不放大也不缩小信号”，让信息稳定地穿过多层网络。
        #√2 是用来补偿 ReLU / ELU 这种激活函数，在前向传播中“丢失一半信号”的问题，而输出层不用*根号2是因为输出层一般不接ReLU / ELU 这种激活函数
class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')#是否使用分离的价值网络” 的标志，布尔
        self.separate_value_mlp = separate_value_mlp#如果是True那么actor和critic不使用相同的网络训练

        actions_num = kwargs.pop('actions_num')# 动作空间的维度
        input_shape = kwargs.pop('input_shape')# 输入数据的形状，观测数据的形状
        self.units = kwargs.pop('actor_units')# 策略网络MLP的各层输出维度。#MLP 隐藏层
        mlp_input_shape = input_shape[0]# 提取输入数据的维度,输入数据是一维的，作为MLP的输入维度

        out_size = self.units[-1]#取最后一层的输出维度，输出的是某种意义上的action但是是一系列的动作特征，比如均值方差什么的

        self.actor_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        if self.separate_value_mlp:
            self.value_mlp = MLP(units=self.units, input_size=mlp_input_shape)#构建价值网络的 MLP，这里就是critic网络
        self.value = torch.nn.Linear(out_size, 1)# 价值输出层：将MLP特征映射为1维状态价值（V(s)）
        self.mu = torch.nn.Linear(out_size, actions_num)# 动作均值输出层：映射为动作空间维度的均值（μ）Actor 的输出层
        #输出的正是每个动作维度对应的具体动作均值，这里是策略网络的均值输出层
        self.sigma = nn.Parameter(
            torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
        #定义策略网络的标准差参数，sigma越大，动作的可探索性越大
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)#初始化网络参数，全连接层：偏置初始化为 0，用正态分布初始化权重，均值为0，方差为fan_out
        nn.init.constant_(self.sigma, 0)#将标准差参数self.sigma初始化为 0

        # policy output layer with scale 0.01
        # value output layer with scale 1
        torch.nn.init.orthogonal_(self.mu.weight, gain=0.01)
        #函数会直接生成一个与 self.mu.weight 形状匹配（如 18×128）的正交矩阵（满足行 / 列向量垂直且单位长度）。
        #将这个正交矩阵整体乘以 0.01，得到的结果就是 self.mu.weight 的初始值。
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
    
    def save_actor(self, actor_mlp_path, actor_head_path):
        """
        Save actor and critic model parameters to files.

        actor_path: Path to save actor model parameters.
        """
        torch.save(self.actor_mlp.state_dict(), actor_mlp_path)#保存MLP主体，后面是路径
        torch.save(self.mu.state_dict(), actor_head_path)#保存mu层，后面是路径

    @torch.no_grad()#禁用梯度计算，提高效率
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)#构建一个高斯分布，均值就是mu方差就是sigma，构建的高斯分布是多维的，维数就是action的维数
        selected_action = distr.sample()#从分布中采一个动作
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),#把刚刚采样得到的那个动作的概率求和动作维度求和把所有动作维度的 log 概率加起来，
            #是为#了得到“这一次采样得到的整个动作向量，在当前策略下出现的概率”。
            'values': value,#一步只会返回一个value。不管action和obs是几维
            'actions': selected_action,
            'mus': mu,
            'sigmas': sigma,
        }
        return result

    @torch.no_grad()
    def act_inference(self, obs_dict):
        # used for testing
        mu, logstd, value = self._actor_critic(obs_dict)
        return mu

    def _actor_critic(self, obs_dict):
        obs = obs_dict['obs']#这个键对应的值是环境观测的核心数据

        x = self.actor_mlp(obs)# 把obs放到神经网络里，x就是最后一层隐藏层的输出
        mu = self.mu(x)#动作分布的均值，但是不是直接算平均的，self.mu = torch.nn.Linear(out_size, actions_num)输出层计算动作均值（未归一化）
        if self.separate_value_mlp:#如果是true就代表actor和critic网络是分开训练的
            x = self.value_mlp(obs)# 若独立，则用 value_mlp 提取特征
        #self.value_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        value = self.value(x)#self.value = torch.nn.Linear(out_size, 1)输出层计算状态价值，如果我现在站在这里不动，再走下去估计能拿多少reward
        #self.value = torch.nn.Linear(out_size, 1)
        sigma = self.sigma
        # Normalize to (-1,1)
        mu = torch.tanh(mu)#压缩到[-1,1]，因为action要求在-1，1之间。并且如果是线性映射的化没法保证每个数都在-1，1之间，因为有的数可能很大
        return mu, mu * 0 + sigma, value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)#历史动作张量
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)#构建正态分布
        entropy = distr.entropy().sum(dim=-1)#对最后一个动作维度求和，计算分布的shang，熵越大说明动作越随机（探索性越强）
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)#计算历史动作的负对数概率
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result