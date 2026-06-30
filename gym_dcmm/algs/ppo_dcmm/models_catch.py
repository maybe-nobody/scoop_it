import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
'''
神经网络：
MLP:多层感知机/全连接神经网络，特点是每层的神经元和前一层的神经元全连接，适合处理结构化数据。标准MLP的输入通常是一个一维向量
CNN:卷积神经网络，适合处理图片
'''

class MLP(nn.Module):#定义了一个神经网络，但是没有定义输出层，现在最后一层是64维的。actor和critic两个都要用
    def __init__(self, units, input_size):
        super(MLP, self).__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

        # orthogonal init of weights
        # hidden layers scale np.sqrt(2)
        self.init_weights(self.mlp, [np.sqrt(2)] * len(units))

    def forward(self, x):
        return self.mlp(x)

    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]


class ActorCritic(nn.Module):
    def __init__(self, kwargs):
        nn.Module.__init__(self)
        separate_value_mlp = kwargs.pop('separate_value_mlp')#kwrgs是一个变量名，接受从外面传进来的字典
        # net_config = {
        #     'actor_units': self.network_config.mlp.units,隐藏层维度
        #     'actions_num': self.actions_num,动作空间维度
        #     'input_shape': self.obs_shape,观察维度
        #     'separate_value_mlp': self.network_config.get('separate_value_mlp', True),critic/value网络是否使用单独的MLP
        # }
        self.separate_value_mlp = separate_value_mlp

        actions_num = kwargs.pop('actions_num')
        input_shape = kwargs.pop('input_shape')
        self.units = kwargs.pop('actor_units')
        mlp_input_shape = input_shape[0]

        out_size = self.units[-1]
        print("mlp_input_shape: ", mlp_input_shape)
        self.actor_mlp_t = MLP(units=self.units, input_size=mlp_input_shape-2)
        #self.actor_mlp_t = MLP(units=self.units, input_size=mlp_input_shape-12)
        self.actor_mlp_c = MLP(units=self.units, input_size=mlp_input_shape-3)
        #self.actor_mlp_c = MLP(units=self.units, input_size=mlp_input_shape-2)减去的是底座？
        # if self.separate_value_mlp:
        self.value_mlp = MLP(units=self.units, input_size=mlp_input_shape)
        self.value = torch.nn.Linear(out_size, 1)#定义critic网络
        self.mu_t = torch.nn.Linear(out_size, actions_num-2)#nn.Linear包含nn.Parameter这个功能
        self.mu_c = torch.nn.Linear(out_size, actions_num-7)##################################################原来是-6
        self.sigma_t = nn.Parameter(
            torch.zeros(actions_num-2, requires_grad=True, dtype=torch.float32), requires_grad=True)#定义actor输出动作分布的标准差分布
        #tensor不会自己更新自己，pytorch提供了计算梯度记录计算图，优化器更新参数的功能，里面这个torch可以计算梯度，但是不能自己更新自己
        self.sigma_c = nn.Parameter(
            torch.zeros(actions_num-7, requires_grad=True, dtype=torch.float32), requires_grad=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                fan_out = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
            if isinstance(m, nn.Linear):
                if getattr(m, 'bias', None) is not None:
                    torch.nn.init.zeros_(m.bias)
        nn.init.constant_(self.sigma_t, 0)
        nn.init.constant_(self.sigma_c, 0)#初始化数值
        # policy output layer with scale 0.01
        # value output layer with scale 1
        torch.nn.init.orthogonal_(self.mu_t.weight, gain=0.01)#把权重做正交初始化，缩放系数很小几乎等于0，缩放系数就是把初始化后中的矩阵系数乘以这个数值。先正交再乘增益
        #会使信号在传播的过程中缩小，但是初始就是要设置很小的动作，不让动作太大。偏置已经在前面初始化为0了
        torch.nn.init.orthogonal_(self.mu_c.weight, gain=0.01)
        torch.nn.init.orthogonal_(self.value.weight, gain=1.0)
        #nn.Linear(64, 8)  y = Wx + b W 是 8×64 的矩阵，b就是一个八维的。正交是不同向量之间内积为0，不管是行向量还是纵向量。正交矩阵的作用是信号传播不会突然放大或者缩小

    @torch.no_grad()
    def act(self, obs_dict):
        # used specifically to collection samples during training
        # it contains exploration so needs to sample from distribution
        mu, logstd, value = self._actor_critic(obs_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        selected_action = distr.sample()
        result = {
            'neglogpacs': -distr.log_prob(selected_action).sum(1),
            'values': value,
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
        obs = obs_dict['obs']
        obs_t = obs_dict['obs_t']
        obs_c = obs_dict['obs_c']

        x_t = self.actor_mlp_t(obs_t)
        x_c = self.actor_mlp_c(obs_c)
        mu_t = self.mu_t(x_t)
        mu_c = self.mu_c(x_c)
        # if self.separate_value_mlp:
        x = self.value_mlp(obs)
        value = self.value(x)

        sigma_t = self.sigma_t
        sigma_c = self.sigma_c
        # Normalize to (-1,1)
        mu_t = torch.tanh(mu_t)
        mu_c = torch.tanh(mu_c)

        # Concatenate mu_t and mu_c
        mu = torch.cat((mu_t, mu_c), dim=1)

        return mu, torch.cat((mu_t * 0 + sigma_t, mu_c * 0 + sigma_c), dim=1), value

    def forward(self, input_dict):
        prev_actions = input_dict.get('prev_actions', None)
        mu, logstd, value = self._actor_critic(input_dict)
        sigma = torch.exp(logstd)
        distr = torch.distributions.Normal(mu, sigma)
        entropy = distr.entropy().sum(dim=-1)
        prev_neglogp = -distr.log_prob(prev_actions).sum(1)
        result = {
            'prev_neglogp': torch.squeeze(prev_neglogp),
            'values': value,
            'entropy': entropy,
            'mus': mu,
            'sigmas': sigma,
        }
        return result