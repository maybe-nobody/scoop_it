from __future__ import annotations
#进行训练的时候是python traim.py.........
import hydra
import torch
import os
import random
import wandb
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from gym_dcmm.utils.util import omegaconf_to_dict
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_two_stage import PPO_Catch_TwoStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_catch_one_stage import PPO_Catch_OneStage
from gym_dcmm.algs.ppo_dcmm.ppo_dcmm_track import PPO_Track
import gymnasium as gym
import gym_dcmm
import datetime
import pytz
# os.environ['MUJOCO_GL'] = 'egl'
OmegaConf.register_new_resolver('resolve_default', lambda default, arg: default if arg=='' else arg)
#使用 OmegaConf 库注册了一个名为 resolve_default 的自定义解析器。接受两个参数default和arg，如果arg是空字符，那么返回default，lambda是匿名函数
@hydra.main(config_name='config', config_path='configs')
#config_name='config'：指定默认加载的配置文件名（不含扩展名），这里表示会加载 config.yaml。config_path='configs'指定配置文件所在的目录路径，这里表示配置文件存放在项目根目录下的 configs 文件夹中
def main(config: DictConfig):#类型注解，这里的config就是从上面的hydra中的configs中的内容传入的
    torch.multiprocessing.set_start_method('spawn')#用于设置 PyTorch 多进程操作的启动方法。
    config.test = config.test#布尔值
    model_path = None
    checkpoint_tracking_path = None
    checkpoint_catching_path = None

    # ==========================================================
    # 统一把 checkpoint 路径转成绝对路径
    # ==========================================================
    if config.checkpoint_tracking:
        config.checkpoint_tracking = to_absolute_path(config.checkpoint_tracking)
        checkpoint_tracking_path = config.checkpoint_tracking

    if config.checkpoint_catching:
        config.checkpoint_catching = to_absolute_path(config.checkpoint_catching)
        checkpoint_catching_path = config.checkpoint_catching

    # ==========================================================
    # 为 test 模式准备 model_path
    #
    # 训练时 Catching_Finetune 不再依赖 model_path 判断加载方式，
    # 而是后面直接调用 restore_like_twostage。
    # ==========================================================
    if config.task == 'Tracking':
        model_path = checkpoint_tracking_path

    elif config.task == 'Catching_Finetune':
        # 测试时优先测试 catching checkpoint；
        # 如果没有 catching checkpoint，才退回 tracking checkpoint。
        model_path = checkpoint_catching_path if checkpoint_catching_path else checkpoint_tracking_path

    elif config.task in ['Catching_TwoStage', 'Catching_OneStage']:
        model_path = checkpoint_catching_path
    # use the device for rl
    config.rl_device = f'cuda:{config.device_id}' if config.device_id >= 0 else 'cpu'#根据配置中的 device_id 确定训练使用的计算设备（GPU 或 CPU），并将结果存入 config.rl_device 供后续使用。
    '''# 等价于以下 if-else 语句
    if config.device_id >= 0:
        # 若 device_id 是大于等于 0 的整数，使用对应编号的 GPU
        config.rl_device = f'cuda:{config.device_id}'
    else:
        # 若 device_id 是负数（通常是 -1),使用 CPU
        config.rl_device = 'cpu'
    '''
    config.seed = random.seed(config.seed)
    #seed - set to -1 to choose random seed
    #seed: -1

    #cprint('Start Building the Environment', 'green', attrs=['bold'])
    # Create and wrap the environment
    env_name = 'gym_dcmm/DcmmVecWorld-v0'
    if config.task == 'Tracking':
        task = 'Tracking'
    elif config.task in ['Catching_Finetune', 'Catching_TwoStage', 'Catching_OneStage']:
        task = 'Catching'
    else:
        raise ValueError(f"Unknown task: {config.task}")
    '''
    if config.task == 'Tracking':
        # 若配置中的任务是"Tracking"(跟踪),则task赋值为"Tracking"
        task = 'Tracking'
    else:
        # 若配置中的任务不是"Tracking",则task统一赋值为"Catching"（捕捉）
        task = 'Catching'
    '''
    #print("config.num_envs: ", config.num_envs)
    env = gym.make_vec(env_name, num_envs=int(config.num_envs), vectorization_mode="sync",
                    task=task, camera_name=["top"],
                    render_per_step=False, render_mode ="rgb_array",#"rgb_array" None 训练的时候不渲染，加快速度，测试的时候要把渲染打开
                    object_name = "object",
                    img_size = config.train.ppo.img_dim,
                    imshow_cam = config.imshow_cam, 
                    viewer = config.viewer,
                    print_obs = False, print_info = False,
                    print_reward = False, print_ctrl = False,
                    print_contacts = False, object_eval = config.object_eval,#False
                    env_time = 2.5, steps_per_policy = 20)
    #创建并行强化学习环境
    output_dif = os.path.join('outputs', config.output_name)#output_name: Dcmm
    # Get the local date and time
    local_tz = pytz.timezone('Asia/Shanghai')#创建一个上海时区的对象
    current_datetime = datetime.datetime.now().astimezone(local_tz)#获取上海的时间
    current_datetime_str = current_datetime.strftime("%Y-%m-%d/%H:%M:%S")#带时区的 datetime 对象（current_datetime）转换为 “年 - 月 - 日 / 时：分: 秒” 格式的字符串，用于作为文件夹名称。
    output_dif = os.path.join(output_dif, current_datetime_str)
    os.makedirs(output_dif, exist_ok=True)

    if config.task in ["Tracking", "Catching_Finetune"]:
        PPO = PPO_Track
    elif config.task == "Catching_TwoStage":
        PPO = PPO_Catch_TwoStage
    elif config.task == "Catching_OneStage":
        PPO = PPO_Catch_OneStage
    else:
        raise ValueError(f"Unknown task: {config.task}")
    agent = PPO(env, output_dif, full_config=config)#实例化，env是智能体 “感知世界” 和 “执行动作” 的接口。智能体在训练过程中会将关键数据保存到这个路径

    cprint('Start Training/Testing the Agent', 'green', attrs=['bold'])
    if config.test:#测试还是训练，训练就是训练，测试是看学会多少
        if model_path:
            print("checkpoint loaded")
            agent.restore_test(model_path)
        print("testing")
        agent.test()
    else:
        # connect to wandb
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.output_name,
            config=omegaconf_to_dict(config),
            mode=config.wandb_mode
        )
        '''
        # wandb config
        output_name: Dcmm
        wandb_mode: "disabled"  # "online" | "offline" | "disabled"
        wandb_entity: 'Your_username'
        # wandb_project: 'RL_Dcmm_Track_Random'
        wandb_project: 'RL_Dcmm_Catch_Random'
        '''

        if config.task == 'Catching_Finetune':
            agent.restore_like_twostage(
                checkpoint_tracking=checkpoint_tracking_path,
                checkpoint_catching=checkpoint_catching_path,
            )
        else:
            agent.restore_train(model_path)

        agent.train()
        wandb.finish()

if __name__ == '__main__':
    main()
#python3 train_DCMM.py test=True task=Tracking num_envs=1 checkpoint_tracking= object_eval=True viewer=True imshow_cam=False
#python3 train_DCMM.py test=False task=Tracking num_envs=
#python3 train_DCMM.py test=True task=Catching_OneStage num_envs=1 checkpoint_catching= object_eval=True viewer=True imshow_cam=False
#python3 train_DCMM.py test=False task=Catching_OneStage num_envs=$(number_of_CPUs)
#反向传播为什么不对奖励进行反向传播而是对损失进行反向传播呢，因为奖励是环境给出来的，而且也通过了采样，没有办法进行求导
#而损失虽然也是通过采样得到的，但是反向传播求导的其实是输出的概率，而且在actor损失里面ratio是概率，而后面的优势函数是具体的采样结果，相当于把这两个过程连接起来了。