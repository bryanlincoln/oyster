"""
Launcher for experiments with PEARL

"""
import os
import pathlib
import numpy as np
import click
import json
import torch
from datetime import datetime

from tensorboardX import SummaryWriter

from rlkit.envs import ENVS
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.torch.networks import FlattenMlp, MlpEncoder, RecurrentEncoder, Mlp
from rlkit.torch.sac.sac import PEARLSoftActorCritic
from rlkit.torch.sac.agent import PEARLAgent
from rlkit.launchers.launcher_util import setup_logger
import rlkit.torch.pytorch_util as ptu
from configs.default import default_config

from rlkit.core import eval_util


def experiment(variant):
    # debugging triggers a lot of printing and logs to a debug directory
    DEBUG = variant["util_params"]["debug"]
    os.environ["DEBUG"] = str(int(DEBUG))

    # create logging directory
    # TODO support Docker
    exp_id = "debug" if DEBUG else None
    experiment_log_dir = setup_logger(
        variant["env_name"],
        variant=variant,
        exp_id=exp_id,
        base_log_dir=variant["util_params"]["base_log_dir"],
    )

    writer = SummaryWriter(
        comment="TaskEmbeddings",
        logdir=os.path.join(
            variant["path_to_weights"] if variant["path_to_weights"] is not None else experiment_log_dir,
            "embeddings",
        ),
    )

    # create multi-task environment and sample tasks
    env = NormalizedBoxEnv(ENVS[variant["env_name"]](**variant["env_params"]))
    tasks = env.get_all_task_idx()
    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(env.action_space.shape))
    reward_dim = 1

    # instantiate networks
    latent_dim = variant["latent_size"]
    context_encoder_input_dim = (
        2 * obs_dim + action_dim + reward_dim
        if variant["algo_params"]["use_next_obs_in_context"]
        else obs_dim + action_dim + reward_dim
    )
    context_encoder_output_dim = (
        latent_dim * 2 if variant["algo_params"]["use_information_bottleneck"] else latent_dim
    )
    net_size = variant["net_size"]
    recurrent = variant["algo_params"]["recurrent"]
    encoder_model = RecurrentEncoder if recurrent else MlpEncoder

    context_encoder = encoder_model(
        hidden_sizes=[200, 200, 200],
        input_size=context_encoder_input_dim,
        output_size=context_encoder_output_dim,
        dropout=variant["context_params"]["use_dropout"],
        dropout_p=variant["context_params"]["dropout_p"],
    )

    # decoder takes in encoder output (latent state) and outputs context
    if variant["curiosity_params"]["use_curiosity"]:
        if variant["curiosity_params"]["pred_next_obs"]:
            # (task, batch, obs + act + z) -> decoder -> (task, batch, next_obs)
            context_decoder = Mlp(
                hidden_sizes=[200, 200, 200],
                input_size=obs_dim + latent_dim + action_dim,
                output_size=obs_dim,
            )
        else:
            # (task, batch, latent context) -> decoder -> (task, batch, obs + act + rew)
            context_decoder = Mlp(
                hidden_sizes=[200, 200, 200],
                input_size=context_encoder_output_dim,
                output_size=context_encoder_input_dim,
            )
    else:
        context_decoder = None

    qf1 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    qf2 = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size],
        input_size=obs_dim + action_dim + latent_dim,
        output_size=1,
    )
    vf = FlattenMlp(
        hidden_sizes=[net_size, net_size, net_size], input_size=obs_dim + latent_dim, output_size=1,
    )
    policy = TanhGaussianPolicy(
        hidden_sizes=[net_size, net_size, net_size],
        obs_dim=obs_dim + latent_dim,
        latent_dim=latent_dim,
        action_dim=action_dim,
    )
    agent = PEARLAgent(latent_dim, context_encoder, context_decoder, policy, **variant["algo_params"])
    algorithm = PEARLSoftActorCritic(
        env=env,
        train_tasks=list(tasks[: variant["n_train_tasks"]]),
        eval_tasks=list(tasks[-variant["n_eval_tasks"] :]),
        nets=[agent, qf1, qf2, vf],
        latent_dim=latent_dim,
        use_curiosity=variant["curiosity_params"]["use_curiosity"],
        pred_next_obs=variant["curiosity_params"]["pred_next_obs"],
        curiosity_eta=variant["curiosity_params"]["eta"],
        add_intrinic_reward=variant["curiosity_params"]["add_intrinic_reward"],
        fwd_lr=variant["curiosity_params"]["fwd_lr"],
        tbwriter=writer,
        embedding_plotter=eval_util.make_embedding_plotter(experiment_log_dir),
        **variant["algo_params"]
    )

    # optionally load pre-trained weights
    if variant["path_to_weights"] is not None:
        path = variant["path_to_weights"]
        context_encoder.load_state_dict(torch.load(os.path.join(path, "context_encoder.pth")))
        qf1.load_state_dict(torch.load(os.path.join(path, "qf1.pth")))
        qf2.load_state_dict(torch.load(os.path.join(path, "qf2.pth")))
        vf.load_state_dict(torch.load(os.path.join(path, "vf.pth")))
        algorithm.target_vf.load_state_dict(torch.load(os.path.join(path, "target_vf.pth")))
        policy.load_state_dict(torch.load(os.path.join(path, "policy.pth")))

    # optional GPU mode
    ptu.set_gpu_mode(variant["util_params"]["use_gpu"], variant["util_params"]["gpu_id"])
    if ptu.gpu_enabled():
        algorithm.to()

    # optionally save eval trajectories as pkl files
    if variant["algo_params"]["dump_eval_paths"]:
        pickle_dir = experiment_log_dir + "/eval_trajectories"
        pathlib.Path(pickle_dir).mkdir(parents=True, exist_ok=True)

    # run the algorithm
    algorithm.train()

    writer.close()


def deep_update_dict(fr, to):
    """ update dict of dicts with new values """
    # assume dicts have same keys
    for k, v in fr.items():
        if type(v) is dict:
            deep_update_dict(v, to[k])
        else:
            to[k] = v
    return to


@click.command()
@click.argument("config", default=None)
@click.option("--gpu", default=None)
@click.option("--docker", is_flag=True, default=False)
@click.option("--debug", is_flag=True, default=False)
@click.option("--path_to_weights", default=None)
def main(config, gpu, docker, debug, path_to_weights):

    variant = default_config
    if config:
        with open(os.path.join(config)) as f:
            exp_params = json.load(f)
        variant = deep_update_dict(exp_params, variant)
    if gpu is not None:
        variant["util_params"]["gpu_id"] = gpu
        variant["util_params"]["use_gpu"] = True
    if path_to_weights is not None:
        variant["path_to_weights"] = path_to_weights

    variant["algo_params"]["experiment_name"] = config.split("/")[-1].split(".")[:-1]

    experiment(variant)


if __name__ == "__main__":
    main()

