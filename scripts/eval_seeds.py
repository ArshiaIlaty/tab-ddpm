import argparse
import subprocess
import tempfile
import lib
import os
import shutil
from pathlib import Path
from copy import deepcopy
from scripts.eval_catboost import train_catboost
from scripts.eval_mlp import train_mlp
from scripts.eval_simple import train_simple

pipeline = {
   'ddpm': 'scripts/pipeline.py',
   'smote': 'smote/pipeline_smote.py',
   'ctabgan': 'CTAB-GAN/pipeline_ctabgan.py',
   'ctabgan-plus': 'CTAB-GAN-Plus/pipeline_ctabgan.py',
   'tvae': 'CTGAN/pipeline_tvae.py'
}

def eval_seeds(
   raw_config,
   n_seeds,
   eval_type,
   sampling_method="ddpm",
   model_type="catboost",
   n_datasets=1,
   dump=True,
   change_val=False
):
   metrics_seeds_report = lib.SeedsMetricsReport()
   parent_dir = Path(raw_config["parent_dir"])

   if eval_type == 'real':
       n_datasets = 1

   temp_config = deepcopy(raw_config)
   with tempfile.TemporaryDirectory() as dir_:
       dir_ = Path(dir_)
       temp_config["parent_dir"] = str(dir_)
       if sampling_method == "ddpm":
           shutil.copy2(parent_dir / "model.pt", temp_config["parent_dir"])
       elif sampling_method in ["ctabgan", "ctabgan-plus"]:
           shutil.copy2(parent_dir / "ctabgan.obj", temp_config["parent_dir"])
       elif sampling_method == "tvae":
           shutil.copy2(parent_dir / "tvae.obj", temp_config["parent_dir"])

       for sample_seed in range(n_datasets):
           temp_config['sample']['seed'] = sample_seed
           lib.dump_config(temp_config, dir_ / "config.toml")
           if eval_type != 'real' and n_datasets > 1:
               subprocess.run(['python3.9', f'{pipeline[sampling_method]}', '--config', f'{str(dir_ / "config.toml")}', '--sample'], check=True)

           T_dict = deepcopy(raw_config['eval']['T'])
           for seed in range(n_seeds):
               print(f'**Eval Iter: {sample_seed*n_seeds + (seed + 1)}/{n_seeds * n_datasets}**')
               if model_type == "catboost":
                   T_dict["normalization"] = None
                   T_dict["cat_encoding"] = None
                   metric_report = train_catboost(
                       parent_dir=temp_config['parent_dir'],
                       real_data_path=temp_config['real_data_path'],
                       eval_type=eval_type,
                       T_dict=T_dict,
                       seed=seed,
                       change_val=change_val
                   )
               elif model_type == "mlp":
                   T_dict["normalization"] = "quantile"
                   T_dict["cat_encoding"] = "one-hot"
                   metric_report = train_mlp(
                       parent_dir=temp_config['parent_dir'],
                       real_data_path=temp_config['real_data_path'],
                       eval_type=eval_type,
                       T_dict=T_dict,
                       seed=seed,
                       change_val=change_val
                   )
               elif model_type == "simple":
                   T_dict["normalization"] = "minmax"
                   T_dict["cat_encoding"] = None
                   metric_report = train_simple(
                       parent_dir=temp_config['parent_dir'],
                       real_data_path=temp_config['real_data_path'],
                       eval_type=eval_type,
                       T_dict=T_dict,
                       seed=seed,
                       change_val=change_val
                   )

               metrics_seeds_report.add_report(metric_report)

   metrics_seeds_report.get_mean_std()
   res = metrics_seeds_report.print_result()
   if os.path.exists(parent_dir/ f"eval_{model_type}.json"):
       eval_dict = lib.load_json(parent_dir / f"eval_{model_type}.json")
       eval_dict = eval_dict | {eval_type: res}
   else:
       eval_dict = {eval_type: res}
   
   if dump:
       lib.dump_json(eval_dict, parent_dir / f"eval_{model_type}.json")

   raw_config['sample']['seed'] = 0
   lib.dump_config(raw_config, parent_dir / 'config.toml')
   return res

def main():
   parser = argparse.ArgumentParser()
   parser.add_argument('--config', metavar='FILE')
   parser.add_argument('n_seeds', type=int, default=10)
   parser.add_argument('sampling_method', type=str, default="ddpm")
   parser.add_argument('eval_type',  type=str, default='synthetic')
   parser.add_argument('model_type',  type=str, default='catboost')
   parser.add_argument('n_datasets', type=int, default=1)
   parser.add_argument('--no_dump', action='store_false',  default=True)

   args = parser.parse_args()
   raw_config = lib.load_config(args.config)
   eval_seeds(
       raw_config,
       n_seeds=args.n_seeds,
       sampling_method=args.sampling_method,
       eval_type=args.eval_type,
       model_type=args.model_type,
       n_datasets=args.n_datasets,
       dump=args.no_dump
   )

if __name__ == '__main__':
   main()
