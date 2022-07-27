from utils import create_sweep
import pandas as pd
from argparse import ArgumentParser
import os.path

data_transform_config = {
    "data__method_name": {
        "value": "real_data"
    },
    "n_iter": {
        "value": "auto",
    },
}

benchmarks = [{"task": "regression",
                   "dataset_size": "medium",
                   "categorical": False,
                   "datasets":  ["cpu_act",
                     "pol",
                     "elevators",
                     "isolet",
                     "wine_quality",
                      "Ailerons",
                      "houses",
                      "house_16H",
                      "diamonds",
                      "Brazilian_houses",
                      "Bike_Sharing_Demand",
                      "nyc-taxi-green-dec-2016",
                      "house_sales",
                      "sulfur",
                      "medical_charges",
                      "MiamiHousing2016",
                      "superconduct",
                      "california",
                      "year",
                      "fifa"]},

                {"task": "regression",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["diamonds",
                                  "nyc-taxi-green-dec-2016",
                                 "year"]},

                {"task": "classif",
                    "dataset_size": "medium",
                    "categorical": False,
                    "datasets": ["electricity",
                                 "covertype",
                                 "pol",
                                 "house_16H",
                                 "kdd_ipums_la_97-small",
                                 "MagicTelescope",
                                 "bank-marketing",
                                 "phoneme",
                                 "MiniBooNE",
                                 "Higgs",
                                 "eye_movements",
                                 "jannis",
                                 "credit",
                                 "california",
                                 "wine"]
                 },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": False,
                    "datasets": ["covertype",
                                 "MiniBooNE",
                                 "Higgs",
                                 "jannis"],
                 },

                {"task": "regression",
                    "dataset_size": "medium",
                    "categorical": True,
                 "datasets": ["yprop_4_1",
                             "analcatdata_supreme",
                             "visualizing_soil",
                             "black_friday",
                             "nyc-taxi-green-dec-2016",
                             "diamonds",
                             "Mercedes_Benz_Greener_Manufacturing",
                             "Brazilian_houses",
                             "Bike_Sharing_Demand",
                             "OnlineNewsPopularity",
                             "house_sales",
                             "particulate-matter-ukair-2017",
                             "SGEMM_GPU_kernel_performance"]},

                {"task": "regression",
                 "dataset_size": "large",
                 "categorical": True,
                 "datasets": ["black_friday",
                     "nyc-taxi-green-dec-2016",
                     "diamonds",
                     "particulate-matter-ukair-2017",
                     "SGEMM_GPU_kernel_performance"]},

                {"task": "classif",
                    "dataset_size": "medium",
                    "categorical": True,
                    "datasets": ["electricity",
                                 "eye_movements",
                                  "KDDCup09_upselling",
                                  "covertype",
                                  "rl",
                                  "road-safety",
                                  "compass"]
                 },

                {"task": "classif",
                    "dataset_size": "large",
                    "categorical": True,
                    "datasets": ["covertype",
                                 "road-safety"]
                 }
]

models = ["gbt", "rf", "xgb", "hgbt",
          "ft_transformer", "resnet", "mlp", "saint"]

if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument('projects', metavar='WANDB_PROJECT_NAMES', type=str, nargs='+', help='WANDB project name')
    parser.add_argument('path', metavar='output_dir', type=str, help='sweep id table output dir')
    args = parser.parse_args()

    sweep_ids = []
    names = []
    projects = []
    for i, benchmark in enumerate(benchmarks):
        for model_name in models:
            for default in [True, False]:
                name = f"{model_name}_{benchmark['task']}_{benchmark['dataset_size']}"
                if benchmark['categorical']:
                    name += "_categorical"
                else:
                    name += "_numerical"
                if default:
                    name += "_default"
                project_name = args.projects[i] if len(args.projects) > 1 else args.projects[0]
                sweep_id = create_sweep(data_transform_config,
                             model_name=model_name,
                             regression=benchmark["task"] == "regression",
                             categorical=benchmark["categorical"],
                             dataset_size = benchmark["dataset_size"],
                             datasets = benchmark["datasets"],
                             default=default,
                             project=project_name,
                             name=name)
                sweep_ids.append(sweep_id)
                names.append(name)
                projects.append(project_name)
                print(f"Created sweep {name}")
                print(f"Sweep id: {sweep_id}")
                print(f"In project {project_name}")

    df = pd.DataFrame({"sweep_id": sweep_ids, "name": names, "project":projects})
    
    output_filepath = os.path.join(args.path, 'benchmark_sweeps.csv')
    df.to_csv(output_filepath, index=False)
    print(f"Check the sweeps id saved at {output_filepath}")