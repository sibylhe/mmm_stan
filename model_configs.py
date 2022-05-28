"""API for building Stan media mix model"""
from copy import deepcopy
import numpy as np
import pandas as pd
import inspect
import sys

from typing import List, Tuple, Union
from abc import ABC, abstractmethod


class BayesianPystanModel(ABC):
    """Abstract base class for """
    def __init__(self, config: Union[dict, None] = None) -> None:
        self._config = self.default_priors()
        if config is not None:
            # overwrite default with supplied config
            self._config.update(config)

        # Flatten model config dict into dict of stan code string variables
        config_priors = {k: v for k, v in self._config["random_variables"].items()}

        self._priors = BayesianPystanModel.flatten_dict(config_priors)

    @abstractmethod
    def default_priors() -> dict:
        pass

    @abstractmethod
    def model_inputs() -> Tuple[dict, str]:
        pass

    @staticmethod
    def flatten_dict(nested_dict: dict) -> dict:
        new_flattened_dict = {}
        for key_1 in nested_dict:
            for key_2 in nested_dict[key_1]:
                key = "{}_{}".format(key_1, key_2)
                new_flattened_dict[key] = nested_dict[key_1][key_2]
        return new_flattened_dict
    
    
class Media(BayesianPystanModel):
    """API for building 'Media mix' layer of Stan media mix model"""
    @staticmethod
    def default_priors():
        """Standard media mix model - sibylhe media mix priors"""
        return {
            "random_variables": {
                "decay": {"distribution": "beta(3,3)", "constraints": "<lower=0,upper=1>",},
                "peak": {
                    "distribution": "uniform(0, ceil(max_lag/2))",
                    "constraints": "<lower=0,upper=ceil(max_lag/2)>",
                },
                "tau": {"distribution": "normal(0, 5)", "constraints": ""},
                "beta": {"distribution": "normal(0, 1)", "constraints": "<lower=0>",},
                "noise_var": {
                    "distribution": "inv_gamma(0.05, 0.05 * 0.01)",
                    "constraints": "<lower=0>",
                },
            },
            "fixed_variables": {"max_lag": 8},
        }

    def model_inputs(
        self,
        mdip_cols: List[str],
        ctrl_vars: List[str],
        df: pd.DataFrame,
        df_mmm: pd.DataFrame,
        additional_groups: List[dict] = [],
    ) -> Tuple[dict, str]:
        """
        Create Stan code and data for a media model, given the shape of media data. 
        Base model is media + baseline. Media variables can have their own priors by specifying 
        additional_groups, and adding their priors to the config of this class.
        
        Example usage: Run a media model in pystan
        >>> config = model_configs.default_config()
        >>> model = model_configs.Media( config )
        >>> model_data, model_code = model.model_inputs(mdip_cols, df, df_mmm)
        >>> sm2 = pystan.StanModel( model_code = model_code2 )
        >>> fit2 = sm2.sampling( data = model_data2, iter=200, chains = 4 )
        
        Example usage: Specify different priors for some media variables. Names of priors must end in the same 
                       suffix as defined in `additional_groups` field. i.e. 'beta2' refers to `df` columns 
                       'mdip_sem',' mdip_sem' and 'mdip_inst'
        >>> media_groups = [ { 'media_cols': [ 'mdip_sem',' mdip_sem', 'mdip_inst' ], 
                               'param_suffix': '2' } ]
        >>> new_priors { "beta2": { "distribution": "beta(3,3)", "constraints": "<lower=0>" },
                         "peak2": { "distribution": "uniform(0,3)", "constraints": "<lower=0>" },
                         "decay2": { "distribution": "beta(5,2)", "constraints": "<lower=0,upper=1>" }
        >>> config[ 'priors' ][ 'random_variables' ].update( new_priors )
        >>> model_data, model_code = model.model_inputs(mdip_cols, df, df_mmm, additional_groups = media_groups)

        :param mdip_cols: List of media impressions variables in 'df' to use in adstock transformation
        :param ctrl_vars: List of control (baseline) variables in 'df', no adstock is applied.
        :param df: Dataframe containing media impressions variables specified in mdip_cols, and optionally in key 
                   'media_cols' of each item in 'additional_groups'.
        :param df_mmm: dataframe contains log 1p transformed 'sales' and 'base_sales'
        :param additional_groups: Optional list of dictionaries containing keys:
                                   * 'media_cols': List of media variable names in 'df' that have priors specified 
                                                   by the suffix in 'param_suffix'
                                   * 'param_suffix': String suffix found at the end of beta, peak & decay names in 
                                                     the input priors that relate to variables in 'media_cols'
                                   * 'group_name': name of the media group (optional, changes comments in stan code only)
        :returns:
            - model_data - dict of data needed by pystan sampoling method
            - model_code - stan code that defines the stan model for the marketing model.
        """

        mu_mdip = df[mdip_cols].apply(np.mean, axis=0).values

        max_lag = self._config["fixed_variables"]["max_lag"]
        num_media = len(mdip_cols)
        
        # padding zero * (max_lag-1) rows
        X_media = np.concatenate((np.zeros((max_lag - 1, num_media)), df[mdip_cols].values), axis=0)
        n_ctrl_vars = len(ctrl_vars)
        X_ctrl = df_mmm[ctrl_vars].values.reshape(len(df), n_ctrl_vars)
        n_observations = df.shape[0]
        model_data = {
            "N": n_observations,
            "max_lag": max_lag,
            "num_media": num_media,
            "X_media": X_media,
            "mu_mdip": mu_mdip,
            "num_ctrl": n_ctrl_vars,
            "X_ctrl": X_ctrl,
            "y": df_mmm["sales"].values,
        }

        # Optionally, add parameters & priors specific to a group of media variables
        if additional_groups:

            for ii, group in enumerate(additional_groups):

                # Set the group name - used in stan code comments only
                if "group_name" not in group:
                    group["group_name"] = f"media group {ii + 2}"

                media_cols = group["media_cols"]
                parameter_suffix = group["param_suffix"]

                mu_vars = df[media_cols].apply(np.mean, axis=0).values
                num_in_group = len(media_cols)
                X_comp = np.concatenate(
                    (np.zeros((max_lag - 1, num_in_group)), df[media_cols].values), axis=0
                )
                model_data.update(
                    {
                        f"X_{parameter_suffix}": X_comp,
                        f"mu_{parameter_suffix}": mu_vars,
                        f"num_{parameter_suffix}": num_in_group,
                    }
                )

        # Stan code block: FUNCTIONS
        functions_block = self._functions_code_block()

        # Stan code block: DATA
        data_block = self._data_code_block(additional_groups)

        # Stan code block: PARAMETERS
        parameters_block = self._parameters_code_block(additional_groups)

        # Stan code block: TRANSFORMED PARAMETERS
        transformed_parameters_block = self._transformed_parameters_code_block(additional_groups)

        # Stan code block: MODEL
        model_block = self._model_code_block(additional_groups)

        model_blocks = (
            functions_block
            + data_block
            + parameters_block
            + transformed_parameters_block
            + model_block
        )

        model_code_dedented = inspect.cleandoc(model_blocks)
        model_code = model_code_dedented.rstrip("\n").lstrip("\n")

        return model_data, model_code

    def _functions_code_block(self):
        return """
        functions {
          // the adstock transformation with a vector of weights
          real Adstock(vector t, row_vector weights) {
            return dot_product(t, weights) / sum(weights);
          }
        }"""

    def _data_code_block(self, extra_media_groups: List[dict] = []) -> str:

        data_block_variables = dict()

        data_block_variables["num_vars"] = ""
        data_block_variables["data_matrix_vars"] = ""
        data_block_variables["data_vector_mean_vars"] = ""
        if len(extra_media_groups) > 0:

            num_vars, data_matrix_vars, data_vector_mean_vars = "", "", ""
            for param_group in extra_media_groups:
                param_suffix = param_group["param_suffix"]
                group_name = param_group["group_name"]

                num_vars += f"""
          // the number of {group_name}
          int<lower=1> num_{param_suffix};"""

                data_matrix_vars += f"""
          // matrix of {group_name} variables
          matrix[N+max_lag-1, num_{param_suffix}] X_{param_suffix};"""

                data_vector_mean_vars += f"""
          // vector of {group_name} variables' mean
          real mu_{param_suffix}[num_{param_suffix}];"""

            data_block_variables.update(
                {
                    "num_vars": num_vars,
                    "data_matrix_vars": data_matrix_vars,
                    "data_vector_mean_vars": data_vector_mean_vars,
                }
            )

        data_block = """
        data {{
          // the total number of observations
          int<lower=1> N;
          // the vector of sales
          real y[N];
          // the maximum duration of lag effect, in weeks
          int<lower=1> max_lag;
          // the number of media channels
          int<lower=1> num_media;{num_vars}
          // matrix of media variables
          matrix[N+max_lag-1, num_media] X_media;{data_matrix_vars}
          // vector of media variables' mean
          real mu_mdip[num_media];{data_vector_mean_vars}
          // the number of other control variables
          int<lower=1> num_ctrl;
          // a matrix of control variables
          matrix[N, num_ctrl] X_ctrl;
        }}""".format(
            **data_block_variables
        )

        return data_block

    def _parameters_code_block(self, extra_media_groups: List[dict]) -> str:
        """If only one group is specified, variables do not have suffixes"""

        residual_intercept = """
        parameters {{
          // residual variance
          real{noise_var_constraints} noise_var;
          // the intercept
          real tau;""".format(
            **self._priors
        )

        parameters_block_variables = self._priors.copy()
        parameters_block_variables["residual_intercept"] = residual_intercept
        parameters_block_variables["tail"] = """}"""
        parameters_block_variables["other_variables"] = ""

        if len(extra_media_groups) > 0:
            # generate new parameters for additional media groups
            variables_blocks = ""
            for param_group in extra_media_groups:
                name_injections = {
                    "beta_name": "beta%s" % param_group["param_suffix"],
                    "decay_name": "decay%s" % param_group["param_suffix"],
                    "peak_name": "peak%s" % param_group["param_suffix"],
                    "group_name": param_group["group_name"],
                    "param_name_suffix": param_group["param_suffix"],
                    "beta_constraints": self._priors[
                        "beta%s_constraints" % param_group["param_suffix"]
                    ],
                    "decay_constraints": self._priors[
                        "decay%s_constraints" % param_group["param_suffix"]
                    ],
                    "peak_constraints": self._priors[
                        "peak%s_constraints" % param_group["param_suffix"]
                    ],
                }
                block = """\
          // the coefficients for {group_name} variables
          vector{beta_constraints}[num_{param_name_suffix}] {beta_name};
          // the decay and peak parameter for the adstock transformation of
          // each {group_name} variable
          vector{decay_constraints}[num_{param_name_suffix}] {decay_name};
          vector{peak_constraints}[num_{param_name_suffix}] {peak_name};""".format(
                    **name_injections
                )
                variables_blocks += "\n" + block
            parameters_block_variables["other_variables"] = variables_blocks

        parameters_block = """{residual_intercept}
          // the coefficients for media variables and base sales
          vector{beta_constraints}[num_media+num_ctrl] beta;
          // the decay and peak parameter for the adstock transformation of
          // each media
          vector{decay_constraints}[num_media] decay;
          vector{peak_constraints}[num_media] peak;{other_variables}
        {tail}""".format(
            **parameters_block_variables
        )
        return parameters_block

    def _transformed_parameters_code_block(self, extra_media_groups: List[dict]):
        transformed_parameters_variables = self._priors.copy()

        transformed_parameters_variables.update(
            {
                "adstocked_variables": "",
                "predictors_matrix": "",
                "lag_weights": "",
                "transform_block": "",
            }
        )

        if len(extra_media_groups) > 0:

            adstocked_variables, predictors_matrix, lag_weights, transform_block = "", "", "", ""
            for param_group in extra_media_groups:
                param_suffix = param_group["param_suffix"]
                group_name = param_group["group_name"]

                adstocked_variables += f"""
          // the cumulative {group_name} effect after adstock
          real cum_effect_{param_suffix};
          // matrix of {group_name} variables after adstock
          matrix[N, num_{param_suffix}] X_{param_suffix}_adstocked;"""
                predictors_matrix += f"""
          // matrix of all {group_name} predictors
          matrix[N, num_{param_suffix}] X{param_suffix};"""
                lag_weights += f"""
          // adstock, mean-center, log1p transformation {group_name}
          row_vector[max_lag] lag_weights{param_suffix};"""
                transform_block += """
          // Transform adstock for {group_name}, into X{param_suffix}
          for (nn in 1:N) {{
            for (var in 1 : num_{param_suffix}) {{
              for (lag in 1 : max_lag) {{
                lag_weights{param_suffix}[max_lag-lag+1] = pow(decay{param_suffix}[var], (lag - 1 - peak{param_suffix}[var]) ^ 2);
              }}
             cum_effect_{param_suffix} = Adstock(sub_col(X_{param_suffix}, nn, var, max_lag), lag_weights{param_suffix});
             X_{param_suffix}_adstocked[nn, var] = log1p(cum_effect_{param_suffix}/mu_{param_suffix}[var]);
            }}
          X{param_suffix} = X_{param_suffix}_adstocked;
          }} """.format(
                    **param_group
                )
            transformed_parameters_variables.update(
                {
                    "adstocked_variables": adstocked_variables,
                    "predictors_matrix": predictors_matrix,
                    "lag_weights": lag_weights,
                    "transform_block": transform_block,
                }
            )

        transformed_parameters = """
        transformed parameters {{
          // the cumulative media effect after adstock
          real cum_effect;
          // matrix of media variables after adstock
          matrix[N, num_media] X_media_adstocked;{adstocked_variables}
          // matrix of all predictors
          matrix[N, num_media+num_ctrl] X;
          {predictors_matrix}{lag_weights}
          // adstock, mean-center, log1p transformation
          row_vector[max_lag] lag_weights;
          for (nn in 1:N) {{
            for (media in 1 : num_media) {{
              for (lag in 1 : max_lag) {{
                lag_weights[max_lag-lag+1] = pow(decay[media], (lag - 1 - peak[media]) ^ 2);
              }}
             cum_effect = Adstock(sub_col(X_media, nn, media, max_lag), lag_weights);
             X_media_adstocked[nn, media] = log1p(cum_effect/mu_mdip[media]);
            }}
          X = append_col(X_media_adstocked, X_ctrl);
          }} {transform_block}
        }}""".format(
            **transformed_parameters_variables
        )
        return transformed_parameters

    def _model_code_block(self, extra_media_groups: List[dict] = []):
        model_block_variables = self._priors.copy()
        model_block_variables["decay_peak_var_distribution"] = ""
        model_block_variables["beta_var_distribution"] = ""
        model_block_variables["xbeta_var_predictors"] = ""

        if len(extra_media_groups) > 0:
            decay_peak_var_distribution, beta_var_distribution, xbeta_var_predictors = "", "", ""
            for param_group in extra_media_groups:
                param_suffix = param_group["param_suffix"]
                group_name = param_group["group_name"]

                decay_distribution = self._priors[f'decay{param_suffix}_distribution']
                peak_distribution = self._priors[f'peak{param_suffix}_distribution']
                decay_peak_var_distribution = f"""
          // decay and peak for {group_name}
          decay{param_suffix} ~ {decay_distribution};
          peak{param_suffix} ~ {peak_distribution};"""

                beta_distribution = self._priors[f'beta{param_suffix}_distribution']
                beta_var_distribution = """
         // beta for {group_name}
         for (i in 1 : num_{param_suffix}) {{
           beta{param_suffix}[i] ~ {beta_distribution};
         }}""".format(
                    **param_group, beta_distribution = beta_distribution
                )

                xbeta_var_predictors = f""" + X{param_suffix} * beta{param_suffix}"""

                model_block_variables.update(
                    {
                        "xbeta_var_predictors": xbeta_var_predictors,
                        "beta_var_distribution": beta_var_distribution,
                        "decay_peak_var_distribution": decay_peak_var_distribution,
                    }
                )

        model_block = """
        model {{
          decay ~ {decay_distribution};
          peak ~ {peak_distribution};{decay_peak_var_distribution}
          tau ~ {tau_distribution};
          for (i in 1 : num_media+num_ctrl) {{
            beta[i] ~ {beta_distribution};
          }}{beta_var_distribution}
          noise_var ~ {noise_var_distribution};
          y ~ normal(tau + X * beta{xbeta_var_predictors}, sqrt(noise_var));
        }}
        """.format(
            **model_block_variables
        )
        return model_block


class Control(BayesianPystanModel):
    @staticmethod
    def default_priors():
        """Standard Control model - sibylhe Control model priors"""
        return {
            "random_variables": {
                "beta1": {"distribution": "normal(0, 1)", "constraints": "<lower=0>"},
                "beta2": {"distribution": "normal(0, 1)", "constraints": None},
                "alpha": {"distribution": None, "constraints": "<lower=0, upper=max_intercept>",},
                "noise_var": {
                    "distribution": "inv_gamma(0.05, 0.05 * 0.01)",
                    "constraints": "<lower=0>",
                },
            }
        }
    
    def model_inputs(self) -> Tuple[dict, str]:
        raise NotImplementedError


class DiminishingReturn(BayesianPystanModel):
    @staticmethod
    def default_priors():
        return {
            "random_variables": {
                "slope": {"distribution": "gamma(3, 1)", "constraints": "<lower=0>"},
                "ec": {"distribution": "beta(2, 2)", "constraints": "<lower=0,upper=1>"},
                "beta_hill": {"distribution": "normal(0, 1)", "constraints": "<lower=0>",},
                "noise_var": {
                    "distribution": "inv_gamma(0.05, 0.05 * 0.01)",
                    "constraints": "<lower=0>",
                },
            }
        }

    def model_inputs(self) -> Tuple[dict, str]:
        raise NotImplementedError


def default_config():
    """Default config for the Bayesian Media Mix model stacking Base, Media & Diminshing returns model"""
    return {
        "model": {
            "info": {"name": "", "description": ""},
            "control_model_config": Control.default_priors(),
            "media_model_config": {
                "priors": Media.default_priors(),
                "additional_param_groups": [],
            },
            "diminsihing_return_model_config": DiminishingReturn.default_priors(),
        },
    }


def scrub(x: dict) -> Union[dict,str]:
    # Converts None to empty string
    ret = deepcopy(x)
    # Handle dictionaries, lits & tuples. Scrub all values
    if isinstance(x, dict):
        for key, value in ret.items():
            ret[key] = scrub(value)
    if isinstance(x, (list, tuple)):
        for index, value in enumerate(ret):
            ret[index] = scrub(value)
    # Handle None
    if x is None:
        ret = ""
    # Finished scrubbing
    return ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="model configs", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--default-config",
        action="store_true",
        help="Output a default config for the media mix modelling pipeline",
    )
    parser.add_argument(
        "--validate-config",
        help="Check a configs values are the right type, given path to a config.json",
    )
    args = parser.parse_args()

    if args.default_config:
        import json

        config = default_config()
        json.dump(scrub(config), sys.stdout, sort_keys=True, separators=(",", ": "), indent=4)
        sys.stdout.write("\n")
        sys.exit(0)

    if args.validate_config:
        import json

        fo = open(args.validate_config, "rt")
        config = json.load(fo)

        for field in config["data"]["keep_media_vars"]:
            # plain names only
            if field.startswith("mdip_") or field.startswith("mdsp_"):
                sys.stderr.write("Plain names only for media vars to keep")
                sys.exit(1)
            # underscore reserved for prefixes
            if "_" in field.partition("_")[2]:
                sys.stderr.write("Underscores in variable names is reserved for prefixes only")
                sys.exit(1)
