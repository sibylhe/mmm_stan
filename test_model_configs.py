from pathlib import Path
import pandas as pd
import numpy as np
import unittest
import inspect

import model_configs


class test_control_model(unittest.TestCase):
    pass


class test_diminishing_returns_model(unittest.TestCase):
    pass


class test_media_model(unittest.TestCase):
    def setUp(self):
        testfile = Path(__file__).resolve().parent.joinpath("modelcode_media_sibylhe.txt")
        with open(testfile, "rt") as fo:

            writer = ""
            self.sibylhe_code_blocks = {}
            self.sibylhe_code = []
            for line in fo.readlines():
                self.sibylhe_code.append(line)

                if line.startswith("transformed parameters"):
                    writer = "transformed parameters"
                    self.sibylhe_code_blocks[writer] = [line]
                elif line.startswith("parameters"):
                    writer = "parameters"
                    self.sibylhe_code_blocks[writer] = [line]
                elif line.startswith("data"):
                    writer = "data"
                    self.sibylhe_code_blocks[writer] = [line]
                elif line.startswith("model"):
                    writer = "model"
                    self.sibylhe_code_blocks[writer] = [line]
                elif writer:
                    self.sibylhe_code_blocks[writer].append(line)

        self.default_multigroup_priors = {
            "random_variables": {
                "decay": {"distribution": "beta(3,3)", "constraints": "<lower=0,upper=1>",},
                "peak": {
                    "distribution": "uniform(0, ceil(max_lag/2))",  # .format(P=np.ceil(max_lag / 2)),
                    "constraints": "<lower=0,upper=ceil(max_lag/2)>",
                },
                "tau": {"distribution": "normal(0, 5)", "constraints": ""},
                "beta": {"distribution": "normal(0, 1)", "constraints": "<lower=0>",},
                "noise_var": {
                    "distribution": "inv_gamma(0.05, 0.05 * 0.01)",
                    "constraints": "<lower=0>",
                },
                "decay2": {"distribution": "beta(3,3)", "constraints": "<lower=0,upper=1>",},
                "peak2": {
                    "distribution": "uniform(0, ceil(max_lag/2))",  # .format(P=np.ceil(max_lag / 2)),
                    "constraints": "<lower=0,upper=ceil(max_lag/2)>",
                },
                "beta2": {"distribution": "normal(0, 1)", "constraints": "<upper=0>",},
            },
            "fixed_variables": {"max_lag": 8},
        }


    def test_data_code_block(self):
        """Test data code block, with extra variable groupings"""

        # Single media groups
        config_single = model_configs.Media.default_priors()

        media_model = model_configs.Media(config_single)

        extra_media_groups = [{"group_name": "B", "param_suffix": "b"}]
        code = media_model._data_code_block(extra_media_groups)
        output = inspect.cleandoc(code)
        lines = output.split("\n")
        self.assertEqual(lines[9], r"  // the number of B")
        self.assertEqual(lines[10], r"  int<lower=1> num_b;")
        self.assertEqual(lines[14], r"  matrix[N+max_lag-1, num_b] X_b;")
        self.assertEqual(
            "".join(lines[-5:]),
            r"  // the number of other control variables  int<lower=1> num_ctrl;  // a matrix of control variables  matrix[N, num_ctrl] X_ctrl;}",
        )

    def test_parameters_code_block(self):
        """Test parameters code block, with single and multiple impressions variable groupings"""

        # Single media groups
        config_single = model_configs.Media.default_priors()

        inputs = {
            "beta": {"constraints": "<lower=7>", "distribution": "normal(0, 1)"},
            "decay": {"constraints": "<upper=9>", "distribution": "beta(5,3)"},
            "peak": {
                "constraints": "<lower=0,upper=ceil(max_lag/2)>",
                "distribution": "uniform(7, ceil(max_lag/2))",
            },
        }
        config_single["random_variables"].update(inputs)

        media_model = model_configs.Media(config_single)
        # No input args, just media_model initialised
        code = media_model._parameters_code_block([])

        output = inspect.cleandoc(code)
        lines = output.split("\n")
        self.assertEqual(lines[0], r"parameters {")
        self.assertEqual(lines[2], r"  real<lower=0> noise_var;")
        self.assertEqual(lines[6], r"  vector<lower=7>[num_media+num_ctrl] beta;")
        self.assertEqual(lines[9], r"  vector<upper=9>[num_media] decay;")
        self.assertEqual(lines[10], r"  vector<lower=0,upper=ceil(max_lag/2)>[num_media] peak;")
        self.assertEqual(lines[-1], r"}")

        # Two media groups
        config_multi = self.default_multigroup_priors

        inputs_multi = {
            "beta2": {"constraints": "<lower=1>", "distribution": "normal(1, 1)"},
            "decay2": {"constraints": "<upper=3>", "distribution": "beta(1,1)"},
            "peak2": {
                "constraints": "<lower=0,upper=ceil(max_lag/2)>",
                "distribution": "uniform(0, ceil(max_lag/2))",
            },
        }
        inputs_multi.update(inputs)
        config_multi["random_variables"].update(inputs_multi)

        media_model = model_configs.Media(config_multi)
        extra_media_groups = [{"group_name": "B", "param_suffix": "2"}]
        code = media_model._parameters_code_block(extra_media_groups=extra_media_groups)

        output = inspect.cleandoc(code)
        lines = output.split("\n")
        self.assertEqual(len(lines), 18)
        self.assertEqual(lines[11], r"  // the coefficients for B variables")
        self.assertEqual(lines[12], r"  vector<lower=1>[num_2] beta2;")
        self.assertEqual(lines[15], r"  vector<upper=3>[num_2] decay2;")
        self.assertEqual(lines[16], r"  vector<lower=0,upper=ceil(max_lag/2)>[num_2] peak2;")
        self.assertEqual(lines[-1], r"}")

        # Three media groups
        config_triple = self.default_multigroup_priors

        inputs_triple = {
            "beta3": {"constraints": "<lower=1>", "distribution": "normal(1, 1)"},
            "decay3": {"constraints": "<upper=3>", "distribution": "beta(1,1)"},
            "peak3": {
                "constraints": "<lower=0,upper=ceil(max_lag/2)>",
                "distribution": "uniform(0, ceil(max_lag/2))",
            },
        }
        inputs_triple.update(inputs_multi)
        config_triple["random_variables"].update(inputs_triple)

        media_model = model_configs.Media(config_triple)
        extra_media_groups = [
            {"group_name": "B", "param_suffix": "2"},
            {"group_name": "C", "param_suffix": "3"},
        ]
        code = media_model._parameters_code_block(extra_media_groups=extra_media_groups)

        output = inspect.cleandoc(code)
        lines = output.split("\n")
        self.assertEqual(len(lines), 24)
        self.assertEqual(lines[-2], r"  vector<lower=0,upper=ceil(max_lag/2)>[num_3] peak3;")

    def test_transformed_parameters_code_block(self):
        """Test parameters code block, with single and multiple impressions variable groupings"""

        # Single media group (default sibylhe priors)
        config_multi = model_configs.Media.default_priors()

        media_model = model_configs.Media(config_multi)
        code = media_model._transformed_parameters_code_block([])

        output = inspect.cleandoc(code)
        output_lines = output.split("\n")
        for target, actual in zip(self.sibylhe_code_blocks["transformed parameters"], output_lines):
            target_line = target.replace("\n", "")
            actual_line = actual.replace(" = ", " <- ")  # Legacy symbol for equality
            self.assertEqual(target_line, actual_line)

        # Two media groups
        config_single = self.default_multigroup_priors

        inputs = {
            "beta": {"constraints": "<lower=7>", "distribution": "normal(0, 1)"},
            "decay": {"constraints": "<upper=9>", "distribution": "beta(5,3)"},
            "peak": {
                "constraints": "<lower=0,upper=ceil(max_lag/2)>",
                "distribution": "uniform(7, ceil(max_lag/2))",
            },
        }
        config_single["random_variables"].update(inputs)

        media_model = model_configs.Media(config_single)
        extra_media_groups = [{"group_name": "B", "param_suffix": "2"}]
        code = media_model._transformed_parameters_code_block(extra_media_groups=extra_media_groups)

        output = inspect.cleandoc(code)
        lines = output.split("\n")
        self.assertEqual(lines[5], r"  // the cumulative B effect after adstock")
        self.assertEqual(lines[6], r"  real cum_effect_2;")
        self.assertEqual(lines[12], r"  // matrix of all B predictors")
        self.assertEqual(lines[13], r"  matrix[N, num_2] X2;")
        self.assertEqual(lines[15], r"  row_vector[max_lag] lag_weights2;")
        self.assertEqual(lines[32], r"        lag_weights2[max_lag-lag+1] = pow(decay2[var], (lag - 1 - peak2[var]) ^ 2);")
        self.assertEqual(lines[-1], r"}")

    def test_model_code_block(self):
        """Test parameters code block, with single and multiple impressions variable groupings"""

        # Single media group (default sibylhe priors)
        config_single = self.default_multigroup_priors

        media_model = model_configs.Media(config_single)
        code = media_model._model_code_block()

        output = inspect.cleandoc(code)
        output_lines = output.split("\n")
        for target, actual in zip(self.sibylhe_code_blocks["model"], output_lines):
            target_line = target.replace("\n", "")
            actual_line = actual.replace(" = ", " <- ")  # Legacy symbol for equality
            self.assertEqual(target_line, actual_line)

        # Multiple media group (default sibylhe priors)
        extra_media_groups = [{"group_name": "B", "param_suffix": "2"}]
        config_multi = self.default_multigroup_priors
        inputs = {
            "beta": {"constraints": "<lower=1>", "distribution": "normal(1, 1)"},
            "decay": {"constraints": "<upper=1>", "distribution": "beta(1,1)"},
            "peak": {
                "constraints": "<lower=1,upper=ceil(max_lag/1)>",
                "distribution": "uniform(1, ceil(max_lag/1))",
            },
            "beta2": {"constraints": "<lower=2>", "distribution": "normal(2, 2)"},
            "decay2": {"constraints": "<upper=2>", "distribution": "beta(2,2)"},
            "peak2": {
                "constraints": "<lower=2,upper=ceil(max_lag/2)>",
                "distribution": "uniform(2, ceil(max_lag/2))",
            },
        }
        config_multi["random_variables"].update(inputs)

        media_model = model_configs.Media(config_multi)
        code = media_model._model_code_block(extra_media_groups=extra_media_groups)

        output = inspect.cleandoc(code)
        lines = output.split("\n")
        self.assertEqual(lines[4], r"  decay2 ~ beta(2,2);")
        self.assertEqual(lines[5], r"  peak2 ~ uniform(2, ceil(max_lag/2));")
        self.assertEqual(lines[12], r"   beta2[i] ~ normal(2, 2);")
        self.assertEqual(lines[11], r" for (i in 1 : num_2) {")
        self.assertEqual(lines[15], r"  y ~ normal(tau + X * beta + X2 * beta2, sqrt(noise_var));")
        self.assertEqual(lines[-1], r"}")

    def test_basic(self):
        """Test stan code generates correctly for sibylhe model"""

        c = model_configs.Media.default_priors()
        mod = model_configs.Media(c)

        media_cols = ["mdip_A", "mdsp_A"]
        ctrl_vars = ["base_sales"]
        model_data = {}
        model_data["df"] = pd.DataFrame(
            np.empty((2, len(ctrl_vars + media_cols + ["sales"]))),
            columns=ctrl_vars + media_cols + ["sales"],
            index=[0, 1],
        )
        model_data["df_mmm"] = model_data["df"].copy()

        mdip_cols = [col for col in model_data["df"].columns if "mdip_" in col]
        actual_data, actual_code = mod.model_inputs(
            mdip_cols, ctrl_vars=ctrl_vars, df=model_data["df"], df_mmm=model_data["df_mmm"]
        )

        # '<-' is a to-be deprecated symbol for variable setting in pystan
        target_code = "".join(self.sibylhe_code).replace("<-", "=")

        self.assertMultiLineEqual(target_code, actual_code)

    def test_multiple_media_groups(self):
        """Test data block calculates correctly, for a multiple media group model"""

        media_cols = ["mdip_A", "mdip_B", "mdip_C"]
        ctrl_vars = ["base_sales"]
        data = {
            "base_sales": [1.0, 0.0],
            "mdip_A": [2.0, 4.0],
            "mdip_B": [6.0, 8.0],
            "mdip_C": [12.0, 14.0],
            "sales": [20.0, 22.0],
        }
        mdip_cols = ["mdip_A", "mdip_B"]
        group2_priors = {
            "decayOther": {"distribution": "beta(3,3)", "constraints": "<lower=0,upper=1>",},
            "peakOther": {
                "distribution": "uniform(0, ceil(max_lag/2))",
                "constraints": "<lower=0,upper=ceil(max_lag/2)>",
            },
            "betaOther": {"distribution": "normal(0, 1)", "constraints": "<upper=0>"},
        }
        additional_groups = [{"media_cols": ["mdip_C"], "param_suffix": "Other"}]

        priors = model_configs.Media.default_priors()
        priors["random_variables"].update(group2_priors)
        mod = model_configs.Media(priors)

        model_data = {}
        model_data["df"] = pd.DataFrame(
            data, columns=ctrl_vars + media_cols + ["sales"], index=[0, 1]
        )
        model_data["df_mmm"] = model_data["df"].copy()

        actual_data, actual_code = mod.model_inputs(
            mdip_cols,
            ctrl_vars=ctrl_vars,
            df=model_data["df"],
            df_mmm=model_data["df_mmm"],
            additional_groups=additional_groups,
        )

        expected_data = {
            "N": 2,
            "max_lag": 8,
            "num_media": 2,
            "X_media": np.array(
                [
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [2.0, 6.0],
                    [4.0, 8.0],
                ]
            ),
            "mu_mdip": np.array([3.0, 7.0]),
            "num_ctrl": 1,
            "X_ctrl": np.array([[1.0], [0.0]]),
            "y": np.array([20.0, 22.0]),
            "X_Other": np.array([[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [12.0], [14.0]]),
            "mu_Other": np.array([13.0]),
            "num_Other": 1,
        }

        for array_key in [
            "N",
            "max_lag",
            "num_media",
            "X_media",
            "mu_mdip",
            "num_ctrl",
            "X_ctrl",
            "y",
            "X_Other",
            "mu_Other",
            "num_Other",
        ]:
            self.assertTrue(array_key in actual_data)
            expected = expected_data[array_key]
            actual = actual_data[array_key]

            if isinstance(expected, np.ndarray):
                np.testing.assert_array_equal(actual_data[array_key], expected_data[array_key])
            else:
                self.assertEqual(expected, actual)
        
        try:
            import pystan
            try:
                sm2 = pystan.StanModel(model_code=actual_code, verbose=True)
                fit2 = sm2.sampling(data=actual_data, iter=2000, chains=4)
                fit1_result = fit2.extract()
            except:
                assert False
        except ImportError as ie:
            print(ie)
        

if __name__ == "__main__":
    unittest.main()
