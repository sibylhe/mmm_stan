from pydantic import BaseModel, validator


class StanMediaConfigValidator(BaseModel):
    config: dict

    @validator("config")
    @classmethod
    def parameter_naming(cls, config: dict):

        priors = config["priors"]
        for k in priors.keys():
            if k not in ["random_variables", "fixed_variables"]:
                raise KeyError(
                    (
                        "Allowed keys are 'random_variables' and 'fixed_variables'. "
                        "See model_configs.Media.default_priors for more info"
                    )
                )
        print('Prior keys: All good!')
        
        if len(config["additional_param_groups"]) > 0:
            # normal media group plus an additional new one
            n_media_groups = len(config["additional_param_groups"]) + 1

            for grp in config["additional_param_groups"]:
                param_suffix = grp["param_suffix"]

                random_variable_prior_names = priors["random_variables"].keys()
                if f"beta{param_suffix}" not in random_variable_prior_names:
                    raise ValueError(
                        "media groups must have parameters named with suffix specified in additional_param_groups"
                        "i.e. if beta2 is in the priors, param suffix 2 must be in additional param groups\n"
                        f"found parameter suffix: {param_suffix} was not a suffix of parameter names: {','.join(random_variable_prior_names)}"
                    )
            print('param group naming: All good!')

    @validator("config")
    @classmethod
    def schema_check(cls, config: dict):
        pass
