**To compare 2 media models**

1. First, save current media model to examples folder;
	```linux
	cp mmm1.json examples/mmm_A.json
	```
2. Specify a config file:
	```linux
	python -c "import json, model_configs; print(json.dumps( model_configs.default_config() ))" > "config.json"
	```
3. Edit the config file manually, to assign new priors to a group of media variables.

4. Validate the config:
	```linux
	path/to/python mmm_stan.py --validate-config config.json
	```
5. Run the model pipeline with config:
	```linux
	path/to/python mmm_stan.py --config config.json
	```
_(WIP: currently it will fail after saving the media model output.)_

6. Save the new model
	```linux
	cp mmm1.json examples/mmm_B.json
	```

7. Compare two models in [notebook](media.ipynb): `examples/media.ipynb`


**Platform**
```python
>>> python -c "import platform; print(platform.platform())"
Linux-5.10.16.3-microsoft-standard-WSL2-x86_64-with-debian-bullseye-sid
```
**Conda environment:** 
[Conda environment specification](stan_env.txt)